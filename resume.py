import pdfplumber
import spacy
import re
import json
import csv
import os
import tempfile
import streamlit as st
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import StringIO

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self._add_custom_patterns()
        self.skill_pattern = re.compile(r'([A-Z][a-z]*\s*[A-Z][a-z]*+)', re.DOTALL)
        
    def _add_custom_patterns(self):
        if "entity_ruler" not in self.nlp.pipe_names:
            config = {
                "phrase_matcher_attr": None,
                "validate": True,
                "overwrite_ents": True
            }
            self.nlp.add_pipe("entity_ruler", config=config, before="ner")
        ruler = self.nlp.get_pipe("entity_ruler")
        patterns = [
            {"label": "DEGREE", "pattern": "BS"},
            {"label": "DEGREE", "pattern": "B.Sc"},
            {"label": "DEGREE", "pattern": "Bachelors"},
            {"label": "DEGREE", "pattern": "Masters"},
            {"label": "DEGREE", "pattern": "PhD"},
            {"label": "SKILL", "pattern": "Python"},
            {"label": "SKILL", "pattern": "Machine Learning"},
            {"label": "SKILL", "pattern": "SQL"},
        ]
        ruler.add_patterns(patterns)

    def extract_text(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            return '\n'.join([page.extract_text() for page in pdf.pages])

    def _clean_text(self, text):
        text = re.sub('\s+', ' ', text)
        text = re.sub('[^a-zA-Z0-9\s\.\-,]', '', text)
        return text.strip()

    def parse_resume(self, pdf_path):
        text = self.extract_text(pdf_path)
        text = self._clean_text(text)
        doc = self.nlp(text)
        
        return {
            'skills': list(set([ent.text for ent in doc.ents if ent.label_ == "SKILL"])),
            'education': list(set([ent.text for ent in doc.ents if ent.label_ == "DEGREE"])),
            'experience': self._extract_experience(text),
            'raw_text': text
        }

    def _extract_experience(self, text):
        pattern = r'(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\s\-,]*\d{4})[\s\–\-]+(present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\s\-,]*\d{4})'
        return [match.group().strip() for match in re.finditer(pattern, text)]

    def rank_resumes(self, resumes, job_description):
        vectorizer = TfidfVectorizer()
        job_vec = vectorizer.fit_transform([job_description])
        
        scores = []
        for resume in resumes:
            resume_vec = vectorizer.transform([resume['raw_text']])
            score = cosine_similarity(job_vec, resume_vec)[0][0]
            scores.append(score)
        
        ranked_resumes = sorted(zip(resumes, scores), 
                          key=lambda x: x[1], 
                          reverse=True)
        return ranked_resumes

def main():
    st.title("AI-Powered Resume Analyzer")
    st.markdown("Upload multiple resumes (PDFs) and get detailed analysis")

    parser = ResumeParser()
    
    uploaded_files = st.file_uploader("Choose PDF resumes", type="pdf", accept_multiple_files=True)
    job_description = st.text_area("Paste Job Description (for ranking)", height=150)

    if uploaded_files and st.button("Analyze Resumes"):
        resumes = []
        with st.spinner("Processing resumes..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    data = parser.parse_resume(tmp_file.name)
                    data['name'] = os.path.splitext(uploaded_file.name)[0]
                    resumes.append(data)
                os.unlink(tmp_file.name)

        if job_description.strip():
            ranked_resumes = parser.rank_resumes(resumes, job_description)
            st.subheader("Ranking Results")
            for idx, (resume, score) in enumerate(ranked_resumes, 1):
                st.markdown(f"**{idx}. {resume['name']}** (Score: {score:.2f})")

        st.subheader("Resume Analysis")
        for resume in resumes:
            with st.expander(f"📄 {resume['name']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Skills**")
                    for skill in resume['skills']:
                        st.markdown(f"- {skill}")
                    
                with col2:
                    st.markdown("**Education**")
                    for edu in resume['education']:
                        st.markdown(f"- {edu}")
                    
                with col3:
                    st.markdown("**Experience**")
                    for exp in resume['experience']:
                        st.markdown(f"- {exp}")
                
                st.markdown("**Download Analysis**")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    json_data = json.dumps(resume, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{resume['name']}_analysis.json",
                        mime="application/json"
                    )
                
                with col_export2:
                    csv_file = StringIO()
                    writer = csv.writer(csv_file)
                    writer.writerow(['Skills', 'Education', 'Experience'])
                    writer.writerow([
                        ', '.join(resume['skills']),
                        ', '.join(resume['education']),
                        ', '.join(resume['experience'])
                    ])
                    st.download_button(
                        label="Download CSV",
                        data=csv_file.getvalue(),
                        file_name=f"{resume['name']}_analysis.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()