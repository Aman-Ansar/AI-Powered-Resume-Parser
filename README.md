# AI-Powered Resume Parser 🔍

An intelligent tool that automatically extracts skills, experience, and education from resumes (PDFs) using NLP, ranks candidates based on job descriptions, and exports parsed data to JSON/CSV formats.


## Features 🚀

- **PDF Resume Parsing**: Extract text from PDF resumes with layout preservation
- **Smart Information Extraction**:
  - Skills detection using Custom NER
  - Education degree recognition
  - Work experience duration extraction
- **AI-Powered Ranking**: TF-IDF based ranking system against job descriptions
- **Multi-Resume Processing**: Analyze multiple resumes simultaneously
- **Export Capabilities**: Download analysis in JSON and CSV formats
- **Web Interface**: User-friendly Streamlit dashboard

## Installation ⚙️

1. **Clone Repository**:
```bash
git clone https://github.com/yourusername/resume-parser.git
cd resume-parser
```
2. **Install dependency**:
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
2. **Run Application**:
```
   streamlit run app.py
```
4. **Requirements**
```
Python >= 3.8
spacy == 3.5.0
pdfplumber == 0.9.0
streamlit == 1.22.0
scikit-learn == 1.2.2
```
