# Legal Document Classifier

A web application that automatically classifies legal documents into three categories using machine learning and rule-based approaches:
- Stock Purchase Agreement
- Certificate of Incorporation
- Investors' Rights Agreement

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Using the Application](#using-the-application)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Overview

This application helps legal professionals, investors, and business managers identify and organize legal documents by their type. It accepts PDF and Word documents, extracts text content, and uses a trained machine learning model to classify them accurately.

## Features

- **Document Upload**: Support for PDF, DOC, and DOCX file formats
- **Hybrid Classification**: Uses both machine learning and rule-based approaches
- **Training Capability**: Can be trained on your own document examples
- **Confidence Scoring**: Provides confidence level for each classification
- **Responsive UI**: Clean, intuitive web interface
- **Docker Support**: Easy deployment with containerization

## Technology Stack

- **Backend**: Python with FastAPI
- **Document Processing**: PyPDF2 (PDF) and mammoth (Word)
- **Machine Learning**: scikit-learn with TF-IDF vectorization and Logistic Regression
- **Frontend**: HTML, CSS, Jinja2 templates
- **Deployment**: Docker and Uvicorn

## Project Structure

```
document-classifier/
│
├── app/                    # Main application code
│   ├── __init__.py
│   ├── main.py             # FastAPI application entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py   # Document classification model
│   ├── utils/
│   │   ├── __init__.py
│   │   └── document_parser.py  # PDF/DOCX text extraction
│   └── static/
│       └── style.css       # Application styling
│
├── data/                   # Training data directory
│   └── examples/           # Example documents for training
│       ├── stock_purchase/
│       ├── certificate_incorporation/
│       └── investors_rights/
│
├── templates/              # HTML templates
│   ├── index.html          # Upload page
│   └── result.html         # Classification results page
│
├── train_model.py          # Script to train the classifier
├── trained_model.joblib    # Saved trained model (generated after training)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

## Installation

### Option 1: Local Installation

1. **Clone the repository or create the project structure manually**

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare training data (optional but recommended)**
   ```bash
   python train_model.py
   ```
   This will create the necessary directory structure for your training documents.
   Place your example documents in the appropriate folders, then run the script again.

### Option 2: Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t document-classifier .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 document-classifier
   ```

## Training the Model

### Preparing Training Data

1. **Organize your sample documents**
   Place your example documents in the following directories:
   - `data/examples/stock_purchase/` - Stock Purchase Agreements
   - `data/examples/certificate_incorporation/` - Certificates of Incorporation
   - `data/examples/investors_rights/` - Investors' Rights Agreements

2. **Document requirements**
   - Each folder should contain at least one document
   - Supported formats: PDF (.pdf), Word (.doc, .docx)
   - Documents should be representative of their type

### Training Process

1. **Run the training script**
   ```bash
   python train_model.py
   ```

2. **What happens during training**
   - The script extracts text from all documents in the example folders
   - It creates feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency)
   - A Logistic Regression model is trained on these features
   - The trained model is saved to `trained_model.joblib`

3. **Verification**
   The script will output the number of documents processed and confirm when training is complete.

## Using the Application

1. **Start the application**
   ```bash
   uvicorn app.main:app --reload
   ```
   The application will be available at `http://localhost:8000`

2. **Upload a document**
   - Click "Choose a file" to select a PDF or Word document
   - Click "Classify Document" to submit

3. **View results**
   - The application will display the detected document type
   - A confidence score shows how certain the classification is
   - Use "Classify Another Document" to process more files

## How It Works

### Text Extraction

The application extracts plain text from uploaded documents:
- For PDFs: Uses PyPDF2 to extract text content from each page
- For Word documents: Uses mammoth to convert DOCX/DOC to plain text

### Classification Approaches

The system uses a hybrid approach to classification:

1. **Machine Learning Classification (Primary)**
   - **Feature Extraction**: Converts document text into TF-IDF feature vectors
   - **Model**: Logistic Regression trained on example documents
   - **Prediction**: New documents are vectorized and classified by the trained model
   - **Confidence**: Based on prediction probabilities from the model

2. **Rule-Based Classification (Fallback)**
   - Used when no trained model is available or confidence is low
   - Searches for specific patterns and keywords typical of each document type
   - Assigns scores based on pattern matches and keyword occurrences
   - Determines document type based on highest score

### Model Details

- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Captures important words and phrases in each document type
  - Reduces importance of common terms that appear across all documents
  - Uses n-grams (1-3 words) to capture legal phrases

- **Classification Algorithm**: Logistic Regression
  - Fast and efficient for text classification
  - Provides probability estimates for confidence scoring
  - Works well even with relatively small training sets

## Troubleshooting

### Common Issues

1. **Installation problems**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - For uvicorn issues, try: `python -m uvicorn app.main:app --reload`

2. **Document extraction failures**
   - Ensure PDF documents are text-based (not scanned images)
   - For Word documents, try saving in the alternate format (DOC vs DOCX)

3. **Classification accuracy issues**
   - Add more training examples for each document type
   - Try to use high-quality, complete documents for training
   - Check that documents are properly extracted by examining the text content

4. **Import errors**
   - Ensure you have the correct project structure with all __init__.py files
   - Run the application from the project root directory

### Logs and Debugging

- Check console output for error messages
- For training issues, the script will output which documents were processed
- For classification issues, try comparing the extracted text with the original document

## Future Improvements

- Support for additional document types
- More advanced NLP features (named entity recognition, semantic analysis)
- User accounts and document history
- Batch processing of multiple documents
- API endpoints for integration with other systems
- Fine-tuning options for the classification model
- Document visualization and comparison tools

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.