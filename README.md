# Document Classifier

A web application that classifies legal documents into three categories:
- Stock Purchase Agreement
- Certificate of Incorporation
- Investors' Rights Agreement

## Features

- Upload PDF or Word documents (.pdf, .doc, .docx)
- Document text extraction
- Classification with confidence score
- Clean, responsive web interface

## Installation

### Option 1: Using Docker

```bash
# Build the Docker image
docker build -t document-classifier .

# Run the container
docker run -p 8000:8000 document-classifier
```

### Option 2: Local Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Upload a legal document (PDF or Word format)
3. View the classification result

## Technical Details

The application uses a rule-based approach to classify documents by looking for patterns and keywords that are characteristic of each document type. A more sophisticated approach would involve:

1. Training a machine learning model on a corpus of labeled legal documents
2. Using NLP techniques for feature extraction
3. Fine-tuning the model for optimal performance

## Project Structure

```
document-classifier/
│
├── app/                    # Main application code
│   ├── main.py            # FastAPI application
│   ├── models/            # Classification models
│   ├── utils/             # Utility functions
│   └── static/            # CSS and other static files
│
├── templates/              # HTML templates
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

## Future Improvements

- Implement machine learning-based classification
- Add support for more document types
- Provide more detailed analysis of document contents
- Implement user authentication and document history