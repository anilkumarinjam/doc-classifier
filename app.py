from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from docx import Document
import PyPDF2
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# BERT model and tokenizer for document classification (this part is for later use)
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to upload files
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        document_type = classify_document(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": f"File successfully uploaded", "document_type": document_type}), 200
    else:
        return jsonify({"error": "Invalid file type. Only PDF and DOCX are allowed."}), 400

# Function to extract text from DOCX
def extract_text_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to extract text from PDF
def extract_text_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

# Function to classify document type using BERT (initial model)
def classify_document(file_path):
    # Extract text based on file type
    if file_path.endswith('.docx'):
        text = extract_text_docx(file_path)
    elif file_path.endswith('.pdf'):
        text = extract_text_pdf(file_path)
    else:
        return "Unsupported file type"

    # Preprocess the text for BERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()

    # Mapping predictions to document types
    if prediction == 0:
        return "Stock Purchase Agreement (SPA)"
    elif prediction == 1:
        return "Certificate of Incorporation (COI)"
    elif prediction == 2:
        return "Investors' Rights Agreement (IRA)"
    else:
        return "Unknown Document Type"

if __name__ == '__main__':
    app.run(debug=True)
