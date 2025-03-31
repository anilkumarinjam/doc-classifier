from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import tempfile
from pathlib import Path
import uvicorn

from app.models.classifier import DocumentClassifier
from app.utils.document_parser import extract_text_from_file

app = FastAPI(title="Document Classifier")

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize the classifier
classifier = DocumentClassifier()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify/", response_class=HTMLResponse)
async def classify_document(request: Request, file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp:
        # Write the uploaded file content to the temporary file
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    
    try:
        # Extract text from the document
        text = extract_text_from_file(temp_path)
        
        # Classify the document
        document_type, confidence = classifier.predict(text)
        
        # Return the classification result
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request, 
                "document_type": document_type,
                "confidence": f"{confidence:.2f}%",
                "filename": file.filename
            }
        )
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
