# train_model.py
import os
from app.models.classifier import DocumentClassifier
from app.utils.document_parser import extract_text_from_file
from pathlib import Path

def preprocess_examples():
    """Organize and preprocess the example documents for training."""
    base_dir = Path("data/examples")
    
    # Create directories if they don't exist
    for folder in ["stock_purchase", "certificate_incorporation", "investors_rights"]:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)
    
    print("Document preprocessing complete. Place your sample documents in the appropriate folders:")
    print(f"- Stock Purchase Agreements: {base_dir / 'stock_purchase'}")
    print(f"- Certificates of Incorporation: {base_dir / 'certificate_incorporation'}")
    print(f"- Investors' Rights Agreements: {base_dir / 'investors_rights'}")
    print("\nThen run this script again to train the model.")

def train_model():
    """Train and save the document classifier model."""
    # Check if example folders exist and contain documents
    base_dir = Path("data/examples")
    if not base_dir.exists():
        print("Example directory not found. Creating directories first.")
        preprocess_examples()
        return
    
    # Count documents in each folder
    doc_counts = {}
    for folder in ["stock_purchase", "certificate_incorporation", "investors_rights"]:
        folder_path = base_dir / folder
        if folder_path.exists():
            docs = list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.doc")) + list(folder_path.glob("*.docx"))
            doc_counts[folder] = len(docs)
        else:
            doc_counts[folder] = 0
    
    # Check if we have enough documents
    if sum(doc_counts.values()) < 3:
        print("Not enough training documents found.")
        preprocess_examples()
        return
    
    print(f"Found {sum(doc_counts.values())} training documents:")
    for folder, count in doc_counts.items():
        print(f"- {folder}: {count} document(s)")
    
    # Train the model
    print("\nTraining document classifier...")
    classifier = DocumentClassifier()
    trained = classifier.train_from_examples()
    
    if trained:
        classifier.save_model("trained_model.joblib")
        print("\nTraining complete! The model is ready to use.")
    else:
        print("\nTraining failed. Please check that your example documents are properly formatted.")

if __name__ == "__main__":
    train_model()