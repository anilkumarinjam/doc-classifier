# app/models/classifier.py
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re
from pathlib import Path
from app.utils.document_parser import extract_text_from_file

class DocumentClassifier:
    def __init__(self, model_path="trained_model.joblib"):
        self.document_types = [
            "Stock Purchase Agreement", 
            "Certificate of Incorporation", 
            "Investors' Rights Agreement"
        ]
        
        # Map folder names to document types
        self.folder_to_type = {
            "stock_purchase": "Stock Purchase Agreement",
            "certificate_incorporation": "Certificate of Incorporation",
            "investors_rights": "Investors' Rights Agreement"
        }
        
        # Create a baseline pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            ('clf', LogisticRegression(max_iter=1000, C=10))
        ])
        
        # Try to load pre-trained model, or train a new one if it doesn't exist
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.pipeline = joblib.load(model_path)
        else:
            print("No pre-trained model found. Training new model from examples...")
            self.train_from_examples()
            self.save_model(model_path)
    
    def train_from_examples(self, examples_dir="data/examples"):
        """Train the classifier using example documents in the data directory."""
        texts = []
        labels = []
        
        # Iterate through each document type directory
        for folder, doc_type in self.folder_to_type.items():
            folder_path = Path(examples_dir) / folder
            if not folder_path.exists():
                print(f"Warning: Example folder {folder_path} not found")
                continue
                
            # Process each document in the folder
            for file_path in folder_path.glob("*.*"):
                if file_path.suffix.lower() in ['.pdf', '.doc', '.docx']:
                    try:
                        print(f"Processing {file_path}")
                        text = extract_text_from_file(str(file_path))
                        texts.append(text)
                        labels.append(doc_type)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        if not texts:
            print("No training documents found. Using fallback rule-based classification.")
            return False
            
        # Train the model
        print(f"Training model with {len(texts)} documents")
        self.pipeline.fit(texts, labels)
        return True
    
    def save_model(self, model_path):
        """Save the trained model to the specified path."""
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to {model_path}")
    
    def predict(self, text):
        """Predict the document type for the given text."""
        # If the pipeline has a predict_proba method (i.e., it's trained)
        if hasattr(self.pipeline, 'predict_proba'):
            # Get the predicted class and probability
            doc_type = self.pipeline.predict([text])[0]
            probabilities = self.pipeline.predict_proba([text])[0]
            
            # Get the index of the predicted class
            class_idx = list(self.pipeline.classes_).index(doc_type)
            confidence = probabilities[class_idx] * 100
            
            return doc_type, confidence
        else:
            # Fallback to rule-based classification
            return self._rule_based_classify(text)
    
    def _rule_based_classify(self, text):
        """Fallback rule-based classification method."""
        # Patterns to look for in each document type
        patterns = {
            "Stock Purchase Agreement": [
                r"stock\s+purchase\s+agreement",
                r"purchase\s+price\s+per\s+share",
                r"representations\s+and\s+warranties\s+of\s+the\s+(company|purchaser)",
                r"sale\s+and\s+issuance\s+of\s+securities",
                r"closing\s+conditions"
            ],
            "Certificate of Incorporation": [
                r"certificate\s+of\s+incorporation",
                r"article\s+[IVX]+",
                r"authoriz(ed|e)\s+capital\s+stock",
                r"registered\s+office",
                r"corporate\s+existence"
            ],
            "Investors' Rights Agreement": [
                r"investors['']?\s+rights\s+agreement",
                r"registration\s+rights",
                r"information\s+rights",
                r"right\s+of\s+first\s+refusal",
                r"drag-along\s+rights"
            ]
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count pattern matches for each document type
        scores = {}
        for doc_type, doc_patterns in patterns.items():
            score = 0
            for pattern in doc_patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 2
            
            # Additional keyword scoring
            if doc_type == "Stock Purchase Agreement":
                if "purchase" in text_lower and "stock" in text_lower:
                    score += 1
                if "shares" in text_lower and "purchase" in text_lower:
                    score += 1
            
            elif doc_type == "Certificate of Incorporation":
                if "certificate" in text_lower and "incorporate" in text_lower:
                    score += 1
                if "bylaws" in text_lower:
                    score += 1
            
            elif doc_type == "Investors' Rights Agreement":
                if "investors" in text_lower and "rights" in text_lower:
                    score += 1
                if "information rights" in text_lower:
                    score += 1
            
            scores[doc_type] = score
        
        # Get the document type with the highest score
        if all(score == 0 for score in scores.values()):
            return "Unknown Document", 0
        
        best_match = max(scores.items(), key=lambda x: x[1])
        document_type = best_match[0]
        
        # Calculate confidence as a percentage
        total_score = sum(scores.values())
        confidence = (best_match[1] / total_score) * 100 if total_score > 0 else 0
        
        return document_type, confidence