import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

class DocumentClassifier:
    def __init__(self, model_path=None):
        self.document_types = [
            "Stock Purchase Agreement", 
            "Certificate of Incorporation", 
            "Investors' Rights Agreement"
        ]
        
        # Patterns to look for in each document type
        self.patterns = {
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
        
        # If a pre-trained model is provided, load it
        if model_path and os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
        else:
            # Create a baseline pipeline with TF-IDF and Logistic Regression
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
                ('clf', LogisticRegression(max_iter=1000, C=10))
            ])
            # Note: In a real application, you would train this pipeline with labeled data
    
    def train(self, texts, labels):
        """Train the classifier with document texts and their labels."""
        self.pipeline.fit(texts, labels)
    
    def save_model(self, model_path):
        """Save the trained model to the specified path."""
        joblib.dump(self.pipeline, model_path)
    
    def predict(self, text):
        """
        Predict the document type for the given text.
        
        This is a rule-based implementation that looks for patterns
        characteristic of each document type. In a production environment,
        you'd want to replace this with your trained model prediction.
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count pattern matches for each document type
        scores = {}
        for doc_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 2  # Weight for exact pattern matches
            
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
        
        # Calculate confidence as a percentage (simplified)
        total_score = sum(scores.values())
        confidence = (best_match[1] / total_score) * 100 if total_score > 0 else 0
        
        return document_type, confidence
