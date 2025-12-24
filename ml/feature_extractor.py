import re
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # We can implement more complex logic here if needed, e.g. using spacy
        pass

    def extract(self, text, timestamp=None):
        """
        Extracts behavioral features from email text.
        Returns a numpy array of features.
        """
        features = []
        
        # 1. Punctuation Frequency combined
        # Exclamation marks
        exclamations = text.count('!')
        # Question marks
        questions = text.count('?')
        features.append(exclamations)
        features.append(questions)

        # 2. Capitalization Ratio
        # Avoid division by zero
        caps_count = sum(1 for c in text if c.isupper())
        total_len = len(text) if len(text) > 0 else 1
        caps_ratio = caps_count / total_len
        features.append(caps_ratio)

        # 3. Email Length (char count)
        features.append(len(text))
        
        # 4. Sentence Analysis (Basic split by .)
        sentences = [s for s in text.split('.') if s.strip()]
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        features.append(avg_sentence_len)

        # 5. Time-based features (if timestamp provided)
        # Assuming timestamp is a datetime object or similar. 
        # For simplicity in this prototype, we'll default to 0 (business hours) if not provided.
        # Feature: 1 if off-hours (before 9am or after 5pm), 0 otherwise
        is_off_hours = 0
        if timestamp:
            hour = timestamp.hour
            if hour < 9 or hour >= 17:
                is_off_hours = 1
        features.append(is_off_hours)

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self):
        return 6
