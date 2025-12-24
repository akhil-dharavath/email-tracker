from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import torch
import numpy as np

# Add ml folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../ml'))

from fusion_model import FusionModel
from feature_extractor import FeatureExtractor
from summarizer import EmailSummarizer
from transformers import DistilBertTokenizer

from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) # Enable CORS for all routes (since Vercel domain is dynamic/unknown yet)

# Load Models
print("Loading models...")
device = torch.device('cpu') # Use CPU for inference for simplicity/compatibility
model = FusionModel(num_classes=4, behavioral_dim=6)

# Try to load trained weights, else use random (for prototype)
weights_path = os.path.join(os.path.dirname(__file__), '../fusion_model.pth')
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Loaded trained model weights.")
else:
    print("Warning: No trained weights found. Using random weights.")

model.eval()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
feature_extractor = FeatureExtractor()
summarizer = EmailSummarizer()

EMOTIONS = ['Angry', 'Anxious', 'Neutral', 'Happy']

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_emails():
    """
    Analyzes a list of emails.
    Input JSON: { "emails": [ { "id": 1, "subject": "...", "body": "...", "timestamp": "...", "sender": "..." } ] }
    """
    data = request.json
    emails = data.get('emails', [])
    results = []

    for email in emails:
        text = email.get('body', '') + " " + email.get('subject', '')
        
        # 1. Classification (Emotion)
        # Tokenize
        encoding = tokenizer(
            text,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Features
        feats = feature_extractor.extract(text) # Add timestamp logic if passed
        # feats is a numpy array (6,)
        # We need (1, 6)
        feats_tensor = torch.from_numpy(feats).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'], feats_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            emotion_idx = torch.argmax(probs).item()
            emotion = EMOTIONS[emotion_idx]
            emotion_conf = probs[0][emotion_idx].item()
            
            # 2. Urgency Score
            # Heuristic calculation based on Emotion + Features
            # Angry(0)/Anxious(1) -> High Urgency
            # Happy(3)/Neutral(2) -> Low Urgency
            
            base_urgency = 0.0
            if emotion == 'Angry': base_urgency += 0.8
            elif emotion == 'Anxious': base_urgency += 0.7
            elif emotion == 'Happy': base_urgency += 0.2
            else: base_urgency += 0.1
            
            # Boost from features: exclamations, caps
            # feats = [exclam, quest, caps_ratio, len, sent_len, off_hours]
            # Normalize rudimentary:
            urgency_boost = (feats[0] * 0.1) + (feats[2] * 0.5) + (feats[5] * 0.3)
            
            final_urgency = float(min(1.0, base_urgency + urgency_boost))
            
        # 3. Summarization (On demand or pre-computed? Let's pre-compute for prototype)
        # Only summarize if text is long enough
        summary = ""
        if len(text.split()) > 30:
            summary = summarizer.summarize(text)
        else:
            summary = text[:100] + "..."

        results.append({
            "id": email.get('id'),
            "emotion": emotion,
            "urgency_score": round(final_urgency, 2),
            "summary": summary
        })

    return jsonify({"results": results})

@app.route('/api/sync', methods=['POST'])
def sync_gmail():
    """
    Fetches emails from Gmail and runs analysis.
    User must have credentials.json in backend/ directory.
    """
    from gmail_client import GmailClient
    
    from gmail_client import GmailClient
    from dotenv import load_dotenv
    import json
    
    # Load .env
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    try:
        # 1. Try Config from Environment
        client_config = None
        token_data = None
        
        if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
             client_config = {
                "installed": {
                    "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                    "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
                    "project_id": os.getenv('GOOGLE_PROJECT_ID'),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": ["http://localhost"]
                }
             }
        
        if os.getenv('GOOGLE_TOKEN_JSON'):
            try:
                token_data = json.loads(os.getenv('GOOGLE_TOKEN_JSON'))
            except Exception as e:
                print(f"Error parsing GOOGLE_TOKEN_JSON from env: {e}")
        
        # 2. Config File Paths (Fallback & Persistence)
        base_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(base_dir)
        
        def find_config_path(filename):
            path = os.path.join(base_dir, filename)
            if os.path.exists(path): return path
            path = os.path.join(project_root, filename)
            if os.path.exists(path): return path
            return os.path.join(base_dir, filename)

        creds_path = find_config_path('credentials.json')
        token_path = find_config_path('token.json')
        
        client = GmailClient(
            client_config=client_config, 
            token_data=token_data,
            credentials_path=creds_path, 
            token_path=token_path
        )
        # This will trigger browser auth on server side if not token exists
        # For a local app, this opens a browser window.
        raw_emails = client.fetch_emails(max_results=10)
        
        # Check if empty
        if not raw_emails:
            return jsonify({"status": "no_emails", "message": "No emails found or auth failed."})
            
        # Analyze them using our existing logic (re-using the logic would be cleaner, but let's just call the function logic or do a recursive call)
        # To avoid code duplication, let's just repackage as a request to /analyze or call internal function.
        # Faster: just helper function.
        
        # We need to construct the analyze payload
        # But wait, we can just return these emails to frontend and let frontend call analyze? 
        # Or better, analyze here and return results.
        
        # Let's verify we can analyze them.
        # We need to reuse the loop in 'analyze_emails'. 
        # Refactoring 'analyze_emails' to a helper function would be best practice, but for minimum diff size:
        # We create a mock request context or just duplicate the small loop.
        
        results = []
        for email in raw_emails:
            text = email.get('body', '') + " " + email.get('subject', '')
            
            # 1. Classification
            encoding = tokenizer(text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
            feats = feature_extractor.extract(text)
            feats_tensor = torch.from_numpy(feats).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(encoding['input_ids'], encoding['attention_mask'], feats_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                emotion_idx = torch.argmax(probs).item()
                emotion = EMOTIONS[emotion_idx]
                
                base_urgency = 0.0
                if emotion == 'Angry': base_urgency += 0.8
                elif emotion == 'Anxious': base_urgency += 0.7
                elif emotion == 'Happy': base_urgency += 0.2
                else: base_urgency += 0.1
                
                urgency_boost = (feats[0] * 0.1) + (feats[2] * 0.5) + (feats[5] * 0.3)
                final_urgency = float(min(1.0, base_urgency + urgency_boost))
            
            summary = ""
            if len(text.split()) > 30:
                summary = summarizer.summarize(text)
            else:
                summary = text[:100] + "..."

            results.append({
                "id": email.get('id'),
                "subject": email.get('subject'),
                "body": email.get('body'),
                "sender": email.get('sender'),
                "timestamp": email.get('timestamp'),
                "isUnread": email.get('isUnread', False),
                "emotion": emotion,
                "urgency_score": round(final_urgency, 2),
                "summary": summary
            })
            
        return jsonify({"results": results})
        
    except Exception as e:
        print(f"Gmail Sync Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9000)
