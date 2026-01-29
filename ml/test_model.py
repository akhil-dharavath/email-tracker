import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
import numpy as np
import os
import sys

# Import our custom modules
from fusion_model import FusionModel
from feature_extractor import FeatureExtractor

def predict(model, tokenizer, feature_extractor, text):
    # 1. Extract Features
    feats = feature_extractor.extract(text)
    feats_tensor = torch.from_numpy(feats).unsqueeze(0) # (1, 6)
    
    # 2. Tokenize
    encoding = tokenizer(
        text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 3. Predict
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'], feats_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        emotion_idx = torch.argmax(probs).item()
        
    emotions = ['Angry', 'Anxious', 'Neutral', 'Happy']
    predicted_emotion = emotions[emotion_idx]
    confidence = probs[0][emotion_idx].item()
    
    return predicted_emotion, confidence

def main():
    print("--- Model Verification Script ---")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(base_dir), 'fusion_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python ml/train_pipeline.py' first.")
        return

    # Load Components
    try:
        device = torch.device('cpu')
        model = FusionModel(num_classes=4, behavioral_dim=6)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        feature_extractor = FeatureExtractor()
        
    except Exception as e:
        print(f"Error loading model or components: {e}")
        return

    # Test Samples
    test_samples = [
        "I am absolutely furious about this error!",
        "I need help urgently, I am worried about the deadline.",
        "Just sending the files you asked for.",
        "Thank you so much, this is great news!"
    ]
    
    print("\nRunning inference on test samples:\n")
    for text in test_samples:
        emotion, conf = predict(model, tokenizer, feature_extractor, text)
        print(f"Text: '{text}'")
        print(f"Prediction: {emotion} (Confidence: {conf:.2f})\n")
        
    # Interactive Mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("Enter text to test (type 'exit' to quit):")
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit': break
            emotion, conf = predict(model, tokenizer, feature_extractor, user_input)
            print(f"Prediction: {emotion} (Confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
