import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import torch.nn.functional as F

from fusion_model import FusionModel
from feature_extractor import FeatureExtractor

# --- Synthetic Data Generation ---
class SyntheticEmailDataset(Dataset):
    def __init__(self, size=200, tokenizer=None):
        self.tokenizer = tokenizer
        self.feature_extractor = FeatureExtractor()
        self.data = []
        
        emotions = ['Angry', 'Anxious', 'Neutral', 'Happy']
        
        print(f"Generating {size} synthetic samples...")
        for _ in range(size):
            label_idx = random.randint(0, 3)
            label_name = emotions[label_idx]
            
            # Simple text generation based on label
            if label_name == 'Angry':
                texts = [
                    "I am extremely furious about this delay! Fix it now.",
                    "Why is this broken again? This is unacceptable.",
                    "I demand a refund immediately!",
                    "You are wasting my time with these errors."
                ]
                text = random.choice(texts)
            elif label_name == 'Anxious':
                texts = [
                    "I am worried that we might miss the deadline. Please check.",
                    "Is the server stable? I'm getting nervous.",
                    "Urgent: Potential data breach?",
                    "Can you please confirm if this is correct? I'm unsure."
                ]
                text = random.choice(texts)
            elif label_name == 'Happy':
                texts = [
                    "Great job on the project! I am very happy with the results.",
                    "The new feature looks amazing. Thanks!",
                    "Congratulations on the launch!",
                    "Everything is working perfectly."
                ]
                text = random.choice(texts)
            else: # Neutral
                texts = [
                    "Here is the report you requested.",
                    "Meeting starts at 10 AM.",
                    "Please find attached the invoice.",
                    "Just checking in on the status."
                ]
                text = random.choice(texts)
            
            # Extract features
            feats = self.feature_extractor.extract(text)
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            self.data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'features': torch.tensor(feats, dtype=torch.float32),
                'label': torch.tensor(label_idx, dtype=torch.long)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: ROC-AUC calc failed (likely missing classes in batch): {e}")
        roc_auc = 0.0

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }

def train():
    print("Initializing Training Pipeline...")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create Dataset (Increased size for stats)
    full_dataset = SyntheticEmailDataset(size=200, tokenizer=tokenizer)
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f"Data Split: {train_size} Training, {test_size} Testing")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model
    model = FusionModel(num_classes=4, behavioral_dim=6).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    epochs = 5 # Increased epochs slightly
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluation
    print("\n--- Evaluation on Test Set ---")
    metrics = evaluate(model, test_loader, device)
    
    report = f"""
Model Performance Report
------------------------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
ROC-AUC:   {metrics['roc_auc']:.4f}

Confusion Matrix:
{metrics['confusion_matrix']}
"""
    print(report)
    
    # Save Report
    with open("ml_metrics.txt", "w") as f:
        f.write(report)
        
    # Save Model
    save_path = "fusion_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()

