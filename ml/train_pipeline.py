import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import numpy as np
import random
import os

from fusion_model import FusionModel
from feature_extractor import FeatureExtractor

# --- Synthetic Data Generation ---
class SyntheticEmailDataset(Dataset):
    def __init__(self, size=100, tokenizer=None):
        self.tokenizer = tokenizer
        self.feature_extractor = FeatureExtractor()
        self.data = []
        
        emotions = ['Angry', 'Anxious', 'Neutral', 'Happy']
        
        for _ in range(size):
            label_idx = random.randint(0, 3)
            label_name = emotions[label_idx]
            
            # Simple text generation based on label
            if label_name == 'Angry':
                text = "I am extremely furious about this delay! Fix it now."
            elif label_name == 'Anxious':
                text = "I am worried that we might miss the deadline. Please check."
            elif label_name == 'Happy':
                text = "Great job on the project! I am very happy with the results."
            else:
                text = "Here is the report you requested. Let me know if you have questions."
            
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

def train():
    print("Initializing Training Pipeline...")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create Dataset
    print("Generating synthetic dataset...")
    train_dataset = SyntheticEmailDataset(size=50, tokenizer=tokenizer) # Small size for prototype
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Model
    model = FusionModel(num_classes=4, behavioral_dim=6).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Low LR for fine-tuning
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    epochs = 2
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
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
        
    # Save Model
    save_path = "fusion_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()
