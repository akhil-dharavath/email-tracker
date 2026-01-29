import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import numpy as np
import random
import os
import pandas as pd

from fusion_model import FusionModel
from feature_extractor import FeatureExtractor

# --- Data Generation / Loading ---
class EmailDataset(Dataset):
    def __init__(self, csv_file=None, size=100, tokenizer=None):
        """
        Args:
            csv_file (str, optional): Path to csv file with 'text' and 'label' columns.
            size (int): Size of synthetic dataset if csv_file is not provided.
            tokenizer: DistilBertTokenizer
        """
        self.tokenizer = tokenizer
        self.feature_extractor = FeatureExtractor()
        self.data = []
        
        emotions = ['Angry', 'Anxious', 'Neutral', 'Happy']
        
        if csv_file and os.path.exists(csv_file):
            print(f"Loading data from {csv_file}...")
            try:
                df = pd.read_csv(csv_file)
                # Ensure columns existence
                if 'text' in df.columns and 'label' in df.columns:
                    for idx, row in df.iterrows():
                        text = str(row['text'])
                        label_idx = int(row['label'])
                        self._process_item(text, label_idx)
                    print(f"Loaded {len(self.data)} samples from CSV.")
                else:
                    print("Error: CSV must contain 'text' and 'label' columns. Fallback to synthetic.")
                    self._generate_synthetic(size, emotions)
            except Exception as e:
                print(f"Error loading CSV: {e}. Fallback to synthetic.")
                self._generate_synthetic(size, emotions)
        else:
            if csv_file:
                 print(f"File {csv_file} not found. Fallback to synthetic.")
            else:
                 print("No CSV provided. Using synthetic data.")
            self._generate_synthetic(size, emotions)

    def _generate_synthetic(self, size, emotions):
        print(f"Generating {size} synthetic samples...")
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
            
            self._process_item(text, label_idx)

    def _process_item(self, text, label_idx):
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
    
    # Dataset
    # Check for local data/train.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'train.csv')
    
    train_dataset = EmailDataset(csv_file=csv_path, size=50, tokenizer=tokenizer)
    
    if len(train_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

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
