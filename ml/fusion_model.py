import torch
import torch.nn as nn
from transformers import DistilBertModel

class FusionModel(nn.Module):
    def __init__(self, num_classes=4, behavioral_dim=6):
        super(FusionModel, self).__init__()
        
        # Text Branch: DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze BERT layers to speed up training for prototype (optional, but good for speed)
        # For better accuracy, we can unfreeze last few layers. Let's keep it trainable for now but be mindful.
        
        self.text_hidden_dim = 768
        self.text_projector = nn.Linear(self.text_hidden_dim, 256)
        self.text_dropout = nn.Dropout(0.3)

        # Behavioral Branch: MLP
        self.behavioral_dim = behavioral_dim
        self.behavioral_mlp = nn.Sequential(
            nn.Linear(behavioral_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.behavioral_out_dim = 64

        # Fusion
        self.fusion_dim = 256 + 64
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes) 
        )

    def forward(self, input_ids, attention_mask, behavioral_features):
        # Text Branch
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use cls token (index 0)
        cls_token = bert_output.last_hidden_state[:, 0, :] 
        text_emb = self.text_projector(cls_token)
        text_emb = self.text_dropout(text_emb)

        # Behavioral Branch
        behavioral_emb = self.behavioral_mlp(behavioral_features)

        # Fusion
        fused = torch.cat((text_emb, behavioral_emb), dim=1)
        logits = self.classifier(fused)
        
        return logits
