import torch
import torch.nn as nn
from transformers import AutoModel

class TextFeatureExtractor(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", out_dim=128):
        super(TextFeatureExtractor, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.out_dim = out_dim
        hidden_size = self.transformer.config.hidden_size  # e.g., 768
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        out = self.fc(cls_embedding)
        return out
