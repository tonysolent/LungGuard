import torch
import torch.nn as nn
from torchvision import models

class ImageFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, out_dim=512):
        super(ImageFeatureExtractor, self).__init__()
        # Use ResNet-50 backbone
        backbone = models.resnet50(pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # Remove final FC
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = backbone.fc.in_features  # 2048
        self.fc = nn.Linear(self.feature_dim, out_dim)  # Reduce dimension for fusion

    def forward(self, x):
        x = self.cnn(x)         # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)          # [B, out_dim]
        return x
