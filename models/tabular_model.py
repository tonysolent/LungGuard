import torch
import torch.nn as nn

class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(TabularFeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # Use LayerNorm here as well
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x.requires_grad_()  # Enable gradient tracking (for SHAP, etc.)
        # Optionally add debug print statements if needed:
        # print(f"🧬 Tabular raw input: {x}")
        out = self.model(x)
        # print(f"🧬 Tabular output: {out}")
        return out
