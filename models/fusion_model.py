import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionClassifier(nn.Module):
    def __init__(
        self,
        image_feature_dim=512,
        tabular_feature_dim=128,
        text_feature_dim=128,
        fusion_type='early',         # 'early', 'late', or 'hybrid'
        num_classes=1,
        dropout=0.3
    ):
        super(FusionClassifier, self).__init__()
        self.fusion_type = fusion_type

        # Heads for late/hybrid fusion (each predicts separately)
        self.image_head = nn.Linear(image_feature_dim, num_classes)
        self.tabular_head = nn.Linear(tabular_feature_dim, num_classes)
        self.text_head = nn.Linear(text_feature_dim, num_classes)

        # Early/Hybrid: concatenate features, then classify
        fusion_input_dim = image_feature_dim + tabular_feature_dim + text_feature_dim
        self.early_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),   # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),    # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Hybrid: combine early and late predictions
        self.hybrid_head = nn.Linear(num_classes * 4, num_classes)  # [early_pred, img, tab, txt]

    def forward(self, img_feat, tab_feat, txt_feat):
        # Individual predictions (for late/hybrid)
        img_pred = self.image_head(img_feat)
        tab_pred = self.tabular_head(tab_feat)
        txt_pred = self.text_head(txt_feat)

        # Early fusion: concatenate and classify
        concat_feat = torch.cat([img_feat, tab_feat, txt_feat], dim=1)
        early_pred = self.early_head(concat_feat)

        if self.fusion_type == 'early':
            return early_pred
        elif self.fusion_type == 'late':
            # Average (can also try weighted sum or learnable fusion)
            out = (img_pred + tab_pred + txt_pred) / 3.0
            return out
        elif self.fusion_type == 'hybrid':
            # Concatenate all predictions + early fusion
            hybrid_in = torch.cat([early_pred, img_pred, tab_pred, txt_pred], dim=1)
            return self.hybrid_head(hybrid_in)
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
