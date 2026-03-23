import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class build_resnet_predictor(nn.Module):
    def __init__(self, num_attrs=5, pretrained=True, pool="mean"):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()    

        assert pool in ["mean", "max"]
        self.pool = pool
        self.head = nn.Linear(feat_dim, num_attrs)  

    def forward(self, x):
        # x: (B,S,3,H,W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feat = self.backbone(x)               
        feat = feat.view(B, S, -1)            

        if self.pool == "max":
            clip_feat = feat.max(dim=1).values
        else:
            clip_feat = feat.mean(dim=1)

        logits = self.head(clip_feat)         
        return logits
