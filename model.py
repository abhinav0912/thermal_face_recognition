"""
model.py
========
Shared model definition and constants used by train.py, inference.py, and
live_inference.py, so the architecture and label set only live in one place.
"""

import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 128
NUM_EXPRESSIONS = 5

EXPR_MAP = {
    "1": "Neutral",
    "2": "Smile",
    "3": "Eyes Closed",
    "4": "Surprised",
    "5": "Sunglasses",
}
EXPR_NAMES = [EXPR_MAP[str(i + 1)] for i in range(NUM_EXPRESSIONS)]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class DualHeadFaceNet(nn.Module):
    """
    MobileNetV2 backbone with two classification heads:
      • identity_head   -> num_persons classes
      • expression_head -> num_expressions classes
    """

    def __init__(self, num_persons: int, num_expressions: int = NUM_EXPRESSIONS,
                 dropout: float = 0.4, pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.backbone = base.features          # output: (B, 1280, 4, 4) for 128x128
        self.pool = nn.AdaptiveAvgPool2d(1)    # -> (B, 1280, 1, 1)

        feat_dim = 1280
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.identity_head = nn.Linear(512, num_persons)
        self.expression_head = nn.Linear(512, num_expressions)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = self.shared_fc(x)
        return self.identity_head(feat), self.expression_head(feat)
