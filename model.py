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
    "1": "Angry",
    "2": "Happy",
    "3": "Neutral",
    "4": "Sad",
    "5": "Surprised",
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

    def get_embedding(self, x):
        """
        The 512-dim shared feature vector, before either classification head.
        Used as a face "embedding" for similarity-based matching (see
        gallery.py) when enrolling someone without retraining the classifier
        — this network was trained for classification, not metric learning,
        so these embeddings won't separate identities as cleanly as a
        purpose-built embedding model, but they're a reasonable fallback for
        people the identity_head has never seen.
        """
        x = self.backbone(x)
        x = self.pool(x)
        return self.shared_fc(x)

    def forward(self, x):
        feat = self.get_embedding(x)
        return self.identity_head(feat), self.expression_head(feat)

    def forward_with_embedding(self, x):
        """Same computation as forward(), but also returns the shared 512-dim feature.
        Classification and the gallery-matching embedding (see get_embedding) come from
        the exact same backbone pass this way -- callers that need both (the live
        classifier-then-gallery-fallback path in inference.py) don't have to run the
        backbone a second time just to get the embedding forward() already computed
        and discarded."""
        feat = self.get_embedding(x)
        return self.identity_head(feat), self.expression_head(feat), feat
