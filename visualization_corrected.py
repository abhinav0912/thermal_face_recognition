import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import sys
import os

class DualHeadFaceNet(nn.Module):
    def __init__(self, num_persons, num_expressions, dropout=0.4):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
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

# Load model
device = torch.device('cpu')
print("Loading model...")
ckpt = torch.load('checkpoints/best_model.pth', map_location=device)

# Get number of persons from checkpoint
num_persons = ckpt['model_state']['identity_head.weight'].shape[0]
print(f"Model has {num_persons} persons")

# Initialize model
model = DualHeadFaceNet(num_persons, 5).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# CORRECT EXPRESSION NAMES - MATCHING THE DATASET
expr_names = ['Neutral', 'Smile', 'Shocked', 'Sunglasses', 'Eyes Closed']

# Get image path
image_path = sys.argv[1] if len(sys.argv) > 1 else 'data/thermal-face-128x128/110-TD-E-2.jpg'

print(f"Processing: {image_path}")

# Extract person ID from filename (e.g., "7-TD-E-2.jpg" -> person 7)
filename = os.path.basename(image_path)
person_from_file = int(filename.split('-')[0])

# Load and preprocess thermal image
thermal_img = Image.open(image_path).convert('RGB')
thermal_array = np.array(thermal_img)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img_tensor = transform(thermal_img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    id_logits, expr_logits = model(img_tensor)
    id_pred = id_logits.argmax(1).item()
    id_conf = torch.softmax(id_logits, 1)[0, id_pred].item()
    expr_pred = expr_logits.argmax(1).item()
    expr_conf = torch.softmax(expr_logits, 1)[0, expr_pred].item()

# The predicted person ID is id_pred + 1 (since indices are 0-111 but persons are 1-112)
predicted_person_id = id_pred + 1

print(f"Actual Person: {person_from_file}")
print(f"Predicted Person: {predicted_person_id} ({id_conf*100:.1f}%)")
print(f"Expression: {expr_names[expr_pred]} ({expr_conf*100:.1f}%)")

# Create visualization
fig = np.ones((400, 1200, 3), dtype=np.uint8) * 30

# Thermal image (left)
thermal_resized = cv2.resize(thermal_array, (300, 400))
fig[:, :300] = thermal_resized

# Heatmap (middle)
thermal_gray = cv2.cvtColor(thermal_array, cv2.COLOR_RGB2GRAY)
thermal_heatmap = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_JET)
thermal_heatmap = cv2.resize(thermal_heatmap, (300, 400))
fig[:, 300:600] = thermal_heatmap

# RGB image (right)
try:
    rgb_path = image_path.replace('thermal-face-128x128', 'RGB-faces-128x128')
    rgb_img = Image.open(rgb_path).convert('RGB')
    rgb_array = np.array(rgb_img)
    rgb_resized = cv2.resize(rgb_array, (300, 400))
    fig[:, 600:900] = rgb_resized
except Exception as e:
    print(f"Note: RGB image not found")

# Title bar with prediction - CORRECTED EXPRESSION NAMES
title_text = f'Person ID: {predicted_person_id} ({id_conf*100:.1f}%) | Expression: {expr_names[expr_pred]} ({expr_conf*100:.1f}%)'
cv2.putText(fig, title_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

# Show visualization
cv2.imshow(f'Thermal Face Recognition - Person {predicted_person_id} | {expr_names[expr_pred]}', fig)
print("\nVisualization displayed. Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
