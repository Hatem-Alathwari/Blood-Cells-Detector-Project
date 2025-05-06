# utils.py

import io
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn , FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import base64

# Class mapping
CLASS_NAMES = {1: 'RBC', 2: 'WBC', 3: 'Platelet'}
CLASS_COLORS = {'RBC': 'red', 'WBC': 'blue', 'Platelet': 'green'}
# Load model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, device, num_classes=4):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform
transform = T.Compose([T.ToTensor()])

# Predict function
# Prediction function
def predict(model, device, image, threshold=0.5):
    image_tensor = transform(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    pred = outputs[0]
    result = []

    for i in range(len(pred['boxes'])):
        score = pred['scores'][i].item()
        if score > threshold:
            label = CLASS_NAMES.get(pred['labels'][i].item(), 'Unknown')
            box = pred['boxes'][i].cpu().numpy().tolist()
            result.append({
                "label": label,
                "confidence": score,
                "bbox": box
            })

    return result


def draw_boxes(image, predictions, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(image))

    for p in predictions:
        label = p['label']
        score = p['confidence']
        box = p['bbox']

        if score > threshold:
            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(label, 'black')  # Default to black if unknown

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{label} {score:.2f}", color='white',
                    fontsize=10, bbox=dict(facecolor=color, alpha=0.7))

    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    return data_uri