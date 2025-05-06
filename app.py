import os

from utils import draw_boxes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import base64
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import matplotlib.patches as patches
import json
import matplotlib
matplotlib.use('Agg')  # No GUI backend
import matplotlib.pyplot as plt
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Settings
MODEL_PATH = "Blood_Cells_Detector_Model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {1: 'RBC', 2: 'WBC', 3: 'Platelet'}
# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = T.Compose([T.ToTensor()])

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



# Routes
from io import BytesIO
from flask import send_file

@app.route('/preview/<string:img>')
def preview(img):
    try:
        decoded = base64.b64decode(img)
        buf = BytesIO(decoded)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return str(e)
    
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])

def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Load image
        image = Image.open(file.stream).convert("RGB")

        # Run prediction
        predictions = predict(model, DEVICE, image)

        # Draw boxes
        image_with_boxes = draw_boxes(image, predictions)

        return jsonify({
            "predictions": predictions,
            "image_with_boxes": image_with_boxes,
            "preview_url": f"http://localhost:5000/preview/{image_with_boxes}"
        
        })
    

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    print("🚀 Starting Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=True)