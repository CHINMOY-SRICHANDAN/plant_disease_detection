import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import os

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow frontend requests


# ----------- Load Model -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same model as training
model = models.resnet50(pretrained=False)
num_classes = 15  # change if your dataset has different count
model.fc = nn.Linear(model.fc.in_features, num_classes)

# load trained weights
model_path = "plant_disease_model.pth"
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ----------- Same Preprocessing as train.py -------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # must match training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# ----------- Prediction function -------------
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = class_names[predicted.item()]

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
