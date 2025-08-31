import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from flask_cors import CORS
import json
import os

# ----------------- Flask Setup -----------------
app = Flask(__name__, static_folder="static")
CORS(app)  # allow frontend requests

# ----------------- Load Model -----------------
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

# ----------------- Preprocessing -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # must match training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# ----------------- Prediction Function -----------------
def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


# ----------------- API Routes -----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file).convert("RGB")
        result = predict_image(img)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    # serve index.html from /static
    return send_from_directory(app.static_folder, "index.html")


# ----------------- Run Locally -----------------
if __name__ == "__main__":
    app.run(debug=True)
