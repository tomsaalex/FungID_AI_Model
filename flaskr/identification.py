import os

import pandas as pd
import torch

from PIL import Image
from flask import (
    Blueprint, request, jsonify
)
from torchvision.transforms import transforms
from werkzeug.utils import secure_filename

import timm as timm

bp = Blueprint('identification', __name__, url_prefix='/classifications')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image):
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image_tensor = transformation(image).unsqueeze(0)

    return image_tensor


def get_class_labels(csv_file_path):
    image_labels = pd.read_csv(csv_file_path)

    text_labels = image_labels.iloc[:, 0]
    unique_labels = text_labels.unique()
    label_map = {idx:label for idx, label in enumerate(unique_labels)}

    for idx, label in enumerate(unique_labels):
        print(idx, label)
    return label_map

image_classes_map = get_class_labels("model/data/mo106_dataset.csv")
model = timm.create_model("mobilevitv2_200", pretrained=False, num_classes=106)
model.load_state_dict(torch.load("model/model_files/mobile_vit_v2_pretrained_82.1_acc.pth"))
model.eval()


def get_image_classification(file):
    image_tensor = process_image(Image.open(file))
    output = model(image_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    class_index = probabilities.argmax()

    # Get the predicted class and probability
    print(image_classes_map)
    predicted_class = image_classes_map[class_index]

    return predicted_class


@bp.route("identify", methods=['POST'])
def identify_mushroom():
    if 'file' not in request.files:
        return jsonify({"classificationResult": "No file found"}), 400

    file = request.files['file']
    if file.filename == '':
        return {"classificationResult": "No selected file"}, 400

    if not allowed_file(file.filename):
        return {"classificationResult": "Invalid file type"}, 400

    if file and allowed_file(file.filename):
        image_classification = get_image_classification(file)
        return {"classificationResult": image_classification}, 200

    return {"classificationResult": "Something went wrong while processing the image"}, 500
