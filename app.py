import os
import subprocess

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from kidney_classifier.pipeline.prediction import PredictionPipeline
from kidney_classifier.utils.common import decode_image


os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = os.path.join("prediction_test_file", "inputImage.jpg")
        self.classifier = PredictionPipeline(self.filename)


os.makedirs("prediction_test_file", exist_ok=True)
cl_app = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train_route():
    subprocess.run(["python", "main.py"], check=True)
    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_route():
    if not os.path.exists(cl_app.classifier.model_path):
        return jsonify({"error": "Model not found. Train the model before prediction."}), 404

    image = request.json.get("image")
    if not image:
        return jsonify({"error": "Image data is required."}), 400

    if "," in image:
        image = image.split(",", 1)[1]

    decode_image(image, cl_app.filename)
    result = cl_app.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
