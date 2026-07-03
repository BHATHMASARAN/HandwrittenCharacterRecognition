import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import CharacterPredictor
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.get("api.max_upload_size_mb", 5) * 1024 * 1024

_predictor = None


def get_predictor() -> CharacterPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CharacterPredictor()
    return _predictor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    model_ready = False
    try:
        get_predictor()
        model_ready = True
    except FileNotFoundError:
        model_ready = False
    return jsonify({"status": "ok", "model_ready": model_ready})


@app.route("/api/predict", methods=["POST"])
def predict_character():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under key 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        predictor = get_predictor()
        image_bytes = file.read()
        result = predictor.predict(image_bytes)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict-word", methods=["POST"])
def predict_word():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided under key 'image'"}), 400

    file = request.files["image"]

    try:
        predictor = get_predictor()
        image_bytes = file.read()
        result = predictor.predict_word(image_bytes)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Word prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 5000)
    debug = config.get("api.debug", False)
    app.run(host=host, port=port, debug=debug)
