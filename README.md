# Handwritten Character Recognition

CNN-based handwritten digit and character recognition trained on MNIST/EMNIST, with an
extendable CRNN + CTC path for full word/sentence recognition. Includes a Flask inference
API with a cyberpunk-themed drawing-canvas UI.

## Architecture

```
HandwrittenCharRecognition/
├── config/
│   └── config.yaml              # Centralized configuration
├── data/
│   ├── raw/                     # Downloaded MNIST/EMNIST
│   └── processed/
├── src/
│   ├── data/
│   │   ├── dataset.py           # MNIST/EMNIST loading + augmentation
│   │   ├── preprocessing.py     # Image cleanup, deskew, segmentation
│   │   └── label_map.py         # Class index <-> character mapping
│   ├── models/
│   │   ├── cnn.py               # CharCNN — single character classifier
│   │   └── crnn.py              # CRNN — CTC-based sequence model
│   ├── training/
│   │   ├── trainer.py           # Training loop, MLflow logging, checkpoints
│   │   ├── metrics.py           # Accuracy/precision/recall/F1, early stopping
│   │   └── evaluate.py          # Test-set evaluation, classification report
│   ├── inference/
│   │   ├── predictor.py         # Single character + segmented word inference
│   │   └── sequence_predictor.py # End-to-end CRNN word/sentence inference
│   ├── utils/                   # Config loader, logger, seed/device helpers
│   ├── train.py                 # CNN training entry point
│   ├── train_crnn.py            # CRNN training entry point
│   └── predict.py               # CLI inference
├── app/                         # Flask API + cyberpunk drawing-canvas UI
├── tests/                       # Unit tests (models, preprocessing, metrics)
├── notebooks/                   # Data exploration
├── checkpoints/                 # Saved model weights
├── requirements.txt
└── Dockerfile
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Train the single-character CNN on EMNIST (balanced split by default, configurable in
`config/config.yaml`):

```bash
python -m src.train --dataset emnist --epochs 30
```

MNIST digits only:

```bash
python -m src.train --dataset mnist --epochs 20
```

Training runs log to `logs/`, checkpoint the best model to `checkpoints/best_model.pt`, and
(if MLflow is installed) track metrics under `mlruns/`.

### Extending to word/sentence recognition (CRNN)

The CNN classifies one pre-segmented character at a time. For full word or sentence
recognition without manual segmentation, train the CRNN on a directory of word images
(e.g. IAM Handwriting Database, or synthetically composed EMNIST words):

```bash
python -m src.train_crnn --data-dir data/words --epochs 40
```

The CRNN combines a convolutional feature extractor with a bidirectional LSTM and CTC loss,
so it learns alignment between image regions and characters without needing per-character
bounding boxes.

## Inference

CLI, single character:

```bash
python -m src.predict path/to/character.png
```

CLI, segmented word (contour-based character splitting + CNN per character):

```bash
python -m src.predict path/to/word.png --word
```

## Running the web app

```bash
python app/app.py
```

Then open `http://localhost:5000`. Draw a character on the canvas and click **ANALYZE** to
get the predicted class, confidence score, and top-k alternatives.

API endpoints:
- `GET /api/health` — model readiness check
- `POST /api/predict` — single character (multipart form field `image`)
- `POST /api/predict-word` — segmented multi-character word

## Testing

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t handwritten-char-recognition .
docker run -p 5000:5000 -v $(pwd)/checkpoints:/workspace/checkpoints handwritten-char-recognition
```

## Model summary

| Component | Details |
|---|---|
| Single-character model | CNN, 3 conv blocks (32/64/128 channels), BatchNorm, dropout regularization |
| Sequence model | CRNN — CNN backbone + 2-layer BiLSTM + CTC decoding |
| Datasets | MNIST (digits), EMNIST (balanced: digits + upper + common lowercase) |
| Preprocessing | Grayscale, Otsu thresholding, deskew/center, EMNIST orientation correction |
| Augmentation | Random affine (rotation, translation, scale) during training |
| Training | AdamW, cosine LR schedule, label smoothing, gradient clipping, early stopping |
| Tracking | MLflow experiment logging (optional) |
