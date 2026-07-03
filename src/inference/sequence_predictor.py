from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from src.data.preprocessing import load_image_grayscale
from src.models.crnn import CRNN
from src.utils.common import get_device
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SequencePredictor:
    """Inference wrapper for the CRNN model: recognizes full words/sentences end-to-end
    without requiring explicit character segmentation, using CTC greedy decoding."""

    def __init__(self, model_path: str = None, config=None):
        self.config = config or get_config()
        self.device = get_device()

        model_path = model_path or self.config.resolve_path(
            self.config.get("inference.crnn_model_path", "checkpoints/best_crnn_model.pt")
        )
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained CRNN model found at {model_path}. Train a CRNN checkpoint first."
            )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_info = checkpoint["label_info"]
        self.idx_to_char = self.label_info["idx_to_char"]

        self.model = CRNN(num_classes=self.label_info["num_classes"], config=self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.target_height = self.config.get("model.crnn.input_height", 32)
        logger.info(f"Loaded SequencePredictor from {model_path}")

    def _prepare_image(self, image_bytes: bytes) -> torch.Tensor:
        img = load_image_grayscale(image_bytes)
        if img.mean() > 127:
            img = 255 - img

        h, w = img.shape
        scale = self.target_height / h
        new_w = max(int(w * scale), self.target_height)
        resized = cv2.resize(img, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        tensor = self._prepare_image(image_bytes).to(self.device)
        decoded = self.model.greedy_decode(tensor, self.idx_to_char)
        text = decoded[0] if decoded else ""
        return {"prediction": text, "length": len(text)}
