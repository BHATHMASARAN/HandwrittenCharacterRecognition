from pathlib import Path
from typing import Dict, List

import torch

from src.data.preprocessing import crop_to_tensor, preprocess_single_char, segment_characters
from src.models.cnn import CharCNN
from src.utils.common import get_device
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CharacterPredictor:
    def __init__(self, model_path: str = None, config=None):
        self.config = config or get_config()
        self.device = get_device()

        model_path = model_path or self.config.resolve_path(
            self.config.get("inference.model_path", "checkpoints/best_model.pt")
        )
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No trained model found at {model_path}. Run src/train.py first.")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_info = checkpoint["label_info"]
        self.idx_to_char = self.label_info["idx_to_char"]

        self.model = CharCNN(num_classes=self.label_info["num_classes"], config=self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.top_k = self.config.get("inference.top_k", 3)
        self.confidence_threshold = self.config.get("inference.confidence_threshold", 0.6)

        logger.info(f"Loaded CharacterPredictor from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Dict:
        tensor = preprocess_single_char(image_bytes).to(self.device)
        probs = self.model.predict_proba(tensor).squeeze(0)

        top_probs, top_indices = probs.topk(min(self.top_k, len(probs)))
        predictions = [
            {"character": self.idx_to_char[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        return {
            "prediction": predictions[0]["character"],
            "confidence": predictions[0]["confidence"],
            "is_confident": predictions[0]["confidence"] >= self.confidence_threshold,
            "top_k": predictions,
        }

    @torch.no_grad()
    def predict_word(self, image_bytes: bytes) -> Dict:
        """Segments a word/line image into individual characters and classifies each one."""
        crops = segment_characters(image_bytes)
        if not crops:
            return {"prediction": "", "characters": [], "confidence": 0.0}

        tensors = torch.cat([crop_to_tensor(c) for c in crops], dim=0).to(self.device)
        probs = self.model.predict_proba(tensors)
        top_probs, top_indices = probs.max(dim=1)

        characters = [
            {"character": self.idx_to_char[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]
        word = "".join(c["character"] for c in characters)
        avg_confidence = sum(c["confidence"] for c in characters) / len(characters)

        return {"prediction": word, "characters": characters, "confidence": round(avg_confidence, 4)}

    @torch.no_grad()
    def predict_batch(self, image_bytes_list: List[bytes]) -> List[Dict]:
        return [self.predict(img_bytes) for img_bytes in image_bytes_list]
