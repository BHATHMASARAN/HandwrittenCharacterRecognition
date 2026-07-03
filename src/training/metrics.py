from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        if not self.all_labels:
            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        avg_loss = self.total_loss / max(self.num_batches, 1)
        accuracy = accuracy_score(self.all_labels, self.all_preds)
        precision = precision_score(self.all_labels, self.all_preds, average="macro", zero_division=0)
        recall = recall_score(self.all_labels, self.all_preds, average="macro", zero_division=0)
        f1 = f1_score(self.all_labels, self.all_preds, average="macro", zero_division=0)

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


class EarlyStopping:
    def __init__(self, patience: int = 6, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
