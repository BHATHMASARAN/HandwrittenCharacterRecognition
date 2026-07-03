from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from src.training.metrics import MetricTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_test_set(model: nn.Module, test_loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tracker = MetricTracker()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        tracker.update(preds, labels, loss.item())

    return tracker.compute()


@torch.no_grad()
def full_classification_report(model: nn.Module, test_loader, device: torch.device, label_info: dict) -> str:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    target_names = [label_info["idx_to_char"][i] for i in range(label_info["num_classes"])]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    return report


@torch.no_grad()
def get_confusion_matrix(model: nn.Module, test_loader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return confusion_matrix(all_labels, all_preds)
