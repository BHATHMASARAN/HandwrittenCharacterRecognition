"""
Training entry point for the CRNN sequence model (word/sentence-level recognition).

This script assumes a synthetic or real word-image dataset with (image, text_label) pairs,
e.g. IAM Handwriting Database or a synthetically composed EMNIST word generator. It is kept
separate from src/train.py (single-character CNN) because CRNN training requires CTC loss,
variable-width images, and text-sequence collation instead of single-label classification.

Usage:
    python -m src.train_crnn --data-dir data/words --epochs 40
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.label_map import build_label_maps
from src.models.crnn import CRNN
from src.utils.common import get_device, set_seed
from src.utils.config_loader import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WordImageDataset(Dataset):
    """Expects a directory of images named as '<label>_<idx>.png', e.g. 'HELLO_003.png'.
    Replace with a proper annotation-file loader (e.g. IAM 'words.txt') for real datasets."""

    def __init__(self, data_dir: str, char_to_idx: dict, image_height: int = 32):
        self.data_dir = Path(data_dir)
        self.paths = sorted(self.data_dir.glob("*.png"))
        self.char_to_idx = char_to_idx
        self.image_height = image_height

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import cv2
        path = self.paths[idx]
        label_text = path.stem.split("_")[0]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        scale = self.image_height / h
        new_w = max(int(w * scale), self.image_height)
        img = cv2.resize(img, (new_w, self.image_height))

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)

        target = torch.tensor([self.char_to_idx[c] + 1 for c in label_text if c in self.char_to_idx])
        return tensor, target, label_text


def collate_fn(batch):
    images, targets, texts = zip(*batch)
    max_width = max(img.shape[2] for img in images)
    height = images[0].shape[1]

    padded_images = torch.zeros(len(images), 1, height, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img

    target_lengths = torch.tensor([len(t) for t in targets])
    flat_targets = torch.cat(targets)

    return padded_images, flat_targets, target_lengths, texts


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN for word/sentence recognition")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory of word images")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    set_seed()
    device = get_device()

    label_info = build_label_maps(config.get("data.dataset", "emnist"), config.get("data.emnist_split", "balanced"))

    image_height = config.get("model.crnn.input_height", 32)
    dataset = WordImageDataset(args.data_dir, label_info["char_to_idx"], image_height=image_height)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = CRNN(num_classes=label_info["num_classes"], config=config).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(config.resolve_path(config.get("training.checkpoint_dir", "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for images, flat_targets, target_lengths, texts in loader:
            images = images.to(device)
            flat_targets = flat_targets.to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)

            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        logger.info(f"Epoch {epoch}/{args.epochs} | CTC Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "label_info": label_info,
            "config": config.all,
        }, checkpoint_dir / "best_crnn_model.pt")

    logger.info("CRNN training complete.")


if __name__ == "__main__":
    main()
