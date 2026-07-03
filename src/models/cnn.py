import torch
import torch.nn as nn

from src.utils.config_loader import get_config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CharCNN(nn.Module):
    """Convolutional Neural Network for single handwritten character classification."""

    def __init__(self, num_classes: int, config=None):
        super().__init__()
        config = config or get_config()
        channels = config.get("model.cnn.conv_channels", [32, 64, 128])
        kernel_size = config.get("model.cnn.kernel_size", 3)
        pool_size = config.get("model.cnn.pool_size", 2)
        dropout = config.get("model.cnn.dropout", 0.4)
        fc_hidden = config.get("model.cnn.fc_hidden", 256)
        in_channels = config.get("data.channels", 1)
        image_size = config.get("data.image_size", 28)

        blocks = []
        prev_c = in_channels
        spatial = image_size
        for c in channels:
            blocks.append(ConvBlock(prev_c, c, kernel_size, pool_size))
            prev_c = c
            spatial = spatial // pool_size
        self.features = nn.Sequential(*blocks)

        flat_dim = prev_c * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
