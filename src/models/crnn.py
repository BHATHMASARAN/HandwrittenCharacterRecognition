import torch
import torch.nn as nn

from src.utils.config_loader import get_config


class CNNFeatureExtractor(nn.Module):
    """CNN backbone that converts a word/line image into a sequence of feature vectors."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(out_channels, out_channels, 2, stride=1, padding=0), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)              # (B, C, H, W) with H collapsed to 1
        features = features.squeeze(2)       # (B, C, W)
        features = features.permute(0, 2, 1)  # (B, W, C) -- sequence of width steps
        return features


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        output = self.fc(recurrent)
        return output


class CRNN(nn.Module):
    """CRNN for word/sentence-level handwriting recognition, trained with CTC loss.

    Extends single-character CNN classification to sequence modeling: a CNN backbone
    extracts a feature sequence along the image width, a bidirectional LSTM models
    temporal/character dependencies, and CTC decoding produces the final transcription
    without requiring pre-segmented characters.

    NOTE: the CNN backbone performs 4 height-reductions (two /2 pools + two (2,1) pools)
    followed by a kernel-2 valid convolution, so input images must be exactly height 32
    (the standard CRNN convention). Width is variable and becomes the output time axis.
    """

    def __init__(self, num_classes: int, config=None):
        super().__init__()
        config = config or get_config()
        in_channels = config.get("data.channels", 1)
        cnn_out = config.get("model.crnn.cnn_out_channels", 256)
        rnn_hidden = config.get("model.crnn.rnn_hidden", 256)
        rnn_layers = config.get("model.crnn.rnn_layers", 2)
        dropout = config.get("model.crnn.dropout", 0.3)

        # num_classes + 1 reserves index 0 for the CTC blank token
        self.blank_idx = 0
        total_classes = num_classes + 1

        self.feature_extractor = CNNFeatureExtractor(in_channels, cnn_out)
        self.rnn = BidirectionalLSTM(cnn_out, rnn_hidden, total_classes, rnn_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.rnn(features)              # (B, T, num_classes+1)
        log_probs = logits.log_softmax(dim=2)
        return log_probs.permute(1, 0, 2)         # (T, B, num_classes+1) for CTCLoss

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor, idx_to_char: dict) -> list:
        self.eval()
        log_probs = self.forward(x)               # (T, B, C)
        preds = log_probs.argmax(dim=2).permute(1, 0)  # (B, T)

        results = []
        for seq in preds:
            chars = []
            prev = self.blank_idx
            for idx in seq.tolist():
                if idx != prev and idx != self.blank_idx:
                    chars.append(idx_to_char.get(idx - 1, "?"))
                prev = idx
            results.append("".join(chars))
        return results
