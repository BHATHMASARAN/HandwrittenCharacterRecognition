import torch

from src.models.cnn import CharCNN
from src.models.crnn import CRNN
from src.utils.config_loader import get_config


def test_char_cnn_output_shape():
    config = get_config()
    num_classes = 47
    model = CharCNN(num_classes=num_classes, config=config)
    batch = torch.randn(8, 1, 28, 28)
    output = model(batch)
    assert output.shape == (8, num_classes)


def test_char_cnn_predict_proba_sums_to_one():
    config = get_config()
    model = CharCNN(num_classes=10, config=config)
    model.eval()
    batch = torch.randn(4, 1, 28, 28)
    probs = model.predict_proba(batch)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-4)


def test_crnn_output_shape():
    config = get_config()
    num_classes = 26
    model = CRNN(num_classes=num_classes, config=config)
    batch = torch.randn(2, 1, 32, 160)
    log_probs = model(batch)
    assert log_probs.shape[1] == 2
    assert log_probs.shape[2] == num_classes + 1


def test_crnn_greedy_decode_runs():
    config = get_config()
    num_classes = 26
    idx_to_char = {i: chr(65 + i) for i in range(num_classes)}
    model = CRNN(num_classes=num_classes, config=config)
    batch = torch.randn(1, 1, 32, 160)
    decoded = model.greedy_decode(batch, idx_to_char)
    assert isinstance(decoded, list)
    assert isinstance(decoded[0], str)
