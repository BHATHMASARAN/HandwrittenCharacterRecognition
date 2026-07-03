import io

import numpy as np
import pytest
from PIL import Image

from src.data.label_map import build_label_maps
from src.data.preprocessing import preprocess_single_char, segment_characters


def _make_char_image_bytes(text_present: bool = True) -> bytes:
    img = np.full((100, 100), 255, dtype=np.uint8)
    if text_present:
        img[30:70, 30:70] = 0
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocess_single_char_shape():
    image_bytes = _make_char_image_bytes()
    tensor = preprocess_single_char(image_bytes)
    assert tensor.shape == (1, 1, 28, 28)


def test_label_map_mnist():
    info = build_label_maps("mnist")
    assert info["num_classes"] == 10
    assert info["idx_to_char"][0] == "0"


def test_label_map_emnist_balanced():
    info = build_label_maps("emnist", "balanced")
    assert info["num_classes"] == 47
    assert "A" in info["char_to_idx"]


def test_segment_characters_returns_list():
    image_bytes = _make_char_image_bytes()
    crops = segment_characters(image_bytes)
    assert isinstance(crops, list)
