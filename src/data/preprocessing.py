from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.utils.config_loader import get_config


def load_image_grayscale(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def deskew_and_center(img: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(255 - img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    size = max(w, h)
    padded = np.full((size, size), 255, dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = cropped

    margin = int(size * 0.2)
    final = np.full((size + 2 * margin, size + 2 * margin), 255, dtype=np.uint8)
    final[margin:margin + size, margin:margin + size] = padded
    return final


def preprocess_single_char(image_bytes: bytes) -> torch.Tensor:
    config = get_config()
    image_size = config.get("data.image_size", 28)
    mean = config.get("data.normalize_mean", 0.1736)
    std = config.get("data.normalize_std", 0.3317)

    img = load_image_grayscale(image_bytes)

    if img.mean() > 127:
        img = 255 - img

    img = 255 - deskew_and_center(255 - img)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(img).float() / 255.0
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def segment_characters(image_bytes: bytes, min_area: int = 30) -> List[np.ndarray]:
    img = load_image_grayscale(image_bytes)
    if img.mean() > 127:
        img_inv = 255 - img
    else:
        img_inv = img

    _, thresh = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]
    boxes.sort(key=lambda b: b[0])

    crops = []
    for x, y, w, h in boxes:
        pad = int(0.15 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        crops.append(img[y0:y1, x0:x1])

    return crops


def crop_to_tensor(crop: np.ndarray) -> torch.Tensor:
    config = get_config()
    image_size = config.get("data.image_size", 28)
    mean = config.get("data.normalize_mean", 0.1736)
    std = config.get("data.normalize_std", 0.3317)

    if crop.mean() > 127:
        crop = 255 - crop

    padded = 255 - deskew_and_center(255 - crop)
    resized = cv2.resize(padded, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor
