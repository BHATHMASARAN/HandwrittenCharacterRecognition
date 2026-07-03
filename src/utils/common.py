import random
from typing import Optional

import numpy as np
import torch

from src.utils.config_loader import get_config


def set_seed(seed: Optional[int] = None) -> None:
    config = get_config()
    seed = seed if seed is not None else config.get("project.seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    config = get_config()
    device_pref = config.get("project.device", "auto")
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
