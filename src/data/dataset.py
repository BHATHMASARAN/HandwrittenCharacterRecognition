from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.data.label_map import build_label_maps
from src.utils.config_loader import get_config


class EMNISTOrientationFix:
    """EMNIST images are stored transposed and vertically flipped relative to MNIST."""

    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(0)

    def transpose(self, img):
        return img


def _emnist_fix(img):
    import torchvision.transforms.functional as F
    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


def build_transforms(train: bool, dataset: str):
    config = get_config()
    image_size = config.get("data.image_size", 28)
    mean = config.get("data.normalize_mean", 0.1736)
    std = config.get("data.normalize_std", 0.3317)
    augment = config.get("data.augment", True) and train
    aug_cfg = config.get("data.augmentation", {})

    ops = []
    if dataset == "emnist":
        ops.append(transforms.Lambda(_emnist_fix))
    if image_size != 28:
        ops.append(transforms.Resize((image_size, image_size)))
    if augment:
        ops.append(
            transforms.RandomAffine(
                degrees=aug_cfg.get("rotation_degrees", 10),
                translate=tuple(aug_cfg.get("translate", [0.1, 0.1])),
                scale=tuple(aug_cfg.get("scale", [0.9, 1.1])),
            )
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize((mean,), (std,)))
    return transforms.Compose(ops)


def get_datasets() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, dict]:
    config = get_config()
    root_dir = config.resolve_path(config.get("data.root_dir", "data/raw"))
    dataset_name = config.get("data.dataset", "emnist")
    val_split = config.get("data.val_split", 0.1)

    train_tf = build_transforms(train=True, dataset=dataset_name)
    eval_tf = build_transforms(train=False, dataset=dataset_name)

    if dataset_name == "mnist":
        full_train = datasets.MNIST(root=root_dir, train=True, download=True, transform=train_tf)
        eval_train = datasets.MNIST(root=root_dir, train=True, download=True, transform=eval_tf)
        test_set = datasets.MNIST(root=root_dir, train=False, download=True, transform=eval_tf)
        label_info = build_label_maps("mnist")
    elif dataset_name == "emnist":
        split = config.get("data.emnist_split", "balanced")
        full_train = datasets.EMNIST(root=root_dir, split=split, train=True, download=True, transform=train_tf)
        eval_train = datasets.EMNIST(root=root_dir, split=split, train=True, download=True, transform=eval_tf)
        test_set = datasets.EMNIST(root=root_dir, split=split, train=False, download=True, transform=eval_tf)
        label_info = build_label_maps("emnist", split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.get("project.seed", 42))
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_set = torch.utils.data.Subset(full_train, train_indices.indices)
    val_set = torch.utils.data.Subset(eval_train, val_indices.indices)

    return train_set, val_set, test_set, label_info


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    config = get_config()
    batch_size = config.get("data.batch_size", 128)
    num_workers = config.get("data.num_workers", 4)

    train_set, val_set, test_set, label_info = get_datasets()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_info
