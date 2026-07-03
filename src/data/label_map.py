import string
from typing import Dict, List


def get_mnist_labels() -> List[str]:
    return [str(i) for i in range(10)]


def get_emnist_labels(split: str) -> List[str]:
    if split == "digits":
        return [str(i) for i in range(10)]
    if split == "letters":
        return list(string.ascii_uppercase)
    if split == "balanced":
        digits = [str(i) for i in range(10)]
        upper = list(string.ascii_uppercase)
        lower_subset = list("abdefghnqrt")
        return digits + upper + lower_subset
    if split in ("byclass", "bymerge"):
        digits = [str(i) for i in range(10)]
        upper = list(string.ascii_uppercase)
        lower = list(string.ascii_lowercase)
        return digits + upper + lower if split == "byclass" else digits + upper + lower_subset_bymerge()
    raise ValueError(f"Unknown EMNIST split: {split}")


def lower_subset_bymerge() -> List[str]:
    return list("abdefghnqrt")


def build_label_maps(dataset: str, emnist_split: str = "balanced") -> Dict:
    labels = get_mnist_labels() if dataset == "mnist" else get_emnist_labels(emnist_split)
    idx_to_char = {i: ch for i, ch in enumerate(labels)}
    char_to_idx = {ch: i for i, ch in enumerate(labels)}
    return {
        "idx_to_char": idx_to_char,
        "char_to_idx": char_to_idx,
        "num_classes": len(labels),
        "labels": labels,
    }
