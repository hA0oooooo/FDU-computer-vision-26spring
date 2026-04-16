from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np


def load_idx_images(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as file:
        _, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        buffer = file.read()

    images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows * cols)
    return images.astype(np.float32) / 255.0


def load_idx_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as file:
        _, num_labels = struct.unpack(">II", file.read(8))
        buffer = file.read()

    labels = np.frombuffer(buffer, dtype=np.uint8, count=num_labels)
    return labels.astype(np.int64)


def load_fashion_mnist(data_dir: str | Path = "data/raw") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    train_images = load_idx_images(data_dir / "train-images-idx3-ubyte")
    train_labels = load_idx_labels(data_dir / "train-labels-idx1-ubyte")
    test_images = load_idx_images(data_dir / "t10k-images-idx3-ubyte")
    test_labels = load_idx_labels(data_dir / "t10k-labels-idx1-ubyte")
    return train_images, train_labels, test_images, test_labels


def split_train_val(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_path: str | Path = "data/splits/split_indices.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        split = np.load(split_path)
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(train_images.shape[0])
        val_size = int(train_images.shape[0] * val_ratio)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        np.savez(split_path, train_idx=train_idx, val_idx=val_idx)

    return (
        train_images[train_idx],
        train_labels[train_idx],
        train_images[val_idx],
        train_labels[val_idx],
    )


def make_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(images.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, images.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield images[batch_idx], labels[batch_idx]