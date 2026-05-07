import json
import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_or_load_split(dataset_len: int, val_ratio: float, seed: int, split_path: str):
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        return split["train_indices"], split["val_indices"]

    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_len)
    rng.shuffle(indices)

    val_size = int(dataset_len * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return train_indices, val_indices


def build_dataloaders(cfg):
    data_cfg = cfg["data"]

    root = data_cfg["root"]
    image_size = data_cfg.get("image_size", 224)
    val_ratio = data_cfg.get("val_ratio", 0.2)
    num_workers = data_cfg.get("num_workers", 4)
    batch_size = cfg["train"]["batch_size"]
    seed = cfg.get("seed", 42)

    transform = build_transform(image_size)

    trainval_dataset = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=True,
        transform=transform,
    )

    test_dataset = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        download=True,
        transform=transform,
    )

    split_path = os.path.join(root, "oxford-iiit-pet", "splits", f"trainval_seed{seed}_val{val_ratio}.json")

    train_indices, val_indices = make_or_load_split(
        dataset_len=len(trainval_dataset),
        val_ratio=val_ratio,
        seed=seed,
        split_path=split_path,
    )

    train_dataset = Subset(trainval_dataset, train_indices)
    val_dataset = Subset(trainval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
