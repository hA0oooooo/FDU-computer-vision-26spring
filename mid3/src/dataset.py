import json
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_split_names(split_file: str) -> List[str]:
    names = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.append(line.split()[0])
    return names


def make_or_load_split(names: List[str], val_ratio: float, seed: int, split_path: str) -> Tuple[List[str], List[str]]:
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

        if "train_names" in split and "val_names" in split:
            return split["train_names"], split["val_names"]

        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        return [names[i] for i in train_indices], [names[i] for i in val_indices]

    rng = np.random.default_rng(seed)
    indices = np.arange(len(names))
    rng.shuffle(indices)

    val_size = int(len(names) * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    split = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_names": [names[i] for i in train_indices],
        "val_names": [names[i] for i in val_indices],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    return split["train_names"], split["val_names"]


class PetSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        names: List[str],
        image_size: int = 256,
    ):
        self.root = root
        self.names = names
        self.image_size = image_size
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image_path = os.path.join(self.root, "images", f"{name}.jpg")
        mask_path = os.path.join(self.root, "annotations", "trimaps", f"{name}.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        size = (self.image_size, self.image_size)
        image = image.resize(size, resample=Image.BILINEAR)
        mask = mask.resize(size, resample=Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - self.mean) / self.std
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        mask_np = np.asarray(mask, dtype=np.int64) - 1
        if mask_np.min() < 0 or mask_np.max() > 2:
            raise ValueError(f"Unexpected trimap values after remap in {mask_path}: {np.unique(mask_np)}")
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


def build_datasets(cfg):
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    root = data_cfg["root"]
    image_size = data_cfg.get("image_size", 256)
    val_ratio = data_cfg.get("val_ratio", 0.2)
    seed = cfg.get("seed", data_cfg.get("seed", 42))
    trainval_names = read_split_names(os.path.join(root, "annotations", "trainval.txt"))
    test_names = read_split_names(os.path.join(root, "annotations", "test.txt"))

    split_path = os.path.join(root, "splits", f"trainval_seed{seed}_val{val_ratio}.json")
    train_names, val_names = make_or_load_split(trainval_names, val_ratio, seed, split_path)

    train_dataset = PetSegmentationDataset(
        root=root,
        names=train_names,
        image_size=image_size,
    )
    val_dataset = PetSegmentationDataset(root=root, names=val_names, image_size=image_size)
    test_dataset = PetSegmentationDataset(root=root, names=test_names, image_size=image_size)
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(cfg):
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    batch_size = cfg["train"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)
    seed = cfg.get("seed", data_cfg.get("seed", 42))
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
