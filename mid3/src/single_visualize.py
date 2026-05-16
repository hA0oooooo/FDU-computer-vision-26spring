import argparse
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_datasets
from .model_unet import build_model
from .utils import ensure_dir, get_device, load_config


NUM_IMAGES = 3
SPLIT = "val"
SEED = 42

MODEL_CONFIGS = {
    "unet_ce": "configs/unet_ce.yaml",
    "unet_dice": "configs/unet_dice.yaml",
    "unet_ce_dice": "configs/unet_ce_dice.yaml",
}

MASK_COLORS = np.array(
    [
        [255, 0, 0],
        [0, 0, 0],
        [255, 220, 0],
    ],
    dtype=np.uint8,
)


def denormalize(image_tensor):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD).reshape(1, 1, 3)
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def colorize_mask(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    return MASK_COLORS[mask_np]


def get_dataset(cfg):
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    if SPLIT == "train":
        return train_dataset
    if SPLIT == "val":
        return val_dataset
    if SPLIT == "test":
        return test_dataset
    raise ValueError(f"Unknown split: {SPLIT}")


def sample_batch(dataset):
    rng = random.Random(SEED)
    indices = rng.sample(range(len(dataset)), k=min(NUM_IMAGES, len(dataset)))
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples], dim=0)
    masks = torch.stack([sample[1] for sample in samples], dim=0)
    return images, masks


@torch.no_grad()
def predict(cfg, images):
    device = get_device(cfg.get("device", "auto"))
    model = build_model(cfg).to(device)
    checkpoint_path = os.path.join(cfg["output_dir"], "checkpoints", "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model(images.to(device)).argmax(dim=1).cpu()


def make_single_figure(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: {', '.join(MODEL_CONFIGS)}")

    cfg = load_config(MODEL_CONFIGS[model_name])
    images, masks = sample_batch(get_dataset(cfg))
    preds = predict(cfg, images)

    fig = plt.figure(figsize=(9, NUM_IMAGES * 3.2))
    grid = GridSpec(
        NUM_IMAGES * 2,
        3,
        figure=fig,
        height_ratios=sum(([1.0, 0.10] for _ in range(NUM_IMAGES)), []),
    )
    for row in range(NUM_IMAGES):
        axes = [fig.add_subplot(grid[row * 2, col]) for col in range(3)]
        axes[0].imshow(denormalize(images[row]))
        axes[1].imshow(colorize_mask(masks[row]))
        axes[2].imshow(colorize_mask(preds[row]))
        for col in range(3):
            axes[col].axis("off")

        label_ax = fig.add_subplot(grid[row * 2 + 1, :])
        label_ax.text(0.5, 0.5, "origin | ground truth | prediction", ha="center", va="center", fontsize=11)
        label_ax.axis("off")

    output_path = os.path.join("outputs", model_name, f"{model_name}_example.png")
    ensure_dir(os.path.dirname(output_path))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.02, hspace=0.08)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=sorted(MODEL_CONFIGS))
    return parser.parse_args()


def main():
    args = parse_args()
    make_single_figure(args.model_name)


if __name__ == "__main__":
    main()
