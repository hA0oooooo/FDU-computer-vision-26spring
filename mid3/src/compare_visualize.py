import json
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

from .dataset import build_datasets
from .model_unet import build_model
from .single_visualize import colorize_mask, denormalize
from .utils import ensure_dir, get_device, load_config


NUM_IMAGES = 5
SPLIT = "val"
SEED = 42
OUTPUT_PATH = "outputs/compare.png"

EXPERIMENTS = [
    ("unet_ce", "U-Net CE", "configs/unet_ce.yaml"),
    ("unet_dice", "U-Net Dice", "configs/unet_dice.yaml"),
    ("unet_ce_dice", "U-Net CE + Dice", "configs/unet_ce_dice.yaml"),
]


def load_summary(output_dir):
    summary_path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_info(model_name, cfg):
    loss_name = cfg.get("loss", {}).get("name", "")
    summary = load_summary(cfg["output_dir"])
    best_miou = summary.get("best_val_mIoU")
    best_epoch = summary.get("best_epoch", "NA")
    if isinstance(best_miou, (int, float)):
        return f"{model_name} | loss={loss_name} | best epoch={best_epoch} | best val mIoU={best_miou:.4f}"
    return f"{model_name} | loss={loss_name}"


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
def predict(cfg, images, device):
    model = build_model(cfg).to(device)
    checkpoint_path = os.path.join(cfg["output_dir"], "checkpoints", "best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model(images.to(device)).argmax(dim=1).cpu()


def make_compare_figure():
    configs = [(name, display, load_config(config_path)) for name, display, config_path in EXPERIMENTS]
    images, masks = sample_batch(get_dataset(configs[0][2]))
    device = get_device(configs[0][2].get("device", "auto"))
    predictions = [(model_name, cfg, predict(cfg, images, device)) for model_name, _, cfg in configs]

    image_rows = 2 + len(configs)
    cols = images.size(0)
    fig = plt.figure(figsize=(cols * 3.0, image_rows * 3.0))
    grid = GridSpec(
        image_rows * 2,
        cols,
        figure=fig,
        height_ratios=sum(([1.0, 0.10] for _ in range(image_rows)), []),
    )

    row_specs = [
        ("origin", [denormalize(image) for image in images]),
        ("ground truth", [colorize_mask(mask) for mask in masks]),
    ]
    row_specs.extend((model_info(model_name, cfg), [colorize_mask(pred) for pred in preds]) for model_name, cfg, preds in predictions)

    for row, (label, row_images) in enumerate(row_specs):
        for col, image in enumerate(row_images):
            ax = fig.add_subplot(grid[row * 2, col])
            ax.imshow(image)
            ax.axis("off")

        label_ax = fig.add_subplot(grid[row * 2 + 1, :])
        label_ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
        label_ax.axis("off")

    ensure_dir(os.path.dirname(OUTPUT_PATH))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.02, hspace=0.08)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    return OUTPUT_PATH


def main():
    make_compare_figure()


if __name__ == "__main__":
    main()
