import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from .model_unet import build_model
from .utils import ensure_dir, get_device, load_config


MASK_COLORS = np.array(
    [
        [255, 0, 0],
        [0, 0, 0],
        [255, 220, 0],
    ],
    dtype=np.uint8,
)


def denormalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.asarray(mean).reshape(1, 1, 3)
    std = np.asarray(std).reshape(1, 1, 3)
    image = image * std + mean
    return np.clip(image, 0.0, 1.0)


def colorize_mask(mask):
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    return MASK_COLORS[mask_np]


@torch.no_grad()
def make_prediction_figure(cfg, checkpoint_path=None, split="val", num_examples=4):
    output_dir = cfg["output_dir"]
    figure_dir = os.path.join(output_dir, "figures")
    ensure_dir(figure_dir)
    checkpoint_path = checkpoint_path or os.path.join(output_dir, "checkpoints", "best.pt")

    device = get_device(cfg.get("device", "auto"))
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images, masks = next(iter(loader))
    images = images[:num_examples].to(device)
    masks = masks[:num_examples]
    preds = model(images).argmax(dim=1).cpu()

    rows = images.size(0)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    for i in range(rows):
        axes[i, 0].imshow(denormalize(images[i]))
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(colorize_mask(masks[i]))
        axes[i, 1].set_title("ground truth")
        axes[i, 2].imshow(colorize_mask(preds[i]))
        axes[i, 2].set_title("prediction")
        for j in range(3):
            axes[i, j].axis("off")

    fig.tight_layout()
    save_path = os.path.join(figure_dir, "pred_examples.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_examples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    make_prediction_figure(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()
