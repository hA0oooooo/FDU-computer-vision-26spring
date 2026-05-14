import argparse
import os

import pandas as pd
import torch

from .dataset import build_dataloaders
from .losses import build_loss
from .model_unet import build_model
from .train import run_epoch
from .utils import count_parameters, ensure_dir, get_device, load_config


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def select_loader(cfg, split):
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unknown split: {split}")


def evaluate_experiment(cfg, checkpoint_path=None, split="val"):
    output_dir = cfg["output_dir"]
    root_output_dir = "outputs"
    device = get_device(cfg.get("device", "auto"))
    checkpoint_path = checkpoint_path or os.path.join(output_dir, "checkpoints", "best.pt")

    loader = select_loader(cfg, split)
    model = build_model(cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = build_loss(cfg)
    num_classes = cfg.get("model", {}).get("num_classes", 3)
    metrics = run_epoch(
        model,
        loader,
        criterion,
        device,
        use_amp=False,
        desc=split,
        num_classes=num_classes,
    )

    total_params, trainable_params = count_parameters(model)
    result = {
        "experiment_name": cfg["experiment_name"],
        "loss": cfg.get("loss", {}).get("name", "ce"),
        "split": split,
        "checkpoint": checkpoint_path,
        "loss_value": metrics["loss"],
        "pixel_acc": metrics["pixel_acc"],
        "foreground_iou": metrics["foreground_iou"],
        "background_iou": metrics["background_iou"],
        "boundary_iou": metrics["boundary_iou"],
        "mIoU": metrics["mIoU"],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    summary_path = os.path.join(root_output_dir, "eval.csv")
    ensure_dir(root_output_dir)
    pd.DataFrame([result]).to_csv(
        summary_path,
        mode="a",
        header=not os.path.exists(summary_path),
        index=False,
    )

    print(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    evaluate_experiment(cfg, checkpoint_path=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
