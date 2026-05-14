import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from .dataset import build_dataloaders
from .losses import build_loss
from .metrics import SegmentationMetric
from .model_unet import build_model
from .utils import count_parameters, ensure_dir, get_device, load_config, save_checkpoint, save_json, set_seed
from .wandb_utils import finish_wandb, init_wandb, log_train_val


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, cfg):
    scheduler_name = cfg["train"].get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, desc="", num_classes=3):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_pixels = 0
    metric = SegmentationMetric(num_classes=num_classes)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            pixels = masks.numel()
            total_loss += loss.item() * pixels
            total_pixels += pixels
            metric.update(logits.detach(), masks.detach())

    metrics = metric.compute()
    metrics["loss"] = total_loss / total_pixels
    return metrics


def flatten_metrics(prefix, metrics):
    return {
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_pixel_acc": metrics["pixel_acc"],
        f"{prefix}_mIoU": metrics["mIoU"],
        f"{prefix}_foreground_iou": metrics["foreground_iou"],
        f"{prefix}_background_iou": metrics["background_iou"],
        f"{prefix}_boundary_iou": metrics["boundary_iou"],
    }


def train_experiment(cfg):
    output_dir = cfg["output_dir"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)
    set_seed(cfg.get("seed", 42))
    save_json(cfg, os.path.join(output_dir, "config.json"))

    device = get_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    num_classes = cfg.get("model", {}).get("num_classes", 3)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    total_params, trainable_params = count_parameters(model)
    use_amp = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    run = init_wandb(cfg)

    train_steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)
    best_val_miou = -1.0
    best_epoch = 0
    best_val_metrics = None
    history = []
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            desc=f"Epoch {epoch}/{epochs} train",
            num_classes=num_classes,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            desc=f"Epoch {epoch}/{epochs} val",
            num_classes=num_classes,
        )

        lr = get_lr(optimizer)
        row = {
            "epoch": epoch,
            "train_steps_per_epoch": train_steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            **flatten_metrics("train", train_metrics),
            **flatten_metrics("val", val_metrics),
            "lr": lr,
        }
        history.append(row)

        is_best = val_metrics["mIoU"] >= best_val_miou
        if is_best:
            best_val_miou = val_metrics["mIoU"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mIoU": best_val_miou,
            "best_epoch": best_epoch,
            "cfg": cfg,
        }
        save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "last.pt"))
        if is_best:
            save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "best.pt"))

        pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
        summary = {
            "experiment_name": cfg["experiment_name"],
            "loss": cfg.get("loss", {}).get("name", "ce"),
            "best_epoch": best_epoch,
            "best_val_mIoU": best_val_miou,
            "best_val_pixel_acc": best_val_metrics["pixel_acc"],
            "best_val_foreground_iou": best_val_metrics["foreground_iou"],
            "best_val_background_iou": best_val_metrics["background_iou"],
            "best_val_boundary_iou": best_val_metrics["boundary_iou"],
            "last_epoch": epoch,
            "train_steps_per_epoch": train_steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "train_iterations_total": epoch * train_steps_per_epoch,
            "last_train_mIoU": train_metrics["mIoU"],
            "last_val_mIoU": val_metrics["mIoU"],
            "last_val_pixel_acc": val_metrics["pixel_acc"],
            "last_val_foreground_iou": val_metrics["foreground_iou"],
            "last_val_background_iou": val_metrics["background_iou"],
            "last_val_boundary_iou": val_metrics["boundary_iou"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_checkpoint": os.path.join(checkpoint_dir, "best.pt"),
            "last_checkpoint": os.path.join(checkpoint_dir, "last.pt"),
        }
        save_json(summary, os.path.join(output_dir, "summary.json"))

        log_train_val(epoch, train_metrics, val_metrics, lr)
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_mIoU"] = best_val_miou

        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_mIoU={train_metrics['mIoU']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_mIoU={val_metrics['mIoU']:.4f} val_pixel_acc={val_metrics['pixel_acc']:.4f}"
        )

    finish_wandb()
    print(f"Best epoch: {best_epoch}, best val_mIoU: {best_val_miou:.4f}")

    return {
        "history": history,
        "best_val_mIoU": best_val_miou,
        "best_epoch": best_epoch,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_experiment(cfg)


if __name__ == "__main__":
    main()
