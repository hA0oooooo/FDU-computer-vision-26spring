import argparse
import os

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .dataset import build_dataloaders
from .models import build_model
from .utils import ensure_dir, get_device, load_config, save_checkpoint, save_json, set_seed
from .wandb_utils import finish_wandb, init_wandb, log_train_val


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]

    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    backbone_lr = train_cfg.get("backbone_lr", 1e-4)
    head_lr = train_cfg.get("head_lr", 1e-3)
    attention_lr = train_cfg.get("attention_lr", head_lr)

    backbone_params = []
    head_params = []
    attention_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("fc.") or name.startswith("head."):
            head_params.append(param)
        elif ".se." in name:
            attention_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "name": "head"},
    ]

    if len(attention_params) > 0:
        param_groups.append({"params": attention_params, "lr": attention_lr, "name": "attention"})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer, cfg):
    scheduler_name = cfg["train"].get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def get_group_lrs(optimizer):
    lrs = {"lr_backbone": None, "lr_head": None, "lr_attention": None}
    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "backbone":
            lrs["lr_backbone"] = group["lr"]
        elif name == "head":
            lrs["lr_head"] = group["lr"]
        elif name == "attention":
            lrs["lr_attention"] = group["lr"]
    return lrs


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, desc=""):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


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
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    run = init_wandb(cfg)
    train_steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)

    best_val_acc = -1.0
    best_epoch = 0
    history = []
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            desc=f"Epoch {epoch}/{epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            desc=f"Epoch {epoch}/{epochs} val",
        )

        lrs = get_group_lrs(optimizer)
        row = {
            "epoch": epoch,
            "train_steps_per_epoch": train_steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **lrs,
        }
        history.append(row)

        is_best = val_acc >= best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "cfg": cfg,
        }
        save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "last.pt"))

        if is_best:
            save_checkpoint(checkpoint, os.path.join(checkpoint_dir, "best.pt"))

        pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
        summary = {
            "experiment_name": cfg["experiment_name"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "last_epoch": epoch,
            "train_steps_per_epoch": train_steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "train_iterations_total": epoch * train_steps_per_epoch,
            "last_train_acc": train_acc,
            "last_val_acc": val_acc,
            "best_checkpoint": os.path.join(checkpoint_dir, "best.pt"),
            "last_checkpoint": os.path.join(checkpoint_dir, "last.pt"),
        }
        save_json(summary, os.path.join(output_dir, "summary.json"))

        log_train_val(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr_backbone=lrs["lr_backbone"],
            lr_head=lrs["lr_head"],
            lr_attention=lrs["lr_attention"],
        )
        run.summary["best_epoch"] = best_epoch
        run.summary["best_val_acc"] = best_val_acc

        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    finish_wandb()

    print(f"Best epoch: {best_epoch}, best val_acc: {best_val_acc:.4f}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
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
