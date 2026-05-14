try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class _DisabledRun:
    summary = {}


def init_wandb(cfg):
    if wandb is None or not cfg.get("wandb", {}).get("enabled", True):
        return _DisabledRun()

    return wandb.init(
        entity=cfg.get("wandb", {}).get("entity"),
        project=cfg.get("wandb", {}).get("project", "mid-project"),
        name=cfg["experiment_name"],
        config=cfg,
    )


def log_train_val(epoch, train_metrics, val_metrics, lr):
    if wandb is None or wandb.run is None:
        return

    log_dict = {
        "epoch": epoch,
        "train/loss": train_metrics["loss"],
        "train/pixel_acc": train_metrics["pixel_acc"],
        "train/mIoU": train_metrics["mIoU"],
        "train/foreground_iou": train_metrics["foreground_iou"],
        "train/background_iou": train_metrics["background_iou"],
        "train/boundary_iou": train_metrics["boundary_iou"],
        "val/loss": val_metrics["loss"],
        "val/pixel_acc": val_metrics["pixel_acc"],
        "val/mIoU": val_metrics["mIoU"],
        "val/foreground_iou": val_metrics["foreground_iou"],
        "val/background_iou": val_metrics["background_iou"],
        "val/boundary_iou": val_metrics["boundary_iou"],
        "lr": lr,
    }
    wandb.log(log_dict, step=epoch)


def finish_wandb():
    if wandb is not None and wandb.run is not None:
        wandb.finish()
