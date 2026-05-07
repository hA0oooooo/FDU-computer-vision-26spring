import wandb


def init_wandb(cfg):
    return wandb.init(
        entity=cfg.get("wandb", {}).get("entity"),
        project=cfg.get("wandb", {}).get("project", "mid1-pet-resnet34"),
        name=cfg["experiment_name"],
        config=cfg,
    )


def log_train_val(epoch, train_loss, train_acc, val_loss, val_acc, lr_backbone, lr_head, lr_attention=None):
    log_dict = {
        "epoch": epoch,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "lr/backbone": lr_backbone,
        "lr/head": lr_head,
    }

    if lr_attention is not None:
        log_dict["lr/attention"] = lr_attention

    wandb.log(log_dict, step=epoch)


def finish_wandb():
    wandb.finish()
