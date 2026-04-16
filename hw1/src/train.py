from __future__ import annotations

from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from src.dataloader import load_fashion_mnist, make_batches, split_train_val
    from src.losses import cross_entropy, weight_decay
    from src.modules import MLP
    from src.tensor import Tensor
    from src.utils import (
        load_config,
        plot_loss,
        plot_val_acc,
        save_history,
        save_json,
        save_model,
        set_seed,
        set_threads,
    )
else:
    from .dataloader import load_fashion_mnist, make_batches, split_train_val
    from .losses import cross_entropy, weight_decay
    from .modules import MLP
    from .tensor import Tensor
    from .utils import (
        load_config,
        plot_loss,
        plot_val_acc,
        save_history,
        save_json,
        save_model,
        set_seed,
        set_threads,
    )


def train_one_epoch(
    model: MLP,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    batch_size: int,
    lr: float,
    weight_decay_value: float,
    seed: int,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_images, batch_labels in make_batches(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    ):
        for param in model.parameters():
            param.zero_grad()

        logits = model(Tensor(batch_images))
        loss = cross_entropy(logits, batch_labels) + weight_decay(model, weight_decay_value)
        loss.backward()

        for param in model.parameters():
            param.data -= lr * param.grad

        batch_size_now = batch_labels.shape[0]
        total_loss += float(loss.data) * batch_size_now
        total_correct += int((np.argmax(logits.data, axis=1) == batch_labels).sum())
        total_samples += batch_size_now

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(
    model: MLP,
    eval_images: np.ndarray,
    eval_labels: np.ndarray,
    batch_size: int,
    weight_decay_value: float,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_images, batch_labels in make_batches(
        eval_images,
        eval_labels,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model(Tensor(batch_images))
        loss = cross_entropy(logits, batch_labels) + weight_decay(model, weight_decay_value)
        batch_size_now = batch_labels.shape[0]
        total_loss += float(loss.data) * batch_size_now
        total_correct += int((np.argmax(logits.data, axis=1) == batch_labels).sum())
        total_samples += batch_size_now

    return total_loss / total_samples, total_correct / total_samples


def train_model(config_path: str | Path = "configs/train.yaml") -> dict[str, float]:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    set_threads(int(config.get("num_threads", 1)))

    train_images, train_labels, _, _ = load_fashion_mnist(config.get("data_dir", "data/raw"))
    x_train, y_train, x_val, y_val = split_train_val(
        train_images,
        train_labels,
        val_ratio=float(config.get("val_ratio", 0.2)),
        seed=int(config.get("seed", 42)),
        split_path=config.get("split_path", "data/splits/split_indices.npz"),
    )

    model = MLP(
        input_dim=x_train.shape[1],
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=10,
        activation=str(config.get("activation", "relu")),
    )

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    base_lr = float(config.get("lr", 0.1))
    lr_decay_step = int(config.get("lr_decay_step", 10))
    lr_decay_gamma = float(config.get("lr_decay_gamma", 0.5))
    patience = int(config.get("patience", 7))
    min_delta = float(config.get("min_delta", 1e-4))
    weight_decay_value = float(config.get("weight_decay", 1e-4))
    early_stopping = bool(config.get("early_stopping", True))

    best_val_acc = -np.inf
    best_epoch = 0
    patience_count = 0

    for epoch in range(int(config.get("epochs", 50))):
        lr = base_lr * (lr_decay_gamma ** (epoch // lr_decay_step))
        train_loss, train_acc = train_one_epoch(
            model,
            x_train,
            y_train,
            batch_size=int(config.get("batch_size", 128)),
            lr=lr,
            weight_decay_value=weight_decay_value,
            seed=int(config.get("seed", 42)) + epoch,
        )
        val_loss, val_acc = eval_epoch(
            model,
            x_val,
            y_val,
            batch_size=int(config.get("batch_size", 128)),
            weight_decay_value=weight_decay_value,
        )

        history["epoch"].append(epoch + 1)
        history["lr"].append(float(lr))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"epoch {epoch + 1:03d} | "
            f"lr={lr:.5f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_count = 0
            save_model(model, config.get("checkpoint_path", "artifacts/checkpoints/best_model.npz"))
            save_json(
                {
                    "epoch": best_epoch,
                    "val_acc": float(best_val_acc),
                    "input_dim": int(x_train.shape[1]),
                    "hidden_dim": int(config.get("hidden_dim", 256)),
                    "output_dim": 10,
                    "activation": str(config.get("activation", "relu")),
                    "batch_size": int(config.get("batch_size", 128)),
                    "seed": int(config.get("seed", 42)),
                },
                config.get("best_meta_path", "artifacts/logs/best_meta.json"),
            )
        else:
            patience_count += 1

        if early_stopping and patience_count >= patience:
            break

    save_history(history, config.get("history_path", "artifacts/logs/history.json"))
    plot_loss(history, config.get("loss_plot_path", "figures/loss.png"))
    plot_val_acc(history, config.get("val_acc_plot_path", "figures/val_acc.png"))

    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": float(best_epoch),
        "epochs_ran": float(len(history["epoch"])),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
    }


if __name__ == "__main__":
    train_model()
