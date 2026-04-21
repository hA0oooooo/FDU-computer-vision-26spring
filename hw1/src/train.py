from __future__ import annotations

from pathlib import Path

import numpy as np

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


def train_model(config_path: str | Path = "configs/train.yaml", config: dict | None = None) -> dict[str, float]:
    config = config if config is not None else load_config(config_path)
    seed = int(config["seed"])
    num_threads = int(config["num_threads"])
    data_dir = config["data_dir"]
    split_path = config["split_path"]
    checkpoint_path = config["checkpoint_path"]
    best_meta_path = config["best_meta_path"]
    history_path = config["history_path"]
    loss_plot_path = config["loss_plot_path"]
    val_acc_plot_path = config["val_acc_plot_path"]
    hidden_dim = int(config["hidden_dim"])
    activation = str(config["activation"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    base_lr = float(config["lr"])
    weight_decay_value = float(config["weight_decay"])
    val_ratio = float(config["val_ratio"])
    early_stopping = bool(config["early_stopping"])
    patience = int(config["patience"])
    min_delta = float(config["min_delta"])
    lr_decay_step = int(config["lr_decay_step"])
    lr_decay_gamma = float(config["lr_decay_gamma"])
    verbose = bool(config["verbose"])
    save_best_model_flag = bool(config["save_best_model"])
    save_best_meta_flag = bool(config["save_best_meta"])
    save_history_flag = bool(config["save_history"])
    save_plots_flag = bool(config["save_plots"])

    set_seed(seed)
    set_threads(num_threads)

    train_images, train_labels, _, _ = load_fashion_mnist(data_dir)
    x_train, y_train, x_val, y_val = split_train_val(
        train_images,
        train_labels,
        val_ratio=val_ratio,
        seed=seed,
        split_path=split_path,
    )

    model = MLP(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=10,
        activation=activation,
    )

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -np.inf
    best_epoch = 0
    patience_count = 0

    for epoch in range(epochs):
        lr = base_lr * (lr_decay_gamma ** (epoch // lr_decay_step))
        train_loss, train_acc = train_one_epoch(
            model,
            x_train,
            y_train,
            batch_size=batch_size,
            lr=lr,
            weight_decay_value=weight_decay_value,
            seed=seed + epoch,
        )
        val_loss, val_acc = eval_epoch(
            model,
            x_val,
            y_val,
            batch_size=batch_size,
            weight_decay_value=weight_decay_value,
        )

        history["epoch"].append(epoch + 1)
        history["lr"].append(float(lr))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        if verbose:
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
            if save_best_model_flag:
                save_model(model, checkpoint_path)
            if save_best_meta_flag:
                save_json(
                    {
                        "epoch": best_epoch,
                        "val_acc": float(best_val_acc),
                        "input_dim": int(x_train.shape[1]),
                        "hidden_dim": hidden_dim,
                        "output_dim": 10,
                        "activation": activation,
                        "batch_size": batch_size,
                        "seed": seed,
                        "lr": base_lr,
                        "weight_decay": weight_decay_value,
                    },
                    best_meta_path,
                )
        else:
            patience_count += 1

        if early_stopping and patience_count >= patience:
            break

    if save_history_flag:
        save_history(history, history_path)
    if save_plots_flag:
        plot_loss(history, loss_plot_path)
        plot_val_acc(history, val_acc_plot_path)

    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history["epoch"])),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "history": history,
    }


if __name__ == "__main__":
    train_model()
