from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def set_threads(num_threads: int) -> None:
    value = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def he_init(in_dim: int, out_dim: int) -> np.ndarray:
    scale = np.sqrt(2.0 / in_dim)
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)


def xavier_init(in_dim: int, out_dim: int) -> np.ndarray:
    scale = np.sqrt(2.0 / (in_dim + out_dim))
    return (np.random.randn(in_dim, out_dim) * scale).astype(np.float32)


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == targets).mean())


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: param.data for name, param in model.named_parameters()}
    np.savez(path, **state)


def load_model(model, path: str | Path) -> None:
    state = np.load(Path(path))
    for name, param in model.named_parameters():
        param.data = state[name].astype(param.data.dtype, copy=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value)}")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)


def save_history(history: dict[str, list[float]], path: str | Path) -> None:
    save_json(history, path)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _read_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _read_scalar(value)
    return config


def plot_loss(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train loss", linewidth=2, color="tab:blue")
    plt.plot(history["epoch"], history["val_loss"], label="val loss", linewidth=2, color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_val_acc(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["val_acc"], label="val acc", linewidth=2, color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(
    matrix: np.ndarray,
    path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(matrix.shape[0])
    labels = class_names if class_names is not None else [str(i) for i in ticks]
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black", fontsize=8)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_errors(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    path: str | Path,
    class_names: list[str] | None = None,
    max_items: int = 25,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrong_idx = np.flatnonzero(labels != preds)[:max_items]

    if wrong_idx.size == 0:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "No errors", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return

    cols = 5
    rows = int(np.ceil(wrong_idx.size / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.4))

    for plot_idx, sample_idx in enumerate(wrong_idx, start=1):
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(images[sample_idx].reshape(28, 28), cmap="gray")
        true_label = class_names[labels[sample_idx]] if class_names is not None else str(labels[sample_idx])
        pred_label = class_names[preds[sample_idx]] if class_names is not None else str(preds[sample_idx])
        plt.title(f"T:{true_label}\nP:{pred_label}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_top_weight_neurons(weight: np.ndarray, path: str | Path, top_k: int = 25) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(weight, axis=0)
    top_indices = np.argsort(norms)[::-1][:top_k]
    cols = 5
    rows = int(np.ceil(len(top_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    vmax = float(np.abs(weight[:, top_indices]).max())
    image = None

    for plot_idx, neuron_idx in enumerate(top_indices, start=1):
        row = (plot_idx - 1) // cols
        col = (plot_idx - 1) % cols
        ax = axes[row, col]
        image = ax.imshow(weight[:, neuron_idx].reshape(28, 28), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"||w||={norms[neuron_idx]:.2f}", fontsize=8, labelpad=4)

    for empty_idx in range(len(top_indices), rows * cols):
        row = empty_idx // cols
        col = empty_idx % cols
        axes[row, col].axis("off")

    fig.subplots_adjust(right=0.9, wspace=0.25, hspace=0.45)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(image, cax=cbar_ax)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
