from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataloader import load_fashion_mnist, make_batches
from .losses import cross_entropy
from .modules import MLP
from .tensor import Tensor
from .utils import load_config, load_model, plot_confusion, save_json, set_seed, set_threads

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def _print_confusion_matrix(matrix: np.ndarray) -> None:
    print("confusion_matrix:")
    for idx, row in enumerate(matrix):
        print(f"{CLASS_NAMES[idx]:>12}: {row.tolist()}")


def test_model(config_path: str | Path = "configs/test.yaml") -> dict[str, float]:
    config = load_config(config_path)
    seed = int(config["seed"])
    num_threads = int(config["num_threads"])
    data_dir = config["data_dir"]
    checkpoint_path = config["checkpoint_path"]
    meta_path = Path(config["meta_path"])
    batch_size = int(config["batch_size"])
    confusion_path = config["confusion_path"]
    confusion_csv_path = Path(config["confusion_csv_path"])
    summary_path = config["summary_path"]

    set_seed(seed)
    set_threads(num_threads)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    _, _, test_images, test_labels = load_fashion_mnist(data_dir)

    model = MLP(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        output_dim=int(meta["output_dim"]),
        activation=str(meta["activation"]),
    )
    load_model(model, checkpoint_path)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[np.ndarray] = []

    for batch_images, batch_labels in make_batches(
        test_images,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model(Tensor(batch_images))
        loss = cross_entropy(logits, batch_labels)
        preds = np.argmax(logits.data, axis=1)
        batch_size_now = batch_labels.shape[0]
        total_loss += float(loss.data) * batch_size_now
        total_correct += int((preds == batch_labels).sum())
        total_samples += batch_size_now
        all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)
    matrix = _confusion_matrix(test_labels, preds, num_classes=10)
    test_acc = total_correct / total_samples
    test_loss = total_loss / total_samples
    _print_confusion_matrix(matrix)

    plot_confusion(matrix, confusion_path, class_names=CLASS_NAMES)
    confusion_csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(confusion_csv_path, matrix, fmt="%d", delimiter=",")

    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "num_test_samples": int(total_samples),
    }
    save_json(summary, summary_path)
    print(f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")
    return summary


if __name__ == "__main__":
    test_model()
