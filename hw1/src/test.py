from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from src.dataloader import load_fashion_mnist, make_batches
    from src.losses import cross_entropy
    from src.modules import MLP
    from src.tensor import Tensor
    from src.utils import load_config, load_model, plot_confusion, save_json, set_seed, set_threads
else:
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
    set_seed(int(config.get("seed", 42)))
    set_threads(int(config.get("num_threads", 1)))

    meta = {}
    meta_path = Path(config.get("meta_path", "artifacts/logs/best.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    _, _, test_images, test_labels = load_fashion_mnist(config.get("data_dir", "data/raw"))

    model = MLP(
        input_dim=int(meta.get("input_dim", config.get("input_dim", 784))),
        hidden_dim=int(meta.get("hidden_dim", config.get("hidden_dim", 256))),
        output_dim=int(meta.get("output_dim", config.get("output_dim", 10))),
        activation=str(meta.get("activation", config.get("activation", "relu"))),
    )
    load_model(model, config.get("checkpoint_path", "artifacts/checkpoints/best.npz"))

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[np.ndarray] = []

    for batch_images, batch_labels in make_batches(
        test_images,
        test_labels,
        batch_size=int(config.get("batch_size", 256)),
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

    plot_confusion(matrix, config.get("confusion_path", "artifacts/eval/confusion.png"), class_names=CLASS_NAMES)
    confusion_csv_path = Path(config.get("confusion_csv_path", "artifacts/eval/confusion.csv"))
    confusion_csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(confusion_csv_path, matrix, fmt="%d", delimiter=",")

    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "num_test_samples": int(total_samples),
    }
    save_json(summary, config.get("summary_path", "artifacts/eval/summary.json"))
    print(f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")
    return summary


if __name__ == "__main__":
    test_model()
