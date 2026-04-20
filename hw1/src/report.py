from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from src.dataloader import load_fashion_mnist
    from src.modules import MLP
    from src.tensor import Tensor
    from src.utils import copy_file, load_config, load_model, plot_errors, plot_top_weight_neurons, plot_weights, set_seed, set_threads
else:
    from .dataloader import load_fashion_mnist
    from .modules import MLP
    from .tensor import Tensor
    from .utils import copy_file, load_config, load_model, plot_errors, plot_top_weight_neurons, plot_weights, set_seed, set_threads


def build_report_assets(
    train_config_path: str | Path = "configs/train.yaml",
    test_config_path: str | Path = "configs/test.yaml",
) -> dict[str, str]:
    train_config = load_config(train_config_path)
    test_config = load_config(test_config_path)
    set_seed(int(test_config.get("seed", 42)))
    set_threads(int(test_config.get("num_threads", 1)))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_loss_src = Path(train_config.get("loss_plot_path", "artifacts/plots/train_loss.png"))
    val_acc_src = Path(train_config.get("val_acc_plot_path", "artifacts/plots/val_acc.png"))
    confusion_src = Path(test_config.get("confusion_path", "artifacts/eval/confusion.png"))

    train_loss_dst = reports_dir / "train_loss.png"
    val_acc_dst = reports_dir / "val_acc.png"
    confusion_dst = reports_dir / "confusion.png"

    if train_loss_src.exists():
        copy_file(train_loss_src, train_loss_dst)
    if val_acc_src.exists():
        copy_file(val_acc_src, val_acc_dst)
    if confusion_src.exists():
        copy_file(confusion_src, confusion_dst)

    meta = {}
    meta_path = Path(test_config.get("meta_path", "artifacts/logs/best.json"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    _, _, test_images, test_labels = load_fashion_mnist(test_config.get("data_dir", "data/raw"))
    model = MLP(
        input_dim=int(meta.get("input_dim", test_config.get("input_dim", 784))),
        hidden_dim=int(meta.get("hidden_dim", test_config.get("hidden_dim", 256))),
        output_dim=int(meta.get("output_dim", test_config.get("output_dim", 10))),
        activation=str(meta.get("activation", test_config.get("activation", "relu"))),
    )
    load_model(model, test_config.get("checkpoint_path", "artifacts/checkpoints/best.npz"))

    logits = model(Tensor(test_images))
    preds = np.argmax(logits.data, axis=1)

    error_examples_path = reports_dir / "errors.png"
    first_layer_weights_path = reports_dir / "weights.png"
    top_weight_neurons_path = reports_dir / "top25_neurons.png"

    plot_errors(
        test_images,
        test_labels,
        preds,
        error_examples_path,
        max_items=25,
    )
    plot_weights(model.fc1.weight.data, first_layer_weights_path)
    plot_top_weight_neurons(model.fc1.weight.data, top_weight_neurons_path, top_k=25)

    return {
        "train_loss": str(train_loss_dst),
        "val_acc": str(val_acc_dst),
        "confusion": str(confusion_dst),
        "errors": str(error_examples_path),
        "weights": str(first_layer_weights_path),
        "top25_neurons": str(top_weight_neurons_path),
    }


if __name__ == "__main__":
    build_report_assets()
