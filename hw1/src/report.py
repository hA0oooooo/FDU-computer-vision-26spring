from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataloader import load_fashion_mnist
from .modules import MLP
from .tensor import Tensor
from .test import CLASS_NAMES
from .utils import load_config, load_model, plot_errors, plot_top_weight_neurons, set_seed, set_threads


def build_report_assets(
    train_config_path: str | Path = "configs/train.yaml",
    test_config_path: str | Path = "configs/test.yaml",
) -> dict[str, str]:
    test_config = load_config(test_config_path)
    set_seed(int(test_config["seed"]))
    set_threads(int(test_config["num_threads"]))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(test_config["meta_path"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    _, _, test_images, test_labels = load_fashion_mnist(test_config["data_dir"])
    model = MLP(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        output_dim=int(meta["output_dim"]),
        activation=str(meta["activation"]),
    )
    load_model(model, test_config["checkpoint_path"])

    logits = model(Tensor(test_images))
    preds = np.argmax(logits.data, axis=1)

    error_examples_path = reports_dir / "errors.png"
    top_weight_neurons_path = reports_dir / "neurons.png"

    plot_errors(
        test_images,
        test_labels,
        preds,
        error_examples_path,
        class_names=CLASS_NAMES,
        max_items=25,
    )
    plot_top_weight_neurons(model.fc1.weight.data, top_weight_neurons_path, top_k=25)

    return {
        "errors": str(error_examples_path),
        "neurons": str(top_weight_neurons_path),
    }


if __name__ == "__main__":
    build_report_assets()
