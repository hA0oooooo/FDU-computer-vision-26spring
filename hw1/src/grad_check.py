from __future__ import annotations

import numpy as np

from .losses import cross_entropy, weight_decay
from .modules import MLP
from .tensor import Tensor
from .utils import set_seed


def _toy_loss(model: MLP, inputs: np.ndarray, targets: np.ndarray, lam: float) -> float:
    logits = model(Tensor(inputs))
    loss = cross_entropy(logits, targets) + weight_decay(model, lam)
    return float(loss.data)


def _check_activation(activation: str, seed: int, eps: float) -> dict[str, object]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    inputs = rng.normal(size=(4, 6)).astype(np.float64)
    targets = np.array([0, 1, 2, 1], dtype=np.int64)
    model = MLP(input_dim=6, hidden_dim=5, output_dim=3, activation=activation)
    for param in model.parameters():
        param.data = param.data.astype(np.float64)

    logits = model(Tensor(inputs))
    loss = cross_entropy(logits, targets) + weight_decay(model, 1e-3)
    for param in model.parameters():
        param.zero_grad()
    loss.backward()

    checks = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    for name, param in model.named_parameters():
        sample_count = min(2, param.data.size)
        sample_indices = rng.choice(param.data.size, size=sample_count, replace=False)

        for flat_idx in sample_indices:
            idx = np.unravel_index(flat_idx, param.data.shape)
            original = param.data[idx]

            param.data[idx] = original + eps
            loss_pos = _toy_loss(model, inputs, targets, 1e-3)

            param.data[idx] = original - eps
            loss_neg = _toy_loss(model, inputs, targets, 1e-3)

            param.data[idx] = original

            numerical = (loss_pos - loss_neg) / (2.0 * eps)
            analytical = float(param.grad[idx])
            abs_diff = abs(numerical - analytical)
            rel_diff = abs_diff / (abs(numerical) + abs(analytical) + 1e-12)

            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            checks.append(
                {
                    "name": name,
                    "index": tuple(int(i) for i in idx),
                    "analytical": analytical,
                    "numerical": numerical,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )

    summary = {
        "activation": activation,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "checks": checks,
    }

    return summary


def grad_check(seed: int = 42, eps: float = 1e-5) -> dict[str, object]:
    activations = ["relu", "softmax", "tanh", "swilu"]
    summaries = []

    for offset, activation in enumerate(activations):
        summary = _check_activation(activation, seed + offset, eps)
        summaries.append(summary)
        print(
            f"{activation}: "
            f"max_abs_diff={summary['max_abs_diff']:.6e} | "
            f"max_rel_diff={summary['max_rel_diff']:.6e}"
        )

    return {
        "activations": activations,
        "summaries": summaries,
        "max_abs_diff": max(item["max_abs_diff"] for item in summaries),
        "max_rel_diff": max(item["max_rel_diff"] for item in summaries),
    }


if __name__ == "__main__":
    grad_check()
