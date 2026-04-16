from __future__ import annotations

import numpy as np

from .modules import Module
from .tensor import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    batch_idx = np.arange(targets.shape[0])
    loss_value = -np.log(probs[batch_idx, targets] + 1e-12).mean()

    out = Tensor(loss_value, requires_grad=logits.requires_grad, op="cross_entropy", _prev=(logits,))

    def _backward() -> None:
        if logits.requires_grad:
            grad = probs.copy()
            grad[batch_idx, targets] -= 1.0
            grad /= targets.shape[0]
            logits.grad += out.grad * grad

    out._backward = _backward
    return out


def weight_decay(model: Module, lam: float) -> Tensor:
    penalty = None
    for name, param in model.named_parameters():
        if name.endswith("weight"):
            term = (param * param).sum()
            penalty = term if penalty is None else penalty + term

    if penalty is None:
        return Tensor(0.0)

    return penalty * (0.5 * lam)
