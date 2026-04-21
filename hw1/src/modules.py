from __future__ import annotations
import numpy as np
from .tensor import Parameter, Tensor
from .utils import he_init, xavier_init

HE_ACTIVATIONS = {"relu", "swilu"}


def _init_weight(in_dim: int, out_dim: int, activation: str) -> np.ndarray:
    if activation in HE_ACTIVATIONS:
        return he_init(in_dim, out_dim)
    return xavier_init(in_dim, out_dim)


def _activate(x: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return x.relu()
    if activation == "softmax":
        return x.softmax(axis=1)
    if activation == "tanh":
        return x.tanh()
    if activation == "swilu":
        return x.swilu()
    raise ValueError(f"Unsupported activation: {activation}")


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> list[Parameter]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        named = []
        for name, param in self._parameters.items():
            full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            named.append((full_name, param))
        for name, module in self._modules.items():
            child_prefix = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            named.extend(module.named_parameters(child_prefix))
        return named

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", name: str = "") -> None:
        super().__init__()
        weight = _init_weight(in_dim, out_dim, activation)

        prefix = f"{name}." if name else ""
        self.weight = Parameter(weight, name=f"{prefix}weight")
        self.bias = Parameter(np.zeros(out_dim, dtype=weight.dtype), name=f"{prefix}bias")

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias


class MLP(Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.fc1 = Linear(input_dim, hidden_dim, activation=activation, name="fc1")
        self.fc2 = Linear(hidden_dim, output_dim, activation=activation, name="fc2")

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.fc1(x)
        hidden = _activate(hidden, self.activation)
        return self.fc2(hidden)
