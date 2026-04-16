from __future__ import annotations
from typing import Callable, Iterable
import numpy as np


def sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    # 处理前向传播时的广播
    reduced = grad
    while reduced.ndim > len(shape):
        reduced = reduced.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)

    return reduced.reshape(shape)


def _to_array(data: np.ndarray | list | float | int) -> np.ndarray:
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.floating):
            return data
        return data.astype(np.float32)

    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.floating):
        return array
    return array.astype(np.float32)


def _ensure_tensor(value: Tensor | np.ndarray | list | float | int) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


class Tensor:
    def __init__(
        self,
        data: np.ndarray | list | float | int,
        requires_grad: bool = False,
        op: str = "",
        name: str = "",
        _prev: Iterable["Tensor"] = (),
    ) -> None:
        self.data = _to_array(data)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._prev = set(_prev)
        self._backward: Callable[[], None] = lambda: None
        self.op = op
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op='{self.op}')"

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad: np.ndarray | float | int | None = None) -> None:
        grad_array = np.ones_like(self.data) if grad is None else _to_array(grad)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

        self.grad = grad_array
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: Tensor | np.ndarray | list | float | int) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            op="add",
            _prev=(self, other),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad, self.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Tensor | np.ndarray | list | float | int) -> "Tensor":
        return self + other

    def __mul__(self, other: Tensor | np.ndarray | list | float | int) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            op="mul",
            _prev=(self, other),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += sum_to_shape(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += sum_to_shape(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Tensor | np.ndarray | list | float | int) -> "Tensor":
        return self * other

    def __matmul__(self, other: Tensor | np.ndarray | list | float | int) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            op="matmul",
            _prev=(self, other),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            op="sum",
            _prev=(self,),
        )

        def _backward() -> None:
            if not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                expanded = np.broadcast_to(grad, self.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                normalized = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                expanded = grad
                if not keepdims:
                    for ax in sorted(normalized):
                        expanded = np.expand_dims(expanded, axis=ax)
                expanded = np.broadcast_to(expanded, self.shape)

            self.grad += expanded

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            op="mean",
            _prev=(self,),
        )

        def _backward() -> None:
            if not self.requires_grad:
                return

            if axis is None:
                expanded = np.broadcast_to(out.grad / self.data.size, self.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                normalized = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                count = 1
                for ax in normalized:
                    count *= self.data.shape[ax]

                expanded = out.grad / count
                if not keepdims:
                    for ax in sorted(normalized):
                        expanded = np.expand_dims(expanded, axis=ax)
                expanded = np.broadcast_to(expanded, self.shape)

            self.grad += expanded

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            op="relu",
            _prev=(self,),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_data = np.tanh(self.data)
        out = Tensor(
            tanh_data,
            requires_grad=self.requires_grad,
            op="tanh",
            _prev=(self,),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (1.0 - tanh_data**2)

        out._backward = _backward
        return out

    def softmax(self, axis: int = 1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        softmax_data = exp_data / exp_data.sum(axis=axis, keepdims=True)
        out = Tensor(
            softmax_data,
            requires_grad=self.requires_grad,
            op="softmax",
            _prev=(self,),
        )

        def _backward() -> None:
            if self.requires_grad:
                dot = (out.grad * softmax_data).sum(axis=axis, keepdims=True)
                self.grad += softmax_data * (out.grad - dot)

        out._backward = _backward
        return out

    def swilu(self) -> "Tensor":
        sigmoid = 1.0 / (1.0 + np.exp(-self.data))
        swilu_data = self.data * sigmoid
        out = Tensor(
            swilu_data,
            requires_grad=self.requires_grad,
            op="swilu",
            _prev=(self,),
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad += out.grad * (sigmoid + self.data * sigmoid * (1.0 - sigmoid))

        out._backward = _backward
        return out

class Parameter(Tensor):
    def __init__(self, data: np.ndarray | list | float | int, name: str = "") -> None:
        super().__init__(data=data, requires_grad=True, op="parameter", name=name)
