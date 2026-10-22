"""The Tensor type and the reverse-mode autodiff machinery."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

ArrayLike = np.ndarray | float | int | list | tuple


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sums `grad` over the axes that NumPy broadcasting expanded to reach `shape`."""
    # Leading axes added by broadcasting.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Axes that were stretched from size 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An n-dimensional array that records the operations applied to it.

    Calling :meth:`backward` on a scalar result propagates gradients to every
    tensor in its history that was created with ``requires_grad=True``.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _parents: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] | None = None
        self._parents = _parents
        self._op = _op

    # ------------------------------------------------------------------ basics

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{grad})"

    # --------------------------------------------------------- graph building

    @staticmethod
    def _lift(value: Tensor | ArrayLike) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value)

    def _make(self, data: np.ndarray, parents: tuple[Tensor, ...], op: str) -> Tensor:
        """Creates a result tensor that tracks gradients if any parent does."""
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad, parents if requires_grad else (), op)

    def _accumulate(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        grad = _unbroadcast(grad, self.shape)
        self.grad = grad.copy() if self.grad is None else self.grad + grad

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        other = self._lift(other)
        out = self._make(self.data + other.data, (self, other), "add")

        def backward() -> None:
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = backward
        return out

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        other = self._lift(other)
        out = self._make(self.data * other.data, (self, other), "mul")

        def backward() -> None:
            self._accumulate(out.grad * other.data)
            other._accumulate(out.grad * self.data)

        out._backward = backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise TypeError("only constant exponents are supported; use exp/log instead")
        out = self._make(self.data**exponent, (self,), "pow")

        def backward() -> None:
            self._accumulate(out.grad * exponent * self.data ** (exponent - 1))

        out._backward = backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        return self + (-self._lift(other))

    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        return self * self._lift(other) ** -1.0

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) - self

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) / self

    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        other = self._lift(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul operands must have at least 2 dimensions")
        out = self._make(self.data @ other.data, (self, other), "matmul")

        def backward() -> None:
            self._accumulate(out.grad @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = backward
        return out

    def __rmatmul__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) @ self

    # ------------------------------------------------------------- reductions

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self._accumulate(np.broadcast_to(grad, self.shape))

        out._backward = backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        axes = range(self.ndim) if axis is None else np.atleast_1d(axis)
        count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def max(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        value = self.data.max(axis=axis, keepdims=True)
        out = self._make(value if keepdims else value.squeeze(axis=axis), (self,), "max")

        def backward() -> None:
            # Split the gradient evenly between tied maxima.
            mask = (self.data == value).astype(np.float64)
            mask /= mask.sum(axis=axis, keepdims=True)
            grad = (
                out.grad if keepdims else np.expand_dims(out.grad, axis if axis is not None else ())
            )
            self._accumulate(mask * grad)

        out._backward = backward
        return out

    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> Tensor:
        """Numerically stable log(sum(exp(x))) along `axis`."""
        shift = self.data.max(axis=axis, keepdims=True)
        exp = np.exp(self.data - shift)
        total = exp.sum(axis=axis, keepdims=True)
        value = shift + np.log(total)
        out = self._make(value if keepdims else value.squeeze(axis=axis), (self,), "logsumexp")

        def backward() -> None:
            grad = out.grad if keepdims else np.expand_dims(out.grad, axis)
            self._accumulate(grad * exp / total)  # d/dx logsumexp = softmax(x)

        out._backward = backward
        return out

    # ------------------------------------------------------ elementwise maths

    def exp(self) -> Tensor:
        value = np.exp(self.data)
        out = self._make(value, (self,), "exp")

        def backward() -> None:
            self._accumulate(out.grad * value)

        out._backward = backward
        return out

    def log(self) -> Tensor:
        out = self._make(np.log(self.data), (self,), "log")

        def backward() -> None:
            self._accumulate(out.grad / self.data)

        out._backward = backward
        return out

    def sqrt(self) -> Tensor:
        return self**0.5

    def abs(self) -> Tensor:
        out = self._make(np.abs(self.data), (self,), "abs")

        def backward() -> None:
            self._accumulate(out.grad * np.sign(self.data))

        out._backward = backward
        return out

    def relu(self) -> Tensor:
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def backward() -> None:
            self._accumulate(out.grad * (self.data > 0))

        out._backward = backward
        return out

    def tanh(self) -> Tensor:
        value = np.tanh(self.data)
        out = self._make(value, (self,), "tanh")

        def backward() -> None:
            self._accumulate(out.grad * (1.0 - value**2))

        out._backward = backward
        return out

    def sigmoid(self) -> Tensor:
        # Stable for large |x|: 1/(1+e^-x) for x>=0 and e^x/(1+e^x) otherwise.
        e = np.exp(-np.abs(self.data))
        value = np.where(self.data >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        out = self._make(value, (self,), "sigmoid")

        def backward() -> None:
            self._accumulate(out.grad * value * (1.0 - value))

        out._backward = backward
        return out

    def softplus(self) -> Tensor:
        """log(1 + e^x), computed without overflow."""
        x = self.data
        out = self._make(np.logaddexp(0.0, x), (self,), "softplus")

        def backward() -> None:
            self._accumulate(out.grad / (1.0 + np.exp(-x)))

        out._backward = backward
        return out

    # ------------------------------------------------------- shape operations

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple | list):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), "reshape")

        def backward() -> None:
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        axes_or_none = axes if axes else None
        out = self._make(np.transpose(self.data, axes_or_none), (self,), "transpose")

        def backward() -> None:
            inverse = np.argsort(axes) if axes else None
            self._accumulate(np.transpose(out.grad, inverse))

        out._backward = backward
        return out

    @property
    def T(self) -> Tensor:  # noqa: N802 - mirrors the NumPy attribute
        return self.transpose()

    def __getitem__(self, index) -> Tensor:
        if isinstance(index, Tensor):
            index = index.data.astype(np.int64)
        out = self._make(self.data[index], (self,), "getitem")

        def backward() -> None:
            grad = np.zeros_like(self.data)
            np.add.at(grad, index, out.grad)  # handles repeated indices
            self._accumulate(grad)

        out._backward = backward
        return out

    # --------------------------------------------------------------- autodiff

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Back-propagates from this tensor through its recorded history.

        ``grad`` defaults to 1 and may only be omitted for scalar tensors.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)

        order = _topological_order(self)
        # Leaf gradients accumulate across calls; intermediate ones are recomputed.
        for node in order:
            if node._parents:
                node.grad = None
        self.grad = np.asarray(grad, dtype=np.float64).reshape(self.shape)
        for node in reversed(order):
            # Constants sit in the graph as operands but never propagate gradients.
            if node.requires_grad and node._backward is not None:
                node._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        """Returns a tensor sharing the data but cut off from the graph."""
        return Tensor(self.data)


def _topological_order(root: Tensor) -> list[Tensor]:
    """Returns every node reachable from `root`, parents before children."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, Iterable[Tensor]]] = [(root, iter(root._parents))]
    visited.add(id(root))
    while stack:
        node, parents = stack[-1]
        for parent in parents:
            if id(parent) not in visited:
                visited.add(id(parent))
                stack.append((parent, iter(parent._parents)))
                break
        else:
            stack.pop()
            order.append(node)
    return order
