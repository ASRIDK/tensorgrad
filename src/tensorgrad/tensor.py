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
            if node._backward is not None:
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
