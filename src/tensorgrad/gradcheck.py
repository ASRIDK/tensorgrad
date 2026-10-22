"""Finite-difference gradient checking."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tensorgrad.tensor import Tensor


def numerical_grad(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float = 1e-6
) -> list[np.ndarray]:
    """Central-difference estimate of d sum(fn(*inputs)) / d input, for every input."""
    grads = []
    for tensor in inputs:
        grad = np.zeros_like(tensor.data)
        it = np.nditer(tensor.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = tensor.data[idx]
            tensor.data[idx] = original + eps
            plus = fn(*inputs).data.sum()
            tensor.data[idx] = original - eps
            minus = fn(*inputs).data.sum()
            tensor.data[idx] = original
            grad[idx] = (plus - minus) / (2 * eps)
        grads.append(grad)
    return grads


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Compares analytic gradients against central differences.

    Raises AssertionError with the worst mismatch if they disagree.
    """
    for tensor in inputs:
        tensor.grad = None
    fn(*inputs).sum().backward()
    expected = numerical_grad(fn, inputs, eps)
    for i, (tensor, numeric) in enumerate(zip(inputs, expected, strict=True)):
        analytic = tensor.grad if tensor.grad is not None else np.zeros_like(tensor.data)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            worst = np.max(np.abs(analytic - numeric))
            raise AssertionError(f"gradient mismatch for input {i}: max abs error {worst:.3e}")
    return True
