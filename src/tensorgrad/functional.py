"""Stateless functions: activations over axes and loss functions."""

from __future__ import annotations

import numpy as np

from tensorgrad.tensor import ArrayLike, Tensor


def _as_tensor(value: Tensor | ArrayLike) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x - x.logsumexp(axis=axis, keepdims=True)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis=axis).exp()


def mse_loss(prediction: Tensor, target: Tensor | ArrayLike) -> Tensor:
    """Mean squared error."""
    diff = prediction - _as_tensor(target)
    return (diff * diff).mean()


def huber_loss(prediction: Tensor, target: Tensor | ArrayLike, delta: float = 1.0) -> Tensor:
    """Quadratic for small errors and linear for large ones; robust to outliers."""
    error = (prediction - _as_tensor(target)).abs()
    quadratic_mask = Tensor((error.data <= delta).astype(np.float64))
    quadratic = 0.5 * error * error
    linear = delta * (error - 0.5 * delta)
    return (quadratic * quadratic_mask + linear * (1.0 - quadratic_mask)).mean()


def cross_entropy(logits: Tensor, targets: np.ndarray | list[int]) -> Tensor:
    """Mean negative log-likelihood of integer class `targets` under softmax(logits).

    `logits` has shape (batch, classes).
    """
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits of shape (N, C) and targets of shape (N,)")
    log_probs = log_softmax(logits, axis=1)
    return -log_probs[np.arange(len(targets)), targets].mean()


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor | ArrayLike) -> Tensor:
    """Mean binary cross-entropy, computed stably from raw scores.

    Uses the identity  BCE(sigmoid(z), y) = softplus(z) - y * z.
    """
    return (logits.softplus() - _as_tensor(targets) * logits).mean()
