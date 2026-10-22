"""Gradient-based optimizers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tensorgrad.nn import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer received an empty parameter list")
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum and L2 decay."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity, strict=True):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += grad
                grad = grad + self.momentum * v if self.nesterov else v
            p.data -= self.lr * grad


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias-corrected moment estimates.

    ``weight_decay`` adds an L2 penalty to the gradient; see :class:`AdamW` for
    the decoupled variant.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _gradient(self, p: Parameter) -> np.ndarray:
        return p.grad + self.weight_decay * p.data

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            grad = self._gradient(p)
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad * grad
            p.data -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay)

    def _gradient(self, p: Parameter) -> np.ndarray:
        return p.grad

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * self.weight_decay * p.data
        super().step()


def clip_grad_norm(params: Iterable[Parameter], max_norm: float) -> float:
    """Rescales gradients in place so their global L2 norm is at most `max_norm`.

    Returns the norm before clipping.
    """
    grads = [p.grad for p in params if p.grad is not None]
    total = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for g in grads:
            g *= scale
    return total


class CosineAnnealingLR:
    """Decays an optimizer's learning rate from its initial value to `min_lr`
    along a half cosine over `total_steps` calls to :meth:`step`."""

    def __init__(self, optimizer: Optimizer, total_steps: int, min_lr: float = 0.0) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.steps = 0

    def step(self) -> None:
        self.steps = min(self.steps + 1, self.total_steps)
        progress = self.steps / self.total_steps
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        self.optimizer.lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
