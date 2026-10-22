"""Neural network building blocks."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from tensorgrad.tensor import Tensor


class Parameter(Tensor):
    """A tensor that is a learnable parameter of a :class:`Module`."""

    __slots__ = ()

    def __init__(self, data) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for layers and models.

    Parameters and sub-modules assigned as attributes (directly, or inside a
    list/tuple) are discovered automatically.
    """

    training: bool = True

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in vars(self).items():
            yield from _named_parameters_of(value, f"{prefix}{name}")

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def modules(self) -> Iterator[Module]:
        yield self
        for value in vars(self).values():
            for child in value if isinstance(value, list | tuple) else (value,):
                if isinstance(child, Module):
                    yield from child.modules()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> Module:
        for module in self.modules():
            module.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        if params.keys() != state.keys():
            raise KeyError(f"state_dict keys do not match: {sorted(set(params) ^ set(state))}")
        for name, p in params.items():
            if p.shape != state[name].shape:
                raise ValueError(f"shape mismatch for {name}: {p.shape} vs {state[name].shape}")
            p.data = np.array(state[name], dtype=np.float64)

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())


def _named_parameters_of(value, name: str) -> Iterator[tuple[str, Parameter]]:
    if isinstance(value, Parameter):
        yield name, value
    elif isinstance(value, Module):
        yield from value.named_parameters(prefix=f"{name}.")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            yield from _named_parameters_of(item, f"{name}.{i}")


class Linear(Module):
    """Affine map ``y = x @ W + b`` with ``W`` of shape (in_features, out_features).

    ``init="he"`` suits ReLU networks; ``init="xavier"`` suits tanh/sigmoid ones.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he",
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"unknown init scheme: {init!r}")
        self.weight = Parameter(rng.standard_normal((in_features, out_features)) * scale)
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Sequential(Module):
    """Applies its layers in order."""

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softplus(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.softplus()


class Dropout(Module):
    """Inverted dropout: zeroes activations with probability `p` during training."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * mask
