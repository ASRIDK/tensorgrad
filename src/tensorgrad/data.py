"""Dataset utilities: batching, splitting and feature scaling."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from tensorgrad.tensor import Tensor


class DataLoader:
    """Iterates over (inputs, targets) in mini-batches of Tensors.

    Reshuffles on every pass when `shuffle` is true.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        n = len(self.inputs)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, np.ndarray]]:
        n = len(self.inputs)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            yield Tensor(self.inputs[idx]), self.targets[idx]


def train_test_split(
    *arrays: np.ndarray, test_size: float = 0.2, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    """Shuffles and splits arrays; returns [a_train, a_test, b_train, b_test, ...]."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")
    rng = rng if rng is not None else np.random.default_rng()
    order = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx, train_idx = order[:n_test], order[n_test:]
    out = []
    for a in arrays:
        a = np.asarray(a)
        out.extend([a[train_idx], a[test_idx]])
    return out


class StandardScaler:
    """Rescales features to zero mean and unit variance, column by column."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0  # leave constant columns unscaled
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def _check_fitted(self) -> None:
        if self.mean is None:
            raise RuntimeError("StandardScaler must be fitted first")
