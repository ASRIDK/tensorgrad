"""End-to-end checks that the whole stack can actually learn."""

import numpy as np

from tensorgrad import functional as F
from tensorgrad import nn, no_grad, optim
from tensorgrad.data import DataLoader
from tensorgrad.tensor import Tensor


def test_mlp_fits_a_nonlinear_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=(512, 1))
    y = np.sin(x)
    model = nn.Sequential(nn.Linear(1, 32, rng=rng), nn.Tanh(), nn.Linear(32, 1, rng=rng))
    opt = optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(x, y, batch_size=64, rng=rng)
    for _ in range(150):
        for xb, yb in loader:
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()
    with no_grad():
        final = F.mse_loss(model(Tensor(x)), y).item()
    assert final < 1e-3


def test_classifier_learns_spirals():
    rng = np.random.default_rng(1)
    n, classes = 100, 3
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1.0, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n, c))
    x, y = np.concatenate(xs), np.concatenate(ys)

    model = nn.Sequential(
        nn.Linear(2, 64, rng=rng),
        nn.ReLU(),
        nn.Linear(64, 64, rng=rng),
        nn.ReLU(),
        nn.Linear(64, classes, rng=rng),
    )
    opt = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(model(Tensor(x)), y).backward()
        opt.step()
    with no_grad():
        accuracy = (model(Tensor(x)).data.argmax(axis=1) == y).mean()
    assert accuracy > 0.95
