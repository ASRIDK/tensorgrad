"""Learn the Black-Scholes call price with a neural network, then recover the
Greeks by differentiating through the trained network with tensorgrad.

Prices are homogeneous of degree one in (S, K), so the network learns
    C / K = f(S / K, T, sigma, r)
on randomly sampled market scenarios. Delta (dC/dS) and vega (dC/dsigma) are
then obtained by back-propagating through the network to its raw inputs and
compared against their closed-form values.

Usage:
    python examples/black_scholes_pricer.py [--samples N] [--epochs E] [--seed S]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from tensorgrad import Tensor, nn, no_grad, optim
from tensorgrad import functional as F
from tensorgrad.data import DataLoader, StandardScaler, train_test_split

# Sampling ranges for (moneyness S/K, maturity T in years, volatility, rate).
RANGES = {
    "moneyness": (0.6, 1.4),
    "maturity": (0.05, 2.0),
    "volatility": (0.05, 0.6),
    "rate": (0.0, 0.08),
}

_erf = np.vectorize(math.erf)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes(m, t, sigma, r):
    """Closed-form call price per unit strike, delta and vega (per unit strike)."""
    sqrt_t = np.sqrt(t)
    d1 = (np.log(m) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    price = m * norm_cdf(d1) - np.exp(-r * t) * norm_cdf(d2)
    delta = norm_cdf(d1)
    vega = m * norm_pdf(d1) * sqrt_t
    return price, delta, vega


def sample_scenarios(n: int, rng: np.random.Generator) -> np.ndarray:
    columns = [rng.uniform(low, high, n) for low, high in RANGES.values()]
    return np.column_stack(columns)


def build_model(rng: np.random.Generator) -> nn.Module:
    # Smooth activations give smooth (and therefore usable) derivatives.
    return nn.Sequential(
        nn.Linear(4, 64, init="xavier", rng=rng),
        nn.Tanh(),
        nn.Linear(64, 64, init="xavier", rng=rng),
        nn.Tanh(),
        nn.Linear(64, 64, init="xavier", rng=rng),
        nn.Tanh(),
        nn.Linear(64, 1, init="xavier", rng=rng),
    )


class Pricer:
    """Wraps the network with input standardization expressed as tensor ops, so
    gradients flow all the way back to the raw market inputs."""

    def __init__(self, model: nn.Module, scaler: StandardScaler) -> None:
        self.model = model
        self.mean = Tensor(scaler.mean)
        self.std = Tensor(scaler.std)

    def __call__(self, raw_inputs: Tensor) -> Tensor:
        return self.model((raw_inputs - self.mean) / self.std)


def train(pricer: Pricer, x: np.ndarray, y: np.ndarray, epochs: int, rng) -> list[float]:
    opt = optim.Adam(pricer.model.parameters(), lr=3e-3)
    loader = DataLoader(x, y, batch_size=256, rng=rng)
    schedule = optim.CosineAnnealingLR(opt, total_steps=epochs * len(loader), min_lr=1e-5)
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.mse_loss(pricer(xb), yb)
            loss.backward()
            opt.step()
            schedule.step()
            total += loss.item() * len(yb)
        history.append(total / len(x))
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x = sample_scenarios(args.samples, rng)
    price, delta, vega = black_scholes(*x.T)
    x_train, x_test, y_train, y_test, _, delta_test, _, vega_test = train_test_split(
        x, price.reshape(-1, 1), delta, vega, test_size=0.2, rng=rng
    )

    pricer = Pricer(build_model(rng), StandardScaler().fit(x_train))
    print(f"model: {pricer.model.num_parameters()} parameters, {len(x_train)} training scenarios")

    start = time.perf_counter()
    history = train(pricer, x_train, y_train, args.epochs, rng)
    elapsed = time.perf_counter() - start
    for epoch in sorted({0, len(history) // 4, len(history) // 2, len(history) - 1}):
        print(f"  epoch {epoch + 1:3d}  train MSE {history[epoch]:.3e}")
    print(f"training time: {elapsed:.1f}s")

    # Prices.
    with no_grad():
        predicted = pricer(Tensor(x_test)).data
    errors = predicted - y_test
    print("\nprice (per unit strike) on held-out scenarios")
    print(f"  RMSE        {np.sqrt(np.mean(errors**2)):.2e}")
    print(f"  mean |err|  {np.mean(np.abs(errors)):.2e}")
    print(f"  max |err|   {np.max(np.abs(errors)):.2e}")

    # Greeks by automatic differentiation through the network.
    inputs = Tensor(x_test, requires_grad=True)
    pricer(inputs).sum().backward()  # rows are independent, so this yields per-row gradients
    ad_delta, ad_vega = inputs.grad[:, 0], inputs.grad[:, 2]
    print("\nGreeks via autodiff vs closed form")
    print(f"  delta  mean |err| {np.mean(np.abs(ad_delta - delta_test)):.2e}")
    print(f"  vega   mean |err| {np.mean(np.abs(ad_vega - vega_test)):.2e}")

    print("\nsample at-the-money quotes (S=K=100, r=2%)")
    print("   T    sigma   BS price   NN price   BS delta   NN delta")
    for t, sigma in [(0.25, 0.2), (1.0, 0.2), (1.0, 0.4)]:
        point = np.array([[1.0, t, sigma, 0.02]])
        bs_price, bs_delta, _ = black_scholes(*point.T)
        tensor_point = Tensor(point, requires_grad=True)
        nn_price = pricer(tensor_point)
        nn_price.sum().backward()
        print(
            f"  {t:4.2f}  {sigma:5.2f}  {100 * bs_price[0]:9.4f}  {100 * nn_price.item():9.4f}"
            f"  {bs_delta[0]:9.4f}  {tensor_point.grad[0, 0]:9.4f}"
        )


if __name__ == "__main__":
    main()
