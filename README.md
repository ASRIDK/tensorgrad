# tensorgrad

[![CI](https://github.com/ASRIDK/tensorgrad/actions/workflows/ci.yml/badge.svg)](https://github.com/ASRIDK/tensorgrad/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**tensorgrad** is a reverse-mode automatic differentiation engine and a compact neural network
library written from scratch on top of NumPy. It provides a `Tensor` type that records every
operation applied to it, a PyTorch-style `nn` module system, standard losses and optimizers, and
data utilities — in roughly a thousand lines of readable, fully tested Python.

The flagship example applies it to quantitative finance: a neural network learns the
**Black-Scholes** pricing function, and the option **Greeks** are then obtained by differentiating
through the trained network — the same idea behind differential machine learning and AAD
(adjoint algorithmic differentiation) used for risk on derivatives books.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Example: neural option pricing with autodiff Greeks](#example-neural-option-pricing-with-autodiff-greeks)
- [How it works](#how-it-works)
- [API overview](#api-overview)
- [Testing](#testing)
- [Project history](#project-history)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- **Autograd engine** — dynamic computation graphs, reverse-mode differentiation with an
  iterative topological sort (no recursion-depth limits), gradient accumulation and `no_grad()`.
- **NumPy broadcasting** — gradients are automatically reduced back to each operand's shape.
- **Operators** — arithmetic, `matmul` (including batched), `sum`/`mean`/`max`, a numerically
  stable `logsumexp`, `exp`, `log`, `sqrt`, `abs`, `relu`, `tanh`, stable `sigmoid` and `softplus`,
  `reshape`, `transpose` and fancy indexing.
- **Neural networks** — `Module` with recursive parameter discovery, `train()`/`eval()`,
  `state_dict()`; `Linear` (He or Xavier initialization), `Sequential`, activations and `Dropout`.
- **Losses** — MSE, Huber, cross-entropy (via log-softmax) and binary cross-entropy with logits.
- **Optimizers** — SGD with (Nesterov) momentum, Adam, AdamW, cosine learning-rate annealing and
  gradient-norm clipping.
- **Data** — mini-batch `DataLoader`, `train_test_split` and `StandardScaler`.
- **Verified** — every differentiable operation is checked against central finite differences.

## Installation

```bash
git clone https://github.com/ASRIDK/tensorgrad.git
cd tensorgrad
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The only runtime dependency is NumPy.

## Quick start

```python
import numpy as np
from tensorgrad import Tensor, nn, optim
from tensorgrad import functional as F

# Automatic differentiation
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x @ x.T).tanh().sum()
y.backward()
print(x.grad)

# Training a small network to learn f(a, b) = a * b
rng = np.random.default_rng(0)
model = nn.Sequential(nn.Linear(2, 16, rng=rng), nn.ReLU(), nn.Linear(16, 1, rng=rng))
opt = optim.Adam(model.parameters(), lr=1e-2)
inputs = rng.normal(size=(256, 2))
targets = inputs[:, :1] * inputs[:, 1:]

for step in range(500):
    opt.zero_grad()
    loss = F.mse_loss(model(Tensor(inputs)), targets)
    loss.backward()
    opt.step()
```

## Example: neural option pricing with autodiff Greeks

[`examples/black_scholes_pricer.py`](examples/black_scholes_pricer.py) trains a 3×64 `tanh`
network on 32,000 random market scenarios. Using the homogeneity of the call price in `(S, K)`,
it learns

```text
C / K = f(S / K, T, σ, r)       S/K ∈ [0.6, 1.4], T ∈ [0.05, 2], σ ∈ [0.05, 0.6], r ∈ [0, 8%]
```

Input standardization is expressed as tensor operations inside the pricer, so a single backward
pass returns the derivatives of the price with respect to the **raw** market inputs. Column 0 of
the input gradient is **delta** (`∂C/∂S`) and column 2 is **vega** (`∂C/∂σ`); neither was
part of the training targets.

```bash
python examples/black_scholes_pricer.py      # ~15 s on a laptop CPU
```

Results on 8,000 held-out scenarios (seed 7, 200 epochs, Adam with cosine annealing):

| Metric                         | Value                          |
|--------------------------------|--------------------------------|
| Price RMSE (per unit strike)   | 9.4 × 10⁻⁴ (≈ 0.09 per 100 of notional) |
| Price mean absolute error      | 6.4 × 10⁻⁴                     |
| Delta mean absolute error      | 1.1 × 10⁻²                     |
| Vega mean absolute error       | 1.1 × 10⁻²                     |

Sample at-the-money quotes (`S = K = 100`, `r = 2%`):

| T    | σ    | Black-Scholes | Network | BS delta | Network delta |
|------|------|---------------|---------|----------|---------------|
| 0.25 | 0.20 | 4.2322        | 4.1440  | 0.5398   | 0.5435        |
| 1.00 | 0.20 | 8.9160        | 8.8835  | 0.5793   | 0.5928        |
| 1.00 | 0.40 | 16.7044       | 16.7443 | 0.5987   | 0.5941        |

The largest errors sit at short maturities near the money, where the payoff kink makes the price
surface most curved — a known difficulty for smooth function approximators.

## How it works

Every operation creates a new `Tensor` that stores references to its inputs and a closure that
knows how to push the output gradient back to them (simplified):

```python
def __mul__(self, other):
    out = Tensor(self.data * other.data, parents=(self, other))

    def backward():
        self._accumulate(out.grad * other.data)   # ∂(a·b)/∂a = b
        other._accumulate(out.grad * self.data)   # ∂(a·b)/∂b = a

    out._backward = backward
    return out
```

`loss.backward()` orders the graph topologically and runs these closures from the output back to
the leaves. `_accumulate` sums gradients over axes introduced by broadcasting, so each tensor
receives a gradient with exactly its own shape. Leaf gradients accumulate across calls (as in
PyTorch) until `zero_grad()` clears them.

| Module                     | Contents                                                     |
|----------------------------|--------------------------------------------------------------|
| `tensorgrad.tensor`        | `Tensor`, the operator set, `backward()`, `no_grad()`        |
| `tensorgrad.nn`            | `Parameter`, `Module`, layers and activations                |
| `tensorgrad.functional`    | `softmax`, `log_softmax` and the loss functions              |
| `tensorgrad.optim`         | `SGD`, `Adam`, `AdamW`, `CosineAnnealingLR`, `clip_grad_norm`|
| `tensorgrad.data`          | `DataLoader`, `train_test_split`, `StandardScaler`           |
| `tensorgrad.gradcheck`     | Finite-difference gradient verification                      |

## API overview

```python
from tensorgrad import Tensor, no_grad, nn, optim
from tensorgrad import functional as F
from tensorgrad.data import DataLoader, StandardScaler, train_test_split
from tensorgrad.gradcheck import gradcheck

model = nn.Sequential(nn.Linear(4, 64, init="xavier"), nn.Tanh(), nn.Linear(64, 1))
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
schedule = optim.CosineAnnealingLR(opt, total_steps=10_000)

with no_grad():                      # inference without building a graph
    prediction = model(Tensor(features))

gradcheck(lambda w: (w @ w.T).tanh(), [Tensor(np.random.randn(3, 3), requires_grad=True)])
```

## Testing

```bash
pytest -q             # 70+ tests
ruff check . && ruff format --check .
```

The suite verifies every operator against central finite differences, checks numerical
stability at extreme inputs (`logsumexp`, `sigmoid`, BCE with logits at ±1000), validates the
Adam update against a hand-written reference, and trains real models end to end (nonlinear
regression and a three-class spiral classifier) to confirm the stack learns. CI runs on
Python 3.10 – 3.13.

## Project history

Built incrementally, one stage per commit:

1. `Tensor` with broadcasting arithmetic and reverse-mode `backward()`
2. Matmul, reductions, nonlinearities, shape operations and a gradient checker
3. `Module`, `Linear`, `Sequential`, activations, dropout and `no_grad()`
4. Softmax and loss functions
5. SGD, Adam and AdamW with gradient clipping
6. Data loading, splitting, scaling and end-to-end training tests
7. Cosine annealing schedule
8. Black-Scholes neural pricer with autodiff Greeks

## Limitations

tensorgrad is built for clarity rather than speed: it runs on the CPU through NumPy, computes in
`float64`, has no convolution or recurrent layers, and supports first-order gradients only (no
higher-order derivatives such as gamma through a second backward pass).

## Acknowledgements

Inspired by the "Build Deep Learning From Scratch" and "Neural Networks: Zero to Hero" entries
of [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x), and by the design of
[micrograd](https://github.com/karpathy/micrograd) and PyTorch, extended here from scalars to
NumPy tensors with broadcasting.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
