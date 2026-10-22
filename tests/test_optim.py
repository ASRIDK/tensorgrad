import numpy as np
import pytest

from tensorgrad import nn, optim


def rosenbrock(p):
    x, y = p[0], p[1]
    return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2


def minimize(optimizer_cls, steps, **kwargs):
    p = nn.Parameter(np.array([-1.5, 2.0]))
    opt = optimizer_cls([p], **kwargs)
    for _ in range(steps):
        opt.zero_grad()
        rosenbrock(p).backward()
        opt.step()
    return p.data


def test_sgd_with_momentum_minimizes_a_quadratic():
    p = nn.Parameter(np.array([5.0, -3.0]))
    opt = optim.SGD([p], lr=0.1, momentum=0.9)
    for _ in range(200):
        opt.zero_grad()
        ((p - 1.0) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(p.data, [1.0, 1.0], atol=1e-4)


def test_nesterov_requires_momentum():
    with pytest.raises(ValueError):
        optim.SGD([nn.Parameter(np.zeros(1))], nesterov=True)


@pytest.mark.parametrize("cls", [optim.Adam, optim.AdamW])
def test_adam_variants_solve_rosenbrock(cls):
    kwargs = {"lr": 0.02}
    if cls is optim.AdamW:
        kwargs["weight_decay"] = 0.0
    np.testing.assert_allclose(minimize(cls, 4000, **kwargs), [1.0, 1.0], atol=1e-2)


def test_adam_matches_reference_update():
    p = nn.Parameter(np.array([0.5]))
    opt = optim.Adam([p], lr=0.1)
    m = v = 0.0
    x = 0.5
    for t in range(1, 6):
        opt.zero_grad()
        (p * p).sum().backward()  # grad = 2x
        opt.step()
        g = 2 * x
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x -= 0.1 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        assert p.data[0] == pytest.approx(x)


def test_adamw_decays_weights_without_gradient_signal():
    p = nn.Parameter(np.array([1.0]))
    opt = optim.AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = np.zeros(1)
    opt.step()
    assert p.data[0] == pytest.approx(1.0 - 0.1 * 0.5)


def test_clip_grad_norm():
    a, b = nn.Parameter(np.zeros(2)), nn.Parameter(np.zeros(1))
    a.grad, b.grad = np.array([3.0, 0.0]), np.array([4.0])
    norm = optim.clip_grad_norm([a, b], max_norm=1.0)
    assert norm == pytest.approx(5.0)
    total = np.sqrt((a.grad**2).sum() + (b.grad**2).sum())
    assert total == pytest.approx(1.0)


def test_optimizer_rejects_empty_parameters():
    with pytest.raises(ValueError):
        optim.SGD([], lr=0.1)
