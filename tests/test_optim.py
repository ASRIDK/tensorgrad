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


def test_cosine_annealing_schedule():
    p = nn.Parameter(np.zeros(1))
    opt = optim.SGD([p], lr=1.0)
    schedule = optim.CosineAnnealingLR(opt, total_steps=4, min_lr=0.0)
    lrs = []
    for _ in range(6):
        schedule.step()
        lrs.append(opt.lr)
    np.testing.assert_allclose(lrs, [0.8535534, 0.5, 0.1464466, 0.0, 0.0, 0.0], atol=1e-6)


def test_rmsprop_matches_reference_update():
    p = nn.Parameter(np.array([1.0, -2.0]))
    opt = optim.RMSprop([p], lr=0.1, alpha=0.9, eps=1e-8)
    p.grad = np.array([0.5, -1.0])
    opt.step()
    sq = 0.1 * p.grad**2
    expected = np.array([1.0, -2.0]) - 0.1 * p.grad / (np.sqrt(sq) + 1e-8)
    np.testing.assert_allclose(p.data, expected)


def test_rmsprop_minimizes_a_quadratic():
    p = nn.Parameter(np.array([5.0, -3.0]))
    opt = optim.RMSprop([p], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        ((p - 1.0) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(p.data, [1.0, 1.0], atol=0.1)


def test_rmsprop_validates_alpha_and_skips_missing_grads():
    with pytest.raises(ValueError):
        optim.RMSprop([nn.Parameter(np.zeros(1))], alpha=1.0)
    p = nn.Parameter(np.array([1.0]))
    optim.RMSprop([p]).step()  # grad is None: parameter untouched
    assert p.data[0] == 1.0


def test_step_lr_schedule():
    opt = optim.SGD([nn.Parameter(np.zeros(1))], lr=1.0)
    sched = optim.StepLR(opt, step_size=2, gamma=0.5)
    lrs = []
    for _ in range(5):
        sched.step()
        lrs.append(opt.lr)
    assert lrs == [1.0, 0.5, 0.5, 0.25, 0.25]


def test_step_lr_validation():
    opt = optim.SGD([nn.Parameter(np.zeros(1))])
    with pytest.raises(ValueError):
        optim.StepLR(opt, step_size=0)
    with pytest.raises(ValueError):
        optim.StepLR(opt, step_size=1, gamma=0.0)


def _param():
    return nn.Parameter(np.zeros(2))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: optim.SGD([_param()], momentum=-0.1),
        lambda: optim.SGD([_param()], weight_decay=-1.0),
        lambda: optim.Adam([_param()], betas=(1.0, 0.999)),
        lambda: optim.Adam([_param()], betas=(0.9, -0.1)),
        lambda: optim.Adam([_param()], eps=-1e-8),
        lambda: optim.AdamW([_param()], weight_decay=-0.01),
        lambda: optim.RMSprop([_param()], eps=-1.0),
        lambda: optim.RMSprop([_param()], weight_decay=-1.0),
    ],
)
def test_optimizers_reject_invalid_hyperparameters(factory):
    with pytest.raises(ValueError):
        factory()


def test_clip_grad_norm_rejects_non_positive_max_norm():
    p = _param()
    p.grad = np.ones(2)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            optim.clip_grad_norm([p], max_norm=bad)
    np.testing.assert_array_equal(p.grad, np.ones(2))  # untouched


def test_clip_grad_norm_leaves_small_gradients_unchanged():
    p = _param()
    p.grad = np.array([0.3, 0.4])
    assert optim.clip_grad_norm([p], max_norm=1.0) == pytest.approx(0.5)
    np.testing.assert_allclose(p.grad, [0.3, 0.4])
