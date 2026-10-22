import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad import functional as F
from tensorgrad.gradcheck import gradcheck

rng = np.random.default_rng(42)


def test_softmax_rows_sum_to_one():
    x = Tensor(rng.standard_normal((5, 4)) * 50)
    np.testing.assert_allclose(F.softmax(x, axis=1).data.sum(axis=1), np.ones(5))


def test_mse_matches_numpy():
    a, b = rng.standard_normal(10), rng.standard_normal(10)
    assert F.mse_loss(Tensor(a), b).item() == pytest.approx(np.mean((a - b) ** 2))


def test_huber_is_quadratic_then_linear():
    pred = Tensor(np.array([0.5, 3.0]))
    loss = F.huber_loss(pred, np.zeros(2), delta=1.0)
    assert loss.item() == pytest.approx((0.5 * 0.25 + (3.0 - 0.5)) / 2)


def test_cross_entropy_matches_reference():
    logits = rng.standard_normal((6, 3))
    targets = np.array([0, 2, 1, 1, 0, 2])
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    expected = -log_probs[np.arange(6), targets].mean()
    assert F.cross_entropy(Tensor(logits), targets).item() == pytest.approx(expected)


def test_cross_entropy_rejects_bad_shapes():
    with pytest.raises(ValueError):
        F.cross_entropy(Tensor(np.zeros((3, 2))), [0, 1])


def test_bce_with_logits_matches_reference_and_is_stable():
    z = np.array([-800.0, -2.0, 0.0, 3.0, 800.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    p = 1 / (1 + np.exp(-z[1:4]))
    middle = -(y[1:4] * np.log(p) + (1 - y[1:4]) * np.log(1 - p))
    expected = (0.0 + middle.sum() + 0.0) / 5
    assert F.binary_cross_entropy_with_logits(Tensor(z), y).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "loss",
    [
        lambda x: F.mse_loss(x, np.ones((4, 3))),
        lambda x: F.huber_loss(x * 3.0, np.zeros((4, 3))),
        lambda x: F.cross_entropy(x, [0, 1, 2, 1]),
        lambda x: F.binary_cross_entropy_with_logits(x, np.eye(4, 3)),
        lambda x: F.softmax(x, axis=1) * Tensor(np.arange(12.0).reshape(4, 3)),
    ],
)
def test_loss_gradients(loss):
    assert gradcheck(loss, [Tensor(rng.standard_normal((4, 3)), requires_grad=True)])
