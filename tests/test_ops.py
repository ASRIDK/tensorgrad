import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad.gradcheck import gradcheck

rng = np.random.default_rng(0)


def param(*shape, positive=False):
    data = rng.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True)


UNARY_CASES = {
    "exp": (lambda x: x.exp(), False),
    "log": (lambda x: x.log(), True),
    "sqrt": (lambda x: x.sqrt(), True),
    "pow": (lambda x: x**3, False),
    "reciprocal": (lambda x: 1.0 / x, True),
    "tanh": (lambda x: x.tanh(), False),
    "sigmoid": (lambda x: x.sigmoid(), False),
    "softplus": (lambda x: x.softplus(), False),
    "relu": (lambda x: (x + 0.1).relu(), False),
    "abs": (lambda x: x.abs(), True),
    "sum_axis0": (lambda x: x.sum(axis=0), False),
    "sum_keepdims": (lambda x: x.sum(axis=1, keepdims=True), False),
    "mean": (lambda x: x.mean(), False),
    "mean_axis": (lambda x: x.mean(axis=1), False),
    "max_axis": (lambda x: x.max(axis=1), False),
    "logsumexp": (lambda x: x.logsumexp(axis=1), False),
    "reshape": (lambda x: x.reshape(4, 3) * Tensor(np.arange(12.0).reshape(4, 3)), False),
    "transpose": (lambda x: x.T * Tensor(np.arange(12.0).reshape(4, 3)), False),
    "getitem": (lambda x: x[np.array([0, 2, 2]), 1:3], False),
}


@pytest.mark.parametrize("name", UNARY_CASES)
def test_unary_gradients(name):
    fn, positive = UNARY_CASES[name]
    assert gradcheck(fn, [param(3, 4, positive=positive)])


def test_matmul_gradient():
    assert gradcheck(lambda a, b: a @ b, [param(5, 3), param(3, 2)])


def test_batched_matmul_gradient():
    assert gradcheck(lambda a, b: a @ b, [param(2, 5, 3), param(3, 4)])


def test_broadcast_binary_gradient():
    assert gradcheck(lambda a, b: (a - b) * a / (b * b + 1.0), [param(4, 3), param(1, 3)])


def test_logsumexp_is_stable_for_large_inputs():
    x = Tensor(np.array([[1000.0, 1000.0], [-1000.0, -1000.0]]), requires_grad=True)
    out = x.logsumexp(axis=1)
    np.testing.assert_allclose(out.data, [1000.0 + np.log(2), -1000.0 + np.log(2)])
    out.sum().backward()
    np.testing.assert_allclose(x.grad, np.full((2, 2), 0.5))


def test_sigmoid_is_stable_for_large_inputs():
    x = Tensor(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(x.sigmoid().data, [0.0, 0.5, 1.0])


def test_max_splits_gradient_between_ties():
    x = Tensor(np.array([[1.0, 3.0, 3.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0.0, 0.5, 0.5]])


def test_matmul_rejects_vectors():
    with pytest.raises(ValueError):
        Tensor(np.ones(3)) @ Tensor(np.ones(3))


def test_gradcheck_detects_wrong_gradients():
    x = param(3)
    broken = Tensor.__mul__

    def wrong(a):
        out = broken(a, a)
        out._backward = lambda: a._accumulate(out.grad)  # should be 2a
        return out

    with pytest.raises(AssertionError):
        gradcheck(wrong, [x])
