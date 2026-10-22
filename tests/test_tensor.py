import numpy as np
import pytest

from tensorgrad import Tensor


def test_scalar_chain_rule():
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(-2.0, requires_grad=True)
    z = x * y + x**2 - y / x  # dz/dx = y + 2x + y/x^2 ; dz/dy = x - 1/x
    z.backward()
    assert x.grad == pytest.approx(-2.0 + 6.0 + (-2.0) / 9.0)
    assert y.grad == pytest.approx(3.0 - 1.0 / 3.0)


def test_gradient_accumulates_when_a_tensor_is_reused():
    x = Tensor(2.0, requires_grad=True)
    (x * x * x).backward()  # d/dx x^3 = 3x^2
    assert x.grad == pytest.approx(12.0)


def test_broadcasting_reduces_gradient_to_operand_shape():
    a = Tensor(np.ones((4, 3)), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    c = Tensor(np.ones((4, 1)), requires_grad=True)
    out = a * b + c
    out.backward(np.ones((4, 3)))
    assert a.grad.shape == (4, 3)
    np.testing.assert_allclose(a.grad, np.tile([1.0, 2.0, 3.0], (4, 1)))
    np.testing.assert_allclose(b.grad, [4.0, 4.0, 4.0])
    np.testing.assert_allclose(c.grad, np.full((4, 1), 3.0))


def test_reflected_operators():
    x = Tensor(4.0, requires_grad=True)
    y = 1.0 - 2.0 * x + 8.0 / x
    y.backward()
    assert y.item() == pytest.approx(1.0 - 8.0 + 2.0)
    assert x.grad == pytest.approx(-2.0 - 8.0 / 16.0)


def test_leaf_gradients_accumulate_across_backward_calls():
    x = Tensor(1.5, requires_grad=True)
    for _ in range(3):
        (x * 2.0).backward()
    assert x.grad == pytest.approx(6.0)


def test_constants_do_not_receive_gradients():
    x = Tensor(2.0, requires_grad=True)
    k = Tensor(5.0)
    (x * k).backward()
    assert k.grad is None


def test_backward_requires_scalar_or_explicit_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == pytest.approx(1.0)
