import numpy as np
import pytest

from tensorgrad import Tensor, nn


def test_leaky_relu_values_and_grad():
    x = Tensor([-2.0, 0.0, 3.0], requires_grad=True)
    y = x.leaky_relu(0.1)
    np.testing.assert_allclose(y.data, [-0.2, 0.0, 3.0])
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [0.1, 0.1, 1.0])


def test_gelu_known_values():
    y = Tensor([0.0, 1.0, -1.0, 10.0]).gelu()
    np.testing.assert_allclose(y.data, [0.0, 0.8411920, -0.1588080, 10.0], atol=1e-5)


def test_activation_modules():
    x = Tensor([-1.0, 2.0])
    np.testing.assert_allclose(nn.LeakyReLU(0.5)(x).data, [-0.5, 2.0])
    np.testing.assert_allclose(nn.GELU()(x).data, x.gelu().data)
    with pytest.raises(ValueError):
        nn.LeakyReLU(-0.1)
