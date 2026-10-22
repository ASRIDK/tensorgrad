import numpy as np
import pytest

from tensorgrad import Tensor, nn, no_grad
from tensorgrad.gradcheck import gradcheck


def make_mlp(seed=0):
    rng = np.random.default_rng(seed)
    return nn.Sequential(
        nn.Linear(3, 8, rng=rng),
        nn.ReLU(),
        nn.Linear(8, 8, rng=rng),
        nn.Tanh(),
        nn.Linear(8, 1, rng=rng),
    )


def test_parameters_are_discovered_recursively():
    model = make_mlp()
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "layers.0.weight",
        "layers.0.bias",
        "layers.2.weight",
        "layers.2.bias",
        "layers.4.weight",
        "layers.4.bias",
    ]
    assert model.num_parameters() == 3 * 8 + 8 + 8 * 8 + 8 + 8 + 1


def test_linear_matches_numpy():
    layer = nn.Linear(4, 2, rng=np.random.default_rng(1))
    x = np.random.default_rng(2).standard_normal((5, 4))
    out = layer(Tensor(x))
    np.testing.assert_allclose(out.data, x @ layer.weight.data + layer.bias.data)


def test_mlp_gradients_match_finite_differences():
    model = make_mlp()
    x = Tensor(np.random.default_rng(3).standard_normal((6, 3)))
    params = model.parameters()

    def loss(*_):
        return (model(x) ** 2).mean()

    assert gradcheck(loss, params)


def test_zero_grad_clears_parameter_gradients():
    model = make_mlp()
    model(Tensor(np.ones((2, 3)))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_state_dict_round_trip():
    a, b = make_mlp(seed=0), make_mlp(seed=1)
    x = Tensor(np.ones((2, 3)))
    assert not np.allclose(a(x).data, b(x).data)
    b.load_state_dict(a.state_dict())
    np.testing.assert_allclose(a(x).data, b(x).data)


def test_load_state_dict_rejects_mismatched_keys():
    with pytest.raises(KeyError):
        make_mlp().load_state_dict({"nope": np.zeros(1)})


def test_no_grad_builds_no_graph():
    model = make_mlp()
    with no_grad():
        out = model(Tensor(np.ones((2, 3))))
    assert not out.requires_grad


def test_dropout_is_identity_in_eval_mode_and_unbiased_in_training():
    drop = nn.Dropout(0.3, rng=np.random.default_rng(0))
    x = Tensor(np.ones((1000, 100)))
    assert np.isclose(drop(x).data.mean(), 1.0, atol=0.02)
    assert ((drop(x).data == 0).mean()) == pytest.approx(0.3, abs=0.02)
    drop.eval()
    np.testing.assert_array_equal(drop(x).data, x.data)


def test_train_and_eval_propagate_to_children():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model.eval()
    assert all(not m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())
