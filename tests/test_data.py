import numpy as np
import pytest

from tensorgrad.data import DataLoader, StandardScaler, train_test_split


def test_dataloader_covers_every_sample_once_per_epoch():
    x = np.arange(10).reshape(10, 1).astype(float)
    y = np.arange(10)
    loader = DataLoader(x, y, batch_size=3, rng=np.random.default_rng(0))
    assert len(loader) == 4
    seen = np.concatenate([batch_y for _, batch_y in loader])
    assert sorted(seen) == list(range(10))
    for batch_x, batch_y in loader:
        np.testing.assert_array_equal(batch_x.data[:, 0], batch_y)


def test_dataloader_drop_last():
    loader = DataLoader(np.zeros((10, 2)), np.zeros(10), batch_size=4, drop_last=True)
    assert len(loader) == 2
    assert [len(y) for _, y in loader] == [4, 4]


def test_train_test_split_keeps_pairs_aligned():
    x = np.arange(100)
    y = x * 2
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, rng=np.random.default_rng(1))
    assert len(x_te) == 25 and len(x_tr) == 75
    np.testing.assert_array_equal(y_tr, x_tr * 2)
    assert set(x_tr) | set(x_te) == set(x)


def test_standard_scaler_round_trip():
    x = np.random.default_rng(2).normal(5.0, 3.0, size=(500, 3))
    x[:, 2] = 7.0  # constant column
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    np.testing.assert_allclose(z[:, :2].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z[:, :2].std(axis=0), 1.0)
    np.testing.assert_allclose(scaler.inverse_transform(z), x)


def test_standard_scaler_requires_fit():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.zeros((2, 2)))
