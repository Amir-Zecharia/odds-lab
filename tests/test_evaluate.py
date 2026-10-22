import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

from src.evaluate import (
    bootstrap_ci,
    brier,
    expanding_season_splits,
    log_loss,
    paired_bootstrap_diff,
    time_ordered_split,
    unpaired_bootstrap_diff,
)


def _fake_matches(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2015-08-01", periods=n, freq="3D"),
            "season": ["2015-16"] * (n // 2) + ["2016-17"] * (n - n // 2),
            "result": rng.choice(["H", "D", "A"], size=n, p=[0.45, 0.25, 0.30]),
        }
    )


def test_log_loss_matches_sklearn():
    df = _fake_matches()
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([4, 2, 3], size=len(df))
    y = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ours = log_loss(probs, df["result"])
    theirs = sk_log_loss(y, probs, labels=[0, 1, 2])
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_perfect_and_uniform_predictions():
    df = _fake_matches(n=30)
    onehot = np.column_stack([(df["result"] == o).to_numpy(float) for o in "HDA"])
    assert log_loss(onehot, df["result"]) < 1e-9
    assert brier(onehot, df["result"]) < 1e-12
    uniform = np.full((len(df), 3), 1 / 3)
    np.testing.assert_allclose(log_loss(uniform, df["result"]), np.log(3), atol=1e-9)


def test_time_ordered_split_never_leaks():
    df = _fake_matches().sample(frac=1.0, random_state=3)  # shuffle rows
    train_idx, test_idx = time_ordered_split(df)
    assert df.loc[train_idx, "date"].max() <= df.loc[test_idx, "date"].min()


def test_expanding_splits_are_strictly_forward():
    df = _fake_matches()
    splits = expanding_season_splits(df, min_train_seasons=1)
    assert splits, "expected at least one split"
    for train_idx, test_idx in splits:
        assert df.loc[train_idx, "date"].max() < df.loc[test_idx, "date"].min()


def test_bootstrap_ci_brackets_mean():
    values = np.random.default_rng(7).normal(1.0, 0.2, size=500)
    mean, lo, hi = bootstrap_ci(values)
    assert lo < mean < hi
    assert hi - lo < 0.1  # tight for n=500


def test_unpaired_bootstrap_handles_unequal_samples():
    rng = np.random.default_rng(13)
    a = rng.normal(1.05, 0.2, size=700)   # worse (higher loss)
    b = rng.normal(1.00, 0.2, size=300)
    res = unpaired_bootstrap_diff(a, b)
    assert res["lo"] < res["diff"] < res["hi"]
    assert res["diff"] > 0


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(11)
    b = rng.normal(1.0, 0.1, size=400)
    a = b - 0.05  # a is uniformly better (lower loss)
    res = paired_bootstrap_diff(a, b)
    assert res["hi"] < 0
    assert res["p_better"] > 0.99
