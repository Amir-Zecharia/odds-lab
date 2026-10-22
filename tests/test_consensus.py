import numpy as np
import pandas as pd
import pytest

from src.consensus import (
    panel_universe,
    performance_weights,
    predict_equal,
    predict_margin,
    predict_sharpest,
    predict_stacked,
    race,
)

PANEL = ["AA", "BB"]


def _panel_league(n=800, seed=4, noise_bb=0.30):
    """Synthetic matches: book AA quotes true probabilities, book BB adds
    noise. Any sensible weighting should prefer AA."""
    rng = np.random.default_rng(seed)
    true = rng.dirichlet([5, 3, 4], size=n)
    results = np.array(
        [rng.choice(["H", "D", "A"], p=p) for p in true]
    )
    bb = true + rng.normal(0, noise_bb, size=true.shape) * true
    bb = np.clip(bb, 0.02, None)
    bb = bb / bb.sum(axis=1, keepdims=True)

    n_seasons = 8
    labels = [f"20{10 + i}-{11 + i}" for i in range(n_seasons)]
    seasons = [labels[(i * n_seasons) // n] for i in range(n)]
    frame = {
        "date": pd.date_range("2010-08-01", periods=n, freq="D"),
        "season": seasons,
        "result": results,
        "margin_AA": 0.03,
        "margin_BB": 0.08,
    }
    for i, o in enumerate("HDA"):
        frame[f"p_AA_{o}"] = true[:, i]
        frame[f"p_BB_{o}"] = bb[:, i]
    return pd.DataFrame(frame)


def test_universe_drops_incomplete_rows():
    df = _panel_league(n=50)
    df.loc[3, "p_AA_H"] = np.nan
    assert len(panel_universe(df, PANEL)) == 49


@pytest.mark.parametrize("fn", [predict_equal, predict_margin, predict_sharpest])
def test_scheme_outputs_are_distributions(fn):
    df = _panel_league(n=60)
    probs = fn(df, PANEL)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs > 0).all()


def test_performance_weights_prefer_accurate_book():
    df = _panel_league()
    w = performance_weights(df, PANEL, temperature=0.01)
    assert w["AA"] > 0.7  # AA quotes the truth, BB is noisy
    assert w["AA"] + w["BB"] == pytest.approx(1.0)


def test_stacking_leans_on_accurate_book():
    df = _panel_league()
    train = df[df["season"] < "2016-17"]
    test = df[df["season"] >= "2016-17"]
    probs, influence, c = predict_stacked(train, test, PANEL)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert influence["AA"] > influence["BB"]


def test_optimal_weights_recover_truth_book():
    from src.consensus import optimal_weights

    df = _panel_league()
    w = optimal_weights(df, PANEL)
    assert w["AA"] > 0.85  # AA quotes true probabilities
    assert w["AA"] + w["BB"] == pytest.approx(1.0)
    assert min(w.values()) >= 0.0


def test_optimal_weights_shrink_toward_uniform():
    from src.consensus import optimal_weights

    df = _panel_league()
    strong = optimal_weights(df, PANEL, shrink=100.0)
    assert strong["AA"] == pytest.approx(0.5, abs=0.05)


def test_race_ranks_truth_over_noise():
    df = _panel_league()
    summary, losses, weights, tuning = race(df, PANEL, min_train_seasons=3)
    # Every scheme is scored on the same matches.
    assert summary["n_matches"].nunique() == 1
    # Weighting toward AA must beat equal weighting on synthetic data
    # where AA is strictly better.
    assert (
        summary.loc["performance weights (auto T)", "log_loss"]
        <= summary.loc["equal weights", "log_loss"] + 1e-9
    )
    assert set(weights.columns) == set(PANEL)
