import numpy as np
import pandas as pd
import pytest

from src.drift_analysis import (
    build_drift_frame,
    drift_surprise_regression,
    miss_rate_by_bucket,
    stability_over_time,
)


def _frame(n=6):
    """Two-book frame with hand-set drifts; consensus favors home."""
    return pd.DataFrame(
        {
            "season": ["2019-20"] * (n // 2) + ["2020-21"] * (n - n // 2),
            "date": pd.date_range("2019-09-01", periods=n, freq="30D"),
            "result": ["H", "A", "H", "D", "H", "A"][:n],
            "p_PS_H": [0.50, 0.50, 0.60, 0.40, 0.55, 0.45][:n],
            "p_PS_D": [0.28, 0.30, 0.22, 0.32, 0.25, 0.30][:n],
            "p_PS_A": [0.22, 0.20, 0.18, 0.28, 0.20, 0.25][:n],
            # Closing moves home prob by a known amount per match.
            "p_PSC_H": [0.52, 0.42, 0.60, 0.46, 0.56, 0.35][:n],
            "p_PSC_D": [0.27, 0.33, 0.22, 0.29, 0.245, 0.34][:n],
            "p_PSC_A": [0.21, 0.25, 0.18, 0.25, 0.195, 0.31][:n],
            "p_cons_H": [0.51, 0.49, 0.59, 0.41, 0.54, 0.44][:n],
            "p_cons_D": [0.28, 0.30, 0.23, 0.32, 0.26, 0.30][:n],
            "p_cons_A": [0.21, 0.21, 0.18, 0.27, 0.20, 0.26][:n],
        }
    )


def test_drift_is_max_component_move():
    out = build_drift_frame(_frame(), "PS", "PSC")
    # Match 0: moves are (.02, .01, .01) -> 0.02; match 1: (.08, .03, .05) -> 0.08.
    assert out["drift"].iloc[0] == pytest.approx(0.02)
    assert out["drift"].iloc[1] == pytest.approx(0.08)


def test_favorite_miss_uses_consensus_favorite():
    out = build_drift_frame(_frame(), "PS", "PSC")
    # Consensus favorite is H everywhere except match 3 (D? no: 0.41 H is max).
    # Results H,A,H,D,H,A -> misses at 1, 3, 5.
    assert out["fav_missed"].tolist() == [0, 1, 0, 1, 0, 1]


def test_missing_quotes_are_dropped():
    df = _frame()
    df.loc[2, "p_PSC_H"] = np.nan
    out = build_drift_frame(df, "PS", "PSC")
    assert len(out) == 5


def _synthetic(n=3000, seed=9, drift_hurts=True):
    """Matches where big drift genuinely raises the miss probability."""
    rng = np.random.default_rng(seed)
    drift = rng.exponential(0.02, size=n)
    fav_prob = rng.uniform(0.40, 0.75, size=n)
    miss_p = np.clip(1 - fav_prob + (8.0 * drift if drift_hurts else 0.0), 0.02, 0.95)
    frame = pd.DataFrame(
        {
            "season": [f"20{12 + (i * 10) // n}-x" for i in range(n)],
            "drift": drift,
            "fav_prob": fav_prob,
            "fav_missed": rng.binomial(1, miss_p),
            "closing_fav_missed": rng.binomial(1, np.clip(1 - fav_prob, 0.02, 0.95)),
            "cons_log_loss": -np.log(np.clip(1 - miss_p, 1e-6, None)),
        }
    )
    return frame


def test_regression_detects_engineered_drift_effect():
    reg = drift_surprise_regression(_synthetic(drift_hurts=True))
    assert reg["logit_drift_coef"] > 0
    assert reg["logit_drift_p"] < 0.01


def test_regression_finds_nothing_when_drift_is_noise():
    reg = drift_surprise_regression(_synthetic(drift_hurts=False))
    assert reg["logit_drift_p"] > 0.05


def test_buckets_are_monotone_in_drift():
    table = miss_rate_by_bucket(_synthetic(), n_buckets=6)
    assert table["drift_mid"].is_monotonic_increasing
    assert table["n"].sum() == 3000
    assert ((table["miss_lo"] <= table["miss_rate"]) & (table["miss_rate"] <= table["miss_hi"])).all()


def test_stability_blocks_report_gap():
    out = stability_over_time(_synthetic(), block_years=3)
    assert not out.empty
    # Engineered effect: big-move miss rate exceeds the rest in most blocks.
    assert (out["gap"] > 0).mean() > 0.5
