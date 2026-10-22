import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from src.mismatch import (
    benjamini_hochberg,
    discover_and_confirm,
    scan_segments,
    stacked_universe,
)


def _matches(n=400, seed=0, league="E0", draw_bias=0.0, seasons=("2015-16",)):
    """Synthetic matches whose outcomes follow the stated probabilities,
    except an optional engineered draw mispricing."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet([5, 3, 4], size=n)
    true = probs.copy()
    true[:, 1] = np.clip(true[:, 1] + draw_bias, 0.01, 0.97)
    true = true / true.sum(axis=1, keepdims=True)
    results = [rng.choice(["H", "D", "A"], p=p) for p in true]
    return pd.DataFrame(
        {
            "league": league,
            "season": rng.choice(list(seasons), size=n),
            "round": rng.integers(1, 39, size=n),
            "result": results,
            "p_cons_H": probs[:, 0],
            "p_cons_D": probs[:, 1],
            "p_cons_A": probs[:, 2],
        }
    )


def test_stacked_universe_shape_and_dimensions():
    long = stacked_universe(_matches(100))
    assert len(long) == 300  # three outcome rows per match
    assert {"era", "band", "phase", "outcome"} <= set(long.columns)
    assert long["hit"].isin([0, 1]).all()
    # Exactly one hit per match.
    assert long["hit"].sum() == 100


def test_scan_finds_engineered_draw_pocket():
    df = _matches(n=6000, draw_bias=0.10, seed=3)
    long = stacked_universe(df)
    table = scan_segments(long, ["outcome"], min_n=100)
    draws = table[table["outcome"] == "draw"].iloc[0]
    assert draws["edge"] > 0.05
    assert draws["fdr_significant"]


def test_scan_is_quiet_on_calibrated_data():
    long = stacked_universe(_matches(n=6000, seed=4))
    table = scan_segments(long, ["outcome", "band"], min_n=150)
    # Calibrated input: at FDR 10% we expect (almost) no flags.
    assert table["fdr_significant"].mean() <= 0.15


def test_benjamini_hochberg_matches_scipy():
    rng = np.random.default_rng(7)
    p = np.concatenate([rng.uniform(0, 1, 40), [1e-6, 1e-5, 2e-4]])
    ours = benjamini_hochberg(p, 0.10)
    scipy_flags = sp_stats.false_discovery_control(p) <= 0.10
    np.testing.assert_array_equal(ours, scipy_flags)


def test_discover_and_confirm_replicates_real_pocket():
    early = _matches(n=8000, draw_bias=0.09, seed=5, seasons=("2014-15", "2016-17"))
    late = _matches(n=8000, draw_bias=0.09, seed=6, seasons=("2019-20", "2021-22"))
    out = discover_and_confirm(pd.concat([stacked_universe(early),
                                          stacked_universe(late)]))
    segments = [p["segment"] for p in out["confirmed_pockets"]]
    assert any("draw" in str(s.values()) for s in segments)


def test_discover_and_confirm_rejects_luck():
    # Pocket exists only in the discovery era -> must NOT be confirmed.
    early = _matches(n=8000, draw_bias=0.09, seed=8, seasons=("2014-15",))
    late = _matches(n=8000, draw_bias=0.0, seed=9, seasons=("2020-21",))
    out = discover_and_confirm(pd.concat([stacked_universe(early),
                                          stacked_universe(late)]))
    assert out["candidates_after_fdr"] > 0  # it was "found" in discovery
    draw_confirmed = [
        p for p in out["confirmed_pockets"]
        if "draw" in str(p["segment"].values())
    ]
    assert draw_confirmed == []
