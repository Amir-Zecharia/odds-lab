import numpy as np
import pandas as pd
import pytest

from src import config
from src.probabilities import (
    add_market_features,
    demargin_power,
    demargin_proportional,
    demargin_shin,
    implied,
    overround,
)

ODDS = np.array(
    [
        [2.0, 3.4, 4.0],     # typical match, ~9% margin
        [1.2, 6.5, 15.0],    # heavy favorite
        [2.9, 3.1, 2.75],    # tight three-way
    ]
)


@pytest.mark.parametrize("fn", [demargin_proportional, demargin_power, demargin_shin])
def test_demargined_probabilities_sum_to_one(fn):
    probs = fn(ODDS)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert (probs > 0).all() and (probs < 1).all()


def test_overround_positive_for_real_odds():
    assert (overround(ODDS) > 0).all()


def test_margin_free_odds_pass_through_unchanged():
    fair = np.array([[2.0, 4.0, 4.0]])  # implied sum exactly 1
    for fn in (demargin_proportional, demargin_power, demargin_shin):
        np.testing.assert_allclose(fn(fair), implied(fair), atol=1e-6)


def test_shin_shrinks_longshots_more_than_proportional():
    """Shin's whole point: the margin sits disproportionately on longshots,
    so relative to proportional it must cut the longshot probability and
    boost the favorite."""
    odds = np.array([[1.3, 5.5, 11.0]])
    prop = demargin_proportional(odds)[0]
    shin = demargin_shin(odds)[0]
    assert shin[0] > prop[0]  # favorite up
    assert shin[2] < prop[2]  # longshot down


def test_shin_handles_negative_overround():
    """Best-price rows can have implied sums below 1 (an arb); Shin must
    fall back to normalization instead of crashing."""
    arb = np.array([[2.2, 3.9, 4.4]])
    assert overround(arb) < 0
    probs = demargin_shin(arb)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-9)


def _toy_frame():
    return pd.DataFrame(
        {
            "B365H": [2.0, 1.5], "B365D": [3.4, 4.2], "B365A": [4.0, 7.0],
            "PSH": [2.05, 1.52], "PSD": [3.45, 4.3], "PSA": [4.1, 7.2],
            "result": ["H", "A"],
        }
    )


def test_market_features_consensus_and_dispersion():
    out = add_market_features(_toy_frame())
    cons = out[[f"p_cons_{o}" for o in config.OUTCOMES]]
    np.testing.assert_allclose(cons.sum(axis=1), 1.0, atol=1e-9)
    assert (out["disp_std"] >= 0).all()
    assert "sharp_soft_gap" in out.columns


def test_market_features_preserve_nan_rows():
    df = _toy_frame()
    df.loc[1, ["PSH", "PSD", "PSA"]] = np.nan
    out = add_market_features(df)
    assert np.isnan(out.loc[1, "p_PS_H"])
    # Consensus still exists from the remaining book.
    assert not np.isnan(out.loc[1, "p_cons_H"])
