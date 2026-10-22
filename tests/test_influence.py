import numpy as np
import pandas as pd
import pytest

from src.influence import (
    convergence_regression,
    follow_the_sharp,
    leadership_matrix,
)


def _panel(n=4000, seed=0, follow_strength=0.8, herd=0.0):
    """Synthetic two-snapshot panel where PS knows the truth early and
    B365/WH close by moving toward PS's early position."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 0.8, n)  # logit home prob

    def prob(x):
        return 1 / (1 + np.exp(-x))

    ps_early = truth + rng.normal(0, 0.05, n)
    b365_early = truth + rng.normal(0, 0.25, n)
    wh_early = truth + rng.normal(0, 0.25, n)
    cons = (ps_early + b365_early + wh_early) / 3

    ps_close = ps_early + rng.normal(0, 0.03, n)
    b365_close = (
        b365_early
        + follow_strength * (ps_early - b365_early)
        + herd * (cons - b365_early)
        + rng.normal(0, 0.03, n)
    )
    wh_close = (
        wh_early
        + follow_strength * (ps_early - wh_early)
        + rng.normal(0, 0.03, n)
    )

    return pd.DataFrame(
        {
            "league": "E0",
            "season": "2021-22",
            "p_cons_H": prob(cons),
            "p_PS_H": prob(ps_early),
            "p_PSC_H": prob(ps_close),
            "p_B365_H": prob(b365_early),
            "p_B365C_H": prob(b365_close),
            "p_WH_H": prob(wh_early),
            "p_WHC_H": prob(wh_close),
        }
    )


def test_leadership_matrix_finds_the_leader():
    matrix = leadership_matrix(_panel())
    ps_leads_b365 = matrix.query("leader=='PS' and follower=='B365'").iloc[0]
    b365_leads_ps = matrix.query("leader=='B365' and follower=='PS'").iloc[0]
    assert ps_leads_b365["z"] > 5
    assert ps_leads_b365["coef"] > 0.5
    assert abs(b365_leads_ps["z"]) < abs(ps_leads_b365["z"])


def test_leadership_silent_when_books_are_independent():
    matrix = leadership_matrix(_panel(follow_strength=0.0, seed=2))
    ps_to_b365 = matrix.query("leader=='PS' and follower=='B365'").iloc[0]
    # B's close = B's early + noise: A's early adds little beyond shared truth.
    assert ps_to_b365["coef"] < 0.3


def test_convergence_recovers_herding_coefficient():
    reg = convergence_regression(_panel(follow_strength=0.0, herd=0.9, seed=3), "B365")
    assert reg["herding_coef"] == pytest.approx(0.9, abs=0.15)


def test_follow_the_sharp_detects_following():
    out = follow_the_sharp(_panel(seed=4), "B365")
    assert out["gap_coef"] == pytest.approx(0.8, abs=0.1)
    assert out["closes_nearer_sharp_early"] > 0.7


def test_follow_the_sharp_null_case():
    out = follow_the_sharp(_panel(follow_strength=0.0, seed=5), "B365")
    assert out["closes_nearer_sharp_early"] < 0.55
