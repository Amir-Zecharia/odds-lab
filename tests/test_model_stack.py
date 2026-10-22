import numpy as np
import pandas as pd
import pytest

from src.model_stack import MIN_TRAIN_SEASONS, race


def _stack_frame(n=6000, seed=11, informative="pi_diff"):
    """Synthetic universe with pre-built level-0 columns where exactly one
    odds-free source carries signal; the market knows the truth best."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.0, n)  # latent home-strength edge

    def noisy(scale):
        return truth + rng.normal(0, scale, n)

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-08-01", periods=n, freq="6h"),
            "season": [f"20{10 + (i * 9) // n}-x" for i in range(n)],
            "dc_logit_H": rng.normal(0, 1, n),
            "elo_diff": rng.normal(0, 1, n),
            "pi_diff": rng.normal(0, 1, n),
            "massey_diff": rng.normal(0, 1, n),
            "colley_diff": rng.normal(0, 1, n),
        }
    )
    frame[informative] = noisy(0.6)

    p_home = 1 / (1 + np.exp(-(0.8 * truth - 0.2)))
    p_draw = np.full(n, 0.26)
    p_home = np.clip(p_home * (1 - p_draw), 0.02, 0.93)
    p_away = 1 - p_home - p_draw
    results = [
        str(np.random.default_rng(seed + i).choice(["H", "D", "A"], p=[h, d, a]))
        for i, (h, d, a) in enumerate(zip(p_home, p_draw, p_away))
    ]
    frame["result"] = results
    # The "market" sees the truth with the least noise.
    mh = 1 / (1 + np.exp(-(0.8 * noisy(0.15) - 0.2))) * (1 - p_draw)
    mh = np.clip(mh, 0.02, 0.93)
    frame["p_cons_H"] = mh
    frame["p_cons_D"] = p_draw
    frame["p_cons_A"] = 1 - mh - p_draw
    frame["mkt_logit_H"] = np.log(frame["p_cons_H"] / frame["p_cons_A"])
    frame["mkt_logit_D"] = np.log(frame["p_cons_D"] / frame["p_cons_A"])
    return frame


def test_race_prefers_the_informative_source_and_the_stack():
    out = race(_stack_frame())
    ll = {r["model"]: r["log_loss"] for r in out["race"]}
    # The informative source beats a pure-noise source...
    assert ll["Pi-ratings alone"] < ll["Colley alone"] - 0.005
    # ...and the odds-free stack is at least as good as its best member.
    assert ll["STACK (odds-free zoo)"] <= ll["Pi-ratings alone"] + 0.003


def test_stack_with_market_collapses_onto_the_market():
    out = race(_stack_frame())
    weights = pd.DataFrame(out["weights_with_market"])
    market_share = weights[["mkt_logit_H", "mkt_logit_D"]].sum(axis=1)
    assert market_share.mean() > 0.5  # the market dominates the learned mix


def test_race_reports_every_contender_with_cis():
    out = race(_stack_frame(n=4000, seed=3))
    assert len(out["race"]) == 8  # 5 solo + 2 stacks + consensus baseline
    for row in out["race"]:
        assert row["lo"] <= row["diff_vs_market"] <= row["hi"]
    assert out["n_test"] > 0 and out["n_universe"] >= out["n_test"]
