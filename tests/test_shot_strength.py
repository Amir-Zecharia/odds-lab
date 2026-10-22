import numpy as np
import pandas as pd
import pytest

from src.shot_strength import (
    conversion_rates,
    fit_shot_model,
    shot_based_goal_probs,
    race_shots_vs_goals,
)


def _league(n_rounds=40, seed=5, transient_finishing=None):
    """Synthetic league where shot volume is the true, persistent strength.

    ``transient_finishing`` gives each team a per-era finishing multiplier
    that is **redrawn between eras** — luck, not skill. That is precisely
    the xG argument: a goals-based model learns the luck as strength and
    carries it into a period where it no longer holds, while a shot-based
    model ignores it.
    """
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    shot_rate = {"A": 8.0, "B": 6.0, "C": 5.0, "D": 3.5}
    rows = []
    date = pd.Timestamp("2015-08-01")
    for r in range(n_rounds):
        era = r // 20
        if transient_finishing is not None:
            luck = {
                t: 1.0 + transient_finishing * rng_era
                for t, rng_era in zip(
                    teams,
                    np.random.default_rng(seed + 100 * era).normal(0, 1, len(teams)),
                )
            }
        else:
            luck = {t: 1.0 for t in teams}
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hs = rng.poisson(shot_rate[h] * 1.15)
                as_ = rng.poisson(shot_rate[a])
                hg = rng.binomial(hs, np.clip(0.12 * luck[h], 0.01, 0.9))
                ag = rng.binomial(as_, np.clip(0.12 * luck[a], 0.01, 0.9))
                rows.append(
                    {
                        "date": date, "season": f"20{15 + era}-x",
                        "league": "E0", "home_team": h, "away_team": a,
                        "home_goals": hg, "away_goals": ag,
                        "home_shots_target": hs, "away_shots_target": as_,
                        "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                    }
                )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_conversion_rate_is_goals_per_shot_on_target():
    df = pd.DataFrame(
        {
            "home_goals": [2, 1], "away_goals": [1, 0],
            "home_shots_target": [8, 4], "away_shots_target": [4, 2],
        }
    )
    rates = conversion_rates(df)
    assert rates["home"] == pytest.approx(3 / 12)
    assert rates["away"] == pytest.approx(1 / 6)


def test_shot_model_ranks_teams_by_shot_volume():
    model = fit_shot_model(_league())
    lam_strong, lam_weak = shot_based_goal_probs(model, "A", "D")
    lam_weak_home, lam_strong_away = shot_based_goal_probs(model, "D", "A")
    assert lam_strong > lam_weak
    assert lam_strong_away > lam_weak_home


def test_shot_model_yields_a_valid_outcome_distribution():
    model = fit_shot_model(_league())
    probs = model.outcome_probs("A", "B")
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert (probs > 0).all()


def test_race_reports_both_models_on_identical_matches():
    out = race_shots_vs_goals(_league(n_rounds=60), min_train_seasons=1)
    assert {"shots_log_loss", "goals_log_loss", "n_test", "diff", "lo", "hi"} <= set(out)
    assert out["n_test"] > 0
    assert out["lo"] <= out["diff"] <= out["hi"]


def test_shots_win_when_finishing_luck_does_not_persist():
    """The whole xG argument: when a team's hot finishing in the training
    era evaporates in the test era, the goals-based model is chasing a
    ghost and the shot-based model is not."""
    noisy = _league(n_rounds=60, seed=9, transient_finishing=0.9)
    out = race_shots_vs_goals(noisy, min_train_seasons=1)
    assert out["shots_log_loss"] < out["goals_log_loss"]
