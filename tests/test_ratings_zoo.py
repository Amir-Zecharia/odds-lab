import numpy as np
import pandas as pd
import pytest

from src.ratings_zoo import (
    add_pi_ratings,
    add_solved_ratings,
    colley_solve,
    massey_solve,
)


def _matches(rows, per_day: int = 1):
    frames = []
    for k, (h, a, hg, ag) in enumerate(rows):
        frames.append(
            {
                "date": pd.Timestamp("2020-09-01")
                + pd.Timedelta(days=7 * (k // per_day)),
                "home_team": h, "away_team": a,
                "home_goals": hg, "away_goals": ag,
                "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            }
        )
    return pd.DataFrame(frames)


def test_massey_orders_a_transitive_round_robin():
    # A beats everyone by 2, B beats C by 2: clear order A > B > C.
    pool = _matches([("A", "B", 2, 0), ("A", "C", 3, 1), ("B", "C", 2, 0)])
    r = massey_solve(pool)
    assert r["A"] > r["B"] > r["C"]
    assert sum(r.values()) == pytest.approx(0.0, abs=1e-9)


def test_massey_ignores_nothing_colley_ignores_margins():
    # Same fixtures, but A's wins are blowouts: Massey moves, Colley doesn't.
    close = _matches([("A", "B", 1, 0), ("A", "C", 1, 0), ("B", "C", 1, 0)])
    blowout = _matches([("A", "B", 6, 0), ("A", "C", 6, 0), ("B", "C", 1, 0)])
    assert massey_solve(blowout)["A"] > massey_solve(close)["A"]
    assert colley_solve(blowout)["A"] == pytest.approx(colley_solve(close)["A"])


def test_colley_draws_count_half():
    pool = _matches([("A", "B", 1, 1)])
    r = colley_solve(pool)
    assert r["A"] == pytest.approx(r["B"])


def test_pi_ratings_are_strictly_pre_match():
    base = _matches([("A", "B", 2, 0), ("B", "A", 0, 2), ("A", "B", 3, 0)])
    alt = base.copy()
    alt.loc[2, ["home_goals", "away_goals", "result"]] = [0, 5, "A"]
    f_base = add_pi_ratings(base)["pi_diff"]
    f_alt = add_pi_ratings(alt)["pi_diff"]
    # Changing the last match's result must not change any pre-match value.
    pd.testing.assert_series_equal(f_base, f_alt)
    # First-ever prediction is neutral; A's rating rises after winning.
    assert f_base.iloc[0] == 0.0
    assert f_base.iloc[2] > 0.0


def test_pi_ratings_move_toward_observed_goal_difference():
    df = add_pi_ratings(_matches([("A", "B", 4, 0)] * 8))
    assert df["pi_diff"].is_monotonic_increasing
    assert df["pi_diff"].iloc[-1] > 0.5


def test_solved_ratings_need_a_pool_and_avoid_leakage():
    rows = [("A", "B", 2, 0), ("C", "D", 1, 1), ("A", "C", 1, 0), ("B", "D", 0, 0)] * 60
    df = _matches(rows, per_day=4)
    out = add_solved_ratings(df)
    # Early matches: pool below threshold -> NaN.
    assert np.isnan(out["massey_diff"].iloc[0])
    # Once ratings exist, A (wins everything) rates above B.
    late = out.dropna(subset=["massey_diff"])
    assert not late.empty
    ab = late[(late["home_team"] == "A") & (late["away_team"] == "B")]
    assert (ab["massey_diff"] > 0).all()
    assert (ab["colley_diff"] > 0).all()
