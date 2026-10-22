import numpy as np
import pandas as pd
import pytest

from src.halftime import (
    HT_FT_STATES,
    add_halftime_targets,
    comeback_rates,
    halftime_transition_matrix,
)


def _matches(rows):
    """rows: (ht_home, ht_away, ft_home, ft_away)."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-08-01") + pd.Timedelta(days=3 * i),
                "season": "2020-21", "league": "E0",
                "home_team": "A", "away_team": "B",
                "ht_home_goals": hth, "ht_away_goals": hta,
                "home_goals": fth, "away_goals": fta,
                "result": "H" if fth > fta else ("A" if fta > fth else "D"),
            }
            for i, (hth, hta, fth, fta) in enumerate(rows)
        ]
    )


def test_derived_targets_are_correct():
    df = add_halftime_targets(_matches([(1, 0, 2, 1), (0, 0, 0, 0), (0, 2, 1, 2)]))
    assert df["ht_result"].tolist() == ["H", "D", "A"]
    assert df["btts"].tolist() == [True, False, True]
    assert df["ht_ft"].tolist() == ["H/H", "D/D", "A/A"]
    # Second-half goals = full time minus half time.
    assert df["second_half_goals"].tolist() == [2, 0, 1]


def test_halftime_draws_are_more_common_than_fulltime_draws():
    """A structural fact worth surfacing: 45 minutes produces far more
    level scores than 90 do."""
    df = add_halftime_targets(
        _matches([(0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 2, 1), (0, 0, 0, 0)])
    )
    assert (df["ht_result"] == "D").mean() > (df["result"] == "D").mean()


def test_transition_matrix_rows_are_probability_distributions():
    df = add_halftime_targets(
        _matches([(1, 0, 2, 0), (1, 0, 1, 1), (0, 1, 0, 2), (0, 0, 1, 1)])
    )
    matrix = halftime_transition_matrix(df)
    assert set(matrix.index) <= {"H", "D", "A"}
    for _, row in matrix.iterrows():
        assert row.sum() == pytest.approx(1.0)
    # Leading at half time -> most likely to win from there.
    assert matrix.loc["H", "H"] >= matrix.loc["H", "A"]


def test_comeback_rates_count_only_reversals():
    df = add_halftime_targets(
        _matches([
            (1, 0, 1, 0),   # led, held
            (1, 0, 1, 2),   # led, lost -> away comeback
            (0, 1, 2, 1),   # trailed, won -> home comeback
            (0, 0, 1, 1),   # level throughout
        ])
    )
    rates = comeback_rates(df)
    assert rates["home_led_at_ht"] == 2
    assert rates["home_led_and_lost"] == 1
    assert rates["away_led_at_ht"] == 1
    assert rates["away_led_and_lost"] == 1
    assert rates["comeback_rate"] == pytest.approx(2 / 3)


def test_ht_ft_states_cover_every_combination():
    df = add_halftime_targets(_matches([(1, 0, 2, 1), (0, 1, 1, 1)]))
    assert set(df["ht_ft"]) <= set(HT_FT_STATES)
    assert len(HT_FT_STATES) == 9
