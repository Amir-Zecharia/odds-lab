import numpy as np
import pandas as pd
import pytest

from src.season_sim import (
    league_table,
    simulate_season,
    summarize_simulations,
)


def _played(rows):
    """rows: (home, away, home_goals, away_goals)."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=3 * i),
                "season": "2025-26", "league": "E0",
                "home_team": h, "away_team": a,
                "home_goals": hg, "away_goals": ag,
                "result": "H" if hg > ag else ("A" if ag > hg else "D"),
            }
            for i, (h, a, hg, ag) in enumerate(rows)
        ]
    )


def test_league_table_awards_points_correctly():
    table = league_table(_played([("A", "B", 3, 0), ("B", "C", 1, 1)]))
    assert table.loc["A", "points"] == 3
    assert table.loc["B", "points"] == 1   # one loss, one draw
    assert table.loc["C", "points"] == 1
    assert table.loc["A", "goal_difference"] == 3
    assert table.loc["A", "played"] == 1


def test_league_table_ranks_by_points_then_goal_difference():
    table = league_table(
        _played([("A", "B", 1, 0), ("C", "D", 5, 0), ("D", "C", 0, 0),
                 ("B", "A", 0, 0)])
    )
    # C: 4 pts (+5), A: 4 pts (+1) -> C ranks first on goal difference.
    assert table.index[0] == "C"
    assert table["rank"].tolist() == sorted(table["rank"].tolist())


def test_simulation_completes_every_remaining_fixture():
    played = _played([("A", "B", 2, 0)])
    remaining = pd.DataFrame(
        [{"home_team": "B", "away_team": "A"}, {"home_team": "A", "away_team": "B"}]
    )
    # Deterministic: home always wins.
    probs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tables = simulate_season(played, remaining, probs, n_sims=5, seed=1)

    assert len(tables) == 5
    for table in tables:
        assert table["played"].sum() == 6  # 1 played + 2 remaining, both teams


def test_simulation_respects_the_supplied_probabilities():
    played = _played([("A", "B", 0, 0)])
    remaining = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    # Away always wins -> B must finish above A every single time.
    probs = np.array([[0.0, 0.0, 1.0]])
    tables = simulate_season(played, remaining, probs, n_sims=20, seed=2)
    assert all(t.index[0] == "B" for t in tables)


def test_summary_reports_title_and_relegation_probabilities():
    played = _played([("A", "B", 1, 0)])
    remaining = pd.DataFrame([{"home_team": "A", "away_team": "B"}])
    probs = np.array([[1.0, 0.0, 0.0]])
    tables = simulate_season(played, remaining, probs, n_sims=50, seed=3)
    summary = summarize_simulations(tables, relegation_places=1)

    assert summary.loc["A", "title"] == pytest.approx(1.0)
    assert summary.loc["B", "relegation"] == pytest.approx(1.0)
    assert summary["expected_points"].loc["A"] == pytest.approx(6.0)
    # Position distribution is a probability distribution per team.
    assert summary["title"].sum() == pytest.approx(1.0)
