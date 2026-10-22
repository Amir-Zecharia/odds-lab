import numpy as np
import pandas as pd
import pytest

from src.features import add_features


def _mini_league(results):
    """Alternating fixtures between two teams, one match a week."""
    n = len(results)
    goals = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}
    return pd.DataFrame(
        {
            "season": "2020-21",
            "date": pd.date_range("2020-09-01", periods=n, freq="7D"),
            "home_team": ["Alpha", "Beta"] * (n // 2) + ["Alpha"] * (n % 2),
            "away_team": ["Beta", "Alpha"] * (n // 2) + ["Beta"] * (n % 2),
            "result": results,
            "home_goals": [goals[r][0] for r in results],
            "away_goals": [goals[r][1] for r in results],
        }
    )


def test_elo_starts_equal_and_rewards_winners():
    df = add_features(_mini_league(["H", "A", "H"]))
    assert df.loc[0, "elo_home"] == df.loc[0, "elo_away"] == 1500.0
    # Alpha won at home (match 0) and away (match 1): rating must rise.
    assert df.loc[2, "elo_home"] > 1500.0  # Alpha
    assert df.loc[2, "elo_away"] < 1500.0  # Beta


def test_features_use_only_past_information():
    """Changing a future result must not change features of earlier matches."""
    base = _mini_league(["H", "A", "H", "D", "A", "H"])
    alt = base.copy()
    alt.loc[5, ["result", "home_goals", "away_goals"]] = ["A", 0, 2]

    cols = ["elo_home", "elo_away", "form_home", "rest_home"]
    f_base = add_features(base).loc[:4, cols]
    f_alt = add_features(alt).loc[:4, cols]
    pd.testing.assert_frame_equal(f_base, f_alt)


def test_form_needs_five_matches():
    # Both teams play in every row, so form appears from row 5 on.
    df = add_features(_mini_league(["H"] * 12))
    assert np.isnan(df.loc[4, "form_home"])
    assert not np.isnan(df.loc[5, "form_home"])
    # Home always wins and teams alternate hosting, so entering row 10
    # Alpha's last five results are L,W,L,W,L -> mean 1.2 points.
    assert df.loc[10, "form_home"] == pytest.approx(1.2)


def test_rounds_count_each_teams_matches():
    from src.features import add_rounds

    df = add_rounds(_mini_league(["H", "A", "H", "D"]))
    # Two teams playing each other weekly: match k is round k+1 for both.
    assert df["round"].tolist() == [1, 2, 3, 4]


def test_rounds_average_diverging_counters():
    from src.features import add_rounds

    df = _mini_league(["H", "A", "H"])
    # Insert a match where Alpha hosts a third team, pushing Alpha's
    # counter one ahead of Beta's before their next meeting.
    extra = df.iloc[[0]].assign(away_team="Gamma", date=df["date"].max() + pd.Timedelta(days=1))
    out = add_rounds(pd.concat([df, extra], ignore_index=True))
    # Final Alpha-Gamma match: Alpha's 4th, Gamma's 1st -> round (4+1)/2 -> 3.
    assert out.iloc[-1]["round"] == 3


def test_rest_days_track_schedule():
    df = add_features(_mini_league(["H", "D", "A", "H"]))
    assert np.isnan(df.loc[0, "rest_home"])
    assert df.loc[1, "rest_home"] == 7.0
