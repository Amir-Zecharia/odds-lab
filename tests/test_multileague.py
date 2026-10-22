import pandas as pd
import pytest

from src import config
from src.features import add_features
from src.ingest import validate


def _league_rows(league: str, season: str, n: int, team_prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": league,
            "season": season,
            "date": pd.date_range(f"{season[:4]}-08-01", periods=n, freq="D"),
            "home_team": [f"{team_prefix}{i % 4}" for i in range(n)],
            "away_team": [f"{team_prefix}{(i + 1) % 4}" for i in range(n)],
            "home_goals": 1,
            "away_goals": 0,
            "result": "H",
        }
    )


def test_league_options_always_include_the_pool():
    opts = config.league_options()
    assert opts[-1] == config.ALL_LEAGUES
    assert set(opts[:-1]) == set(config.LEAGUES)
    assert config.league_display_name(config.ALL_LEAGUES) == "All leagues"
    # Any league added to the registry joins the selector automatically.
    for code in config.LEAGUES:
        assert config.league_display_name(code) == config.LEAGUES[code]["name"]


def test_validate_uses_per_league_expected_counts():
    # A full EPL season (380) must not be flagged against the E1 count (552),
    # and 400 EPL rows must be flagged as too many.
    ok = pd.concat(
        [
            _league_rows("E0", "2023-24", 380, "epl"),
            _league_rows("E1", "2023-24", 552, "ch"),
        ]
    )
    over = pd.concat([ok, _league_rows("E0", "2020-21", 400, "epl")])

    # Only judge the leagues present in the fixture — validate also flags
    # other registry leagues as missing, which is correct but not under test.
    ok_problems = [
        p for p in validate(ok)
        if "2023-24" in p and (" E0 " in p or " E1 " in p)
    ]
    assert ok_problems == []
    assert any("E0 2020-21 has 400" in p for p in validate(over))


def test_validate_requires_recent_seasons_per_league():
    # E1 missing a required recent season entirely -> loud failure.
    df = pd.concat(
        [
            _league_rows("E0", s, 380, "epl")
            for s in config.REQUIRED_RECENT_SEASONS
        ]
    )
    problems = validate(df)
    assert any(p.startswith("required recent season E1") for p in problems)


def test_elo_is_isolated_between_leagues():
    """A team's rating in one league must not leak from another league's
    matches — the per-league feature computation in run_analyses depends
    on callers splitting by league, which this guards."""
    epl = _league_rows("E0", "2023-24", 10, "shared")
    both = pd.concat(
        [epl, _league_rows("E1", "2023-24", 10, "shared")], ignore_index=True
    )

    solo = add_features(epl.copy())
    per_league = pd.concat(
        [add_features(g.copy()) for _, g in both.groupby("league")],
        ignore_index=True,
    )
    merged = per_league[per_league["league"] == "E0"].reset_index(drop=True)
    pd.testing.assert_series_equal(
        solo["elo_home"].reset_index(drop=True), merged["elo_home"]
    )
