import numpy as np
import pandas as pd
import pytest

from src.forecast import MatchForecast, forecast_match


def _history(n_rounds=40, seed=4):
    """A small league with a clear strength order and stable count stats."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    attack = {"A": 1.9, "B": 1.4, "C": 1.1, "D": 0.8}
    rows = []
    date = pd.Timestamp("2020-08-01")
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg = rng.poisson(attack[h] * 1.15)
                ag = rng.poisson(attack[a])
                ht_h = rng.binomial(hg, 0.45)
                ht_a = rng.binomial(ag, 0.45)
                rows.append(
                    {
                        "date": date, "season": "2020-21", "league": "E0",
                        "home_team": h, "away_team": a,
                        "home_goals": hg, "away_goals": ag,
                        "ht_home_goals": ht_h, "ht_away_goals": ht_a,
                        "home_corners": rng.poisson(5), "away_corners": rng.poisson(4),
                        "home_yellows": rng.poisson(1.5), "away_yellows": rng.poisson(1.7),
                        "home_reds": 0, "away_reds": 0,
                        "referee": "Ref A",
                        "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                    }
                )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_forecast_produces_a_valid_outcome_distribution():
    f = forecast_match(_history(), "A", "D")
    probs = [f.p_home, f.p_draw, f.p_away]
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < p < 1 for p in probs)
    # Strongest home side vs weakest visitor.
    assert f.p_home > f.p_away


def test_forecast_reverses_when_the_fixture_reverses():
    strong_home = forecast_match(_history(), "A", "D")
    weak_home = forecast_match(_history(), "D", "A")
    assert strong_home.p_home > weak_home.p_home


def test_forecast_includes_count_markets():
    f = forecast_match(_history(), "A", "B")
    assert f.expected_corners > 0
    assert f.expected_cards > 0
    assert 0.0 <= f.p_over_corners_line <= 1.0
    assert 0.0 <= f.p_over_25_goals <= 1.0


def test_forecast_includes_scorelines_and_half_time():
    f = forecast_match(_history(), "A", "B")
    assert len(f.top_scorelines) >= 3
    total = sum(s["prob"] for s in f.top_scorelines)
    assert 0 < total <= 1.0
    ht = f.half_time
    assert sum(ht.values()) == pytest.approx(1.0, abs=1e-6)


def test_referee_shifts_the_card_forecast():
    history = _history()
    strict = history.copy()
    # A second official who books far more, with enough matches to register.
    strict.loc[strict.index[:200], "referee"] = "Strict Ref"
    strict.loc[strict.index[:200], "home_yellows"] = 5
    strict.loc[strict.index[:200], "away_yellows"] = 5

    calm = forecast_match(strict, "A", "B", referee="Ref A")
    harsh = forecast_match(strict, "A", "B", referee="Strict Ref")
    assert harsh.expected_cards > calm.expected_cards


def test_forecast_is_a_dataclass_with_a_summary():
    f = forecast_match(_history(), "A", "B")
    assert isinstance(f, MatchForecast)
    summary = f.as_dict()
    assert {"p_home", "p_draw", "p_away", "expected_corners"} <= set(summary)
