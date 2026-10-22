import numpy as np
import pandas as pd
import pytest

from src.match_stats import (
    CountModel,
    evaluate_lines,
    referee_multipliers,
    rolling_forecast,
)


def _season(rows, season="2020-21", start="2020-08-01"):
    """rows: (home, away, home_count, away_count, referee)."""
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(start) + pd.Timedelta(days=3 * i),
                "season": season,
                "league": "E0",
                "home_team": h,
                "away_team": a,
                "home_corners": hc,
                "away_corners": ac,
                "referee": ref,
            }
            for i, (h, a, hc, ac, ref) in enumerate(rows)
        ]
    )


def _balanced(n=12, ref="Ref A"):
    """Every team equal: 5 corners each, every match."""
    teams = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        rows.append((h, a, 5, 5, ref))
    return _season(rows)


def test_equal_teams_get_neutral_strengths():
    model = CountModel(target="corners").fit(_balanced())
    lam_home, lam_away = model.expected("A", "B")
    assert lam_home == pytest.approx(5.0, abs=0.3)
    assert lam_away == pytest.approx(5.0, abs=0.3)


def test_dominant_team_gets_a_higher_expected_count():
    """A team that always wins the corner count must project higher."""
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(24):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        # Team A wins corners heavily whenever it plays.
        hc, ac = (10, 2) if h == "A" else ((2, 10) if a == "A" else (5, 5))
        rows.append((h, a, hc, ac, "Ref A"))
    model = CountModel(target="corners").fit(_season(rows))

    strong_home, weak_away = model.expected("A", "B")
    assert strong_home > weak_away
    assert strong_home > 6.0


def test_prob_over_is_a_probability_and_falls_with_the_line():
    model = CountModel(target="corners").fit(_balanced())
    probs = [model.prob_over("A", "B", line) for line in (6.5, 9.5, 12.5)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] > probs[1] > probs[2]


def test_referee_multiplier_shrinks_small_samples_toward_one():
    """578 referees, most with a handful of matches — an extreme rate from
    two appearances must not be taken at face value."""
    rows = [("A", "B", 5, 5, "Busy Ref") for _ in range(60)]
    rows += [("C", "D", 30, 30, "Rare Ref") for _ in range(2)]
    mult = referee_multipliers(_season(rows), target="corners")

    assert mult["Rare Ref"] == pytest.approx(1.0, abs=0.5)
    assert abs(mult["Busy Ref"] - 1.0) < 0.2


def test_referee_with_many_matches_keeps_a_real_effect():
    rows = [("A", "B", 4, 4, "Calm Ref") for _ in range(80)]
    rows += [("C", "D", 12, 12, "Card Happy") for _ in range(80)]
    mult = referee_multipliers(_season(rows), target="corners")
    assert mult["Card Happy"] > mult["Calm Ref"] * 1.5


def test_referee_multipliers_are_measured_within_league():
    """Italian officials card more than English ones. Pooling leagues makes
    a multiplier measure the league, not the referee — so a strict ref in a
    lenient league must still read as strict."""
    calm_league = _season(
        [("A", "B", 2, 2, "Calm Ref") for _ in range(60)]
    ).assign(league="E0")
    # Same league, but this official runs hot relative to his own league.
    hot_in_calm = _season(
        [("C", "D", 6, 6, "Strict Ref") for _ in range(60)]
    ).assign(league="E0")
    # A different, card-heavy league — should not drag E0 multipliers around.
    busy_league = _season(
        [("E", "F", 12, 12, "Italian Ref") for _ in range(60)]
    ).assign(league="I1")

    df = pd.concat([calm_league, hot_in_calm, busy_league], ignore_index=True)
    mult = referee_multipliers(df, target="corners", by_league=True)

    assert mult["Strict Ref"] > 1.2   # strict for his own league
    assert mult["Calm Ref"] < 0.9
    # The Italian ref is normal *for Italy*, despite the highest raw count.
    assert mult["Italian Ref"] == pytest.approx(1.0, abs=0.15)


def test_evaluate_lines_scores_probability_forecasts():
    """Counts are judged the way the rest of the platform judges anything:
    as probabilities, with log loss and realized-vs-predicted calibration."""
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(120):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        rows.append((h, a, 5, 5, "Ref A"))
    forecast = rolling_forecast(_season(rows), target="corners", min_train=20)
    scored = evaluate_lines(forecast, lines=(8.5, 10.5))

    assert set(scored["line"]) == {8.5, 10.5}
    for _, row in scored.iterrows():
        assert 0.0 <= row["predicted_over_rate"] <= 1.0
        assert 0.0 <= row["actual_over_rate"] <= 1.0
        assert row["log_loss"] > 0
    # Totals are always exactly 10: over 8.5 always, over 10.5 never.
    assert scored.set_index("line").loc[8.5, "actual_over_rate"] == 1.0
    assert scored.set_index("line").loc[10.5, "actual_over_rate"] == 0.0


def test_recency_weighting_tracks_a_drifting_rate():
    """Corner and card rates drift across eras. A model that weights 2003
    equally with last month is systematically stale — the real data shows
    corners over-predicted and cards under-predicted for exactly this
    reason."""
    old = _season(
        [("A", "B", 8, 8, "Ref A") for _ in range(60)], start="2015-08-01"
    )
    recent = _season(
        [("A", "B", 2, 2, "Ref A") for _ in range(60)], start="2024-08-01"
    )
    df = pd.concat([old, recent], ignore_index=True)

    flat = CountModel(target="corners").fit(df)
    decayed = CountModel(target="corners", half_life_days=365).fit(df)

    flat_total = sum(flat.expected("A", "B"))
    decayed_total = sum(decayed.expected("A", "B"))
    # The recent regime is 4 corners a match; decay must move toward it.
    assert decayed_total < flat_total
    assert decayed_total == pytest.approx(4.0, abs=1.5)


def test_referee_multipliers_are_baselined_within_era():
    """Cards per match have risen for decades, so pooling eras makes an
    old lenient-era referee look personally lenient."""
    old_league = _season(
        [("A", "B", 1, 1, "Old Ref") for _ in range(60)], season="2004-05"
    ).assign(league="E0")
    new_league = _season(
        [("C", "D", 5, 5, "New Ref") for _ in range(60)], season="2024-25"
    ).assign(league="E0")
    df = pd.concat([old_league, new_league], ignore_index=True)

    mult = referee_multipliers(df, target="corners", by_league=True, by_season=True)
    # Each is exactly typical for their own era -> both neutral.
    assert mult["Old Ref"] == pytest.approx(1.0, abs=0.15)
    assert mult["New Ref"] == pytest.approx(1.0, abs=0.15)


def test_rolling_forecast_never_uses_the_match_it_predicts():
    """Leakage guard: changing a match's own counts must not change its
    own prediction."""
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(80):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        rows.append((h, a, 5, 5, "Ref A"))
    base = _season(rows)
    tampered = base.copy()
    tampered.loc[len(tampered) - 1, "home_corners"] = 99

    f_base = rolling_forecast(base, target="corners", min_train=20)
    f_tampered = rolling_forecast(tampered, target="corners", min_train=20)
    last = f_base.index[-1]
    assert f_base.loc[last, "pred_home"] == pytest.approx(
        f_tampered.loc[last, "pred_home"]
    )


def test_rolling_forecast_reports_predictions_and_actuals():
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(60):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        rows.append((h, a, 6, 4, "Ref A"))
    out = rolling_forecast(_season(rows), target="corners", min_train=20)

    assert {"pred_home", "pred_away", "pred_total", "actual_total"} <= set(out.columns)
    assert out["pred_total"].notna().all()
    # Balanced synthetic data: the model should land near the true total.
    assert out["pred_total"].mean() == pytest.approx(10.0, abs=1.0)
