import numpy as np
import pandas as pd
import pytest

from src.evaluate import log_loss_per_match
from src.ledger import (
    MODEL_VERSION,
    append_predictions,
    empty_ledger,
    load_ledger,
    scoreboard,
    settle,
)

NOW = pd.Timestamp("2026-08-15 12:00", tz="UTC")


def _predictions(rows):
    """rows: (kickoff_offset_hours, home, away, p_home, p_draw, p_away)."""
    return pd.DataFrame(
        [
            {
                "kickoff": NOW + pd.Timedelta(hours=h),
                "home_team": home, "away_team": away,
                "p_home": ph, "p_draw": pd_, "p_away": pa,
                "source": "Pinnacle", "books": 5, "league": "E0",
            }
            for h, home, away, ph, pd_, pa in rows
        ]
    )


def test_recording_appends_new_fixtures():
    preds = _predictions([(24, "Arsenal", "Chelsea", 0.5, 0.27, 0.23)])
    ledger, stats = append_predictions(empty_ledger(), preds, now=NOW)
    assert stats == {"added": 1, "updated": 0, "frozen": 0}
    assert len(ledger) == 1
    assert ledger.iloc[0]["model_version"] == MODEL_VERSION
    assert pd.isna(ledger.iloc[0]["result"])


def test_future_fixture_updates_instead_of_duplicating():
    first = _predictions([(24, "Arsenal", "Chelsea", 0.50, 0.27, 0.23)])
    ledger, _ = append_predictions(empty_ledger(), first, now=NOW)
    # Odds moved before kickoff: the latest pre-match price is the forecast.
    second = _predictions([(24, "Arsenal", "Chelsea", 0.58, 0.24, 0.18)])
    ledger, stats = append_predictions(ledger, second, now=NOW)
    assert stats["updated"] == 1 and stats["added"] == 0
    assert len(ledger) == 1
    assert ledger.iloc[0]["p_home"] == pytest.approx(0.58)


def test_past_kickoff_predictions_are_frozen():
    """The integrity rule the whole exercise rests on: once a match has
    started, its recorded forecast can never be revised."""
    preds = _predictions([(-2, "Arsenal", "Chelsea", 0.50, 0.27, 0.23)])
    ledger, _ = append_predictions(empty_ledger(), preds, now=NOW - pd.Timedelta(hours=5))
    # Now the match has kicked off; a re-run tries to write better numbers.
    tampered = _predictions([(-2, "Arsenal", "Chelsea", 0.99, 0.005, 0.005)])
    ledger, stats = append_predictions(ledger, tampered, now=NOW)
    assert stats["frozen"] == 1 and stats["updated"] == 0
    assert len(ledger) == 1
    assert ledger.iloc[0]["p_home"] == pytest.approx(0.50)  # untouched


def _matches(rows):
    """rows: (date, home, away, result)."""
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "home_team": h, "away_team": a,
             "result": r, "league": "E0"}
            for d, h, a, r in rows
        ]
    )


def test_settlement_scores_against_the_real_result():
    preds = _predictions([(-24, "Arsenal", "Chelsea", 0.5, 0.27, 0.23)])
    ledger, _ = append_predictions(empty_ledger(), preds, now=NOW - pd.Timedelta(days=2))
    matches = _matches([("2026-08-14", "Arsenal", "Chelsea", "H")])
    ledger, stats = settle(ledger, matches, now=NOW)

    assert stats["settled"] == 1
    row = ledger.iloc[0]
    assert row["result"] == "H"
    expected = log_loss_per_match(np.array([[0.5, 0.27, 0.23]]), pd.Series(["H"]))[0]
    assert row["log_loss"] == pytest.approx(expected)
    assert row["brier"] == pytest.approx((0.5 - 1) ** 2 + 0.27**2 + 0.23**2)


def test_settlement_is_idempotent_and_leaves_unknown_fixtures_open():
    preds = _predictions([
        (-24, "Arsenal", "Chelsea", 0.5, 0.27, 0.23),
        (-24, "Leeds", "Everton", 0.4, 0.3, 0.3),
    ])
    ledger, _ = append_predictions(empty_ledger(), preds, now=NOW - pd.Timedelta(days=2))
    matches = _matches([("2026-08-14", "Arsenal", "Chelsea", "H")])

    ledger, first = settle(ledger, matches, now=NOW)
    assert (first["settled"], first["still_open"]) == (1, 1)
    ledger, second = settle(ledger, matches, now=NOW)
    assert second["settled"] == 0  # nothing double-scored
    assert ledger["result"].notna().sum() == 1


def test_future_fixtures_are_not_settled():
    preds = _predictions([(48, "Arsenal", "Chelsea", 0.5, 0.27, 0.23)])
    ledger, _ = append_predictions(empty_ledger(), preds, now=NOW)
    matches = _matches([("2026-08-17", "Arsenal", "Chelsea", "H")])
    ledger, stats = settle(ledger, matches, now=NOW)
    assert stats["settled"] == 0
    assert pd.isna(ledger.iloc[0]["result"])


def test_scoreboard_summarizes_settled_rows():
    preds = _predictions([
        (-48, "Arsenal", "Chelsea", 0.60, 0.25, 0.15),
        (-48, "Leeds", "Everton", 0.30, 0.30, 0.40),
    ])
    ledger, _ = append_predictions(empty_ledger(), preds, now=NOW - pd.Timedelta(days=3))
    matches = _matches([
        ("2026-08-13", "Arsenal", "Chelsea", "H"),
        ("2026-08-13", "Leeds", "Everton", "A"),
    ])
    ledger, _ = settle(ledger, matches, now=NOW)

    board = scoreboard(ledger)
    assert board["n_settled"] == 2 and board["n_open"] == 0
    assert board["favorite_hit_rate"] == pytest.approx(1.0)  # both favorites won
    assert board["log_loss_lo"] <= board["log_loss"] <= board["log_loss_hi"]
    assert board["sample_is_meaningful"] is False  # n=2 must not claim precision


def test_empty_ledger_round_trips(tmp_path):
    path = tmp_path / "ledger.parquet"
    assert load_ledger(path).empty
    assert scoreboard(empty_ledger()) == {"n_settled": 0, "n_open": 0}
    empty_ledger().to_parquet(path, index=False)
    assert load_ledger(path).empty


@pytest.mark.parametrize('offset', [-24, 0])
def test_new_predictions_at_or_after_kickoff_are_rejected(offset):
    predictions = _predictions([(offset, 'Arsenal', 'Chelsea', .5, .27, .23)])
    ledger, stats = append_predictions(empty_ledger(), predictions, now=NOW)
    assert ledger.empty
    assert stats == {'added': 0, 'updated': 0, 'frozen': 1}


@pytest.mark.parametrize('probabilities', [(np.nan, .2, .8), (np.inf, .2, .8),
    (-.1, .4, .7), (.2, .2, .2), (True, 0., 0.)])
def test_invalid_probabilities_are_rejected_at_recording_boundary(probabilities):
    predictions = _predictions([(24, 'Arsenal', 'Chelsea', *probabilities)])
    with pytest.raises(ValueError, match='probabilities'):
        append_predictions(empty_ledger(), predictions, now=NOW)


def _open_prediction(home='Manchester City', away='Manchester United', league='E0'):
    predictions = _predictions([(-24, home, away, .5, .27, .23)])
    predictions['league'] = league
    return append_predictions(empty_ledger(), predictions, now=NOW-pd.Timedelta(days=2))[0]


def test_explicit_team_aliases_settle():
    ledger = _open_prediction()
    matches = _matches([('2026-08-14', 'Man City', 'Man United', 'H')])
    result, stats = settle(ledger, matches, now=NOW)
    assert stats['settled'] == 1
    assert result.iloc[0]['result'] == 'H'


def test_similar_prefixes_do_not_match():
    ledger = _open_prediction('Manchester Rovers', 'Chelsea')
    matches = _matches([('2026-08-14', 'Manchester City', 'Chelsea', 'H')])
    result, stats = settle(ledger, matches, now=NOW)
    assert stats == {'settled': 0, 'still_open': 1}


def test_results_from_another_league_cannot_settle():
    ledger = _open_prediction('Arsenal', 'Chelsea', league='E1')
    matches = _matches([('2026-08-14', 'Arsenal', 'Chelsea', 'H')])
    assert settle(ledger, matches, now=NOW)[1] == {'settled': 0, 'still_open': 1}


def test_ambiguous_neighboring_dates_remain_open():
    ledger = _open_prediction('Arsenal', 'Chelsea')
    matches = _matches([('2026-08-13', 'Arsenal', 'Chelsea', 'H'),
                        ('2026-08-15', 'Arsenal', 'Chelsea', 'A')])
    assert settle(ledger, matches, now=NOW)[1] == {'settled': 0, 'still_open': 1}


@pytest.mark.parametrize('provider,source', [
    ('Tottenham Hotspur', 'Tottenham'), ('Wolverhampton Wanderers', 'Wolves'),
    ('Newcastle United', 'Newcastle'), ('Leeds United', 'Leeds'),
    ('Nottingham Forest', 'Nottm Forest'), ('Sheffield United', 'Sheffield United'),
])
def test_known_provider_aliases_settle(provider, source):
    ledger = _open_prediction(provider, 'Chelsea')
    matches = _matches([('2026-08-14', source, 'Chelsea', 'H')])
    assert settle(ledger, matches, now=NOW)[1]['settled'] == 1


def test_malformed_late_update_leaves_frozen_record_untouched():
    ledger = _open_prediction('Arsenal', 'Chelsea')
    before = ledger.copy(deep=True)
    bad = _predictions([(-24, 'Arsenal', 'Chelsea', np.nan, .2, .8)])
    after, stats = append_predictions(ledger, bad, now=NOW)
    pd.testing.assert_frame_equal(before, after)
    assert stats == {'added': 0, 'updated': 0, 'frozen': 1}


def test_single_neighboring_day_result_can_settle():
    ledger = _open_prediction('Arsenal', 'Chelsea')
    matches = _matches([('2026-08-13', 'Arsenal', 'Chelsea', 'D')])
    result, stats = settle(ledger, matches, now=NOW)
    assert stats['settled'] == 1
    assert result.iloc[0]['result'] == 'D'


def test_sheffield_wednesday_matches_ingest_canonical_name():
    ledger = _open_prediction('Sheffield Wednesday', 'Chelsea')
    matches = _matches([('2026-08-14', 'Sheffield Weds', 'Chelsea', 'H')])
    assert settle(ledger, matches, now=NOW)[1]['settled'] == 1


@pytest.mark.parametrize('kickoff', [None, pd.NaT])
def test_recording_missing_kickoff_is_rejected(kickoff):
    predictions = _predictions([(24, 'Arsenal', 'Chelsea', .5, .27, .23)])
    predictions['kickoff'] = kickoff
    with pytest.raises(ValueError, match='kickoff'):
        append_predictions(empty_ledger(), predictions, now=NOW)
