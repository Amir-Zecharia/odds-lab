import numpy as np
import pandas as pd
import pytest


def odds(event_id='event-1', kickoff='2026-10-21T18:00:00Z'):
    return [dict(id=event_id, home_team='Home', away_team='Away', commence_time=kickoff,
        bookmakers=[dict(key='pinnacle', markets=[dict(key='h2h', outcomes=[
            dict(name='Home', price=2), dict(name='Draw', price=3), dict(name='Away', price=4)])])])]


def result(event_id='event-1', kickoff='2026-10-21T18:00:00Z', home='2'):
    return [dict(id=event_id, completed=True, home_team='Home', away_team='Away',
        commence_time=kickoff, last_update='2026-10-21T20:00:00Z',
        scores=[dict(name='Home', score=home), dict(name='Away', score='1')])]


@pytest.fixture
def store(tmp_path):
    from src.live_store import LiveStore
    db = LiveStore(tmp_path / 'collector.sqlite3')
    yield db
    db.close()


def test_prediction_is_immutable_and_snapshot_identity_survives(store):
    row = store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')[0]
    assert row['event_id'] == 'event-1'
    assert store.save_prediction(row['snapshot_id'], 'market', [.5,.3,.2], 'baseline')
    assert not store.save_prediction(row['snapshot_id'], 'market', [.1,.1,.8], 'baseline')
    store.save_results('E0', result(), '2026-10-21T21:00:00Z')
    comparison = store.comparison_frame()
    np.testing.assert_allclose(comparison.iloc[0][['p_home','p_draw','p_away']].astype(float), [.5,.3,.2])
    assert comparison.iloc[0].log_loss == pytest.approx(-np.log(.5))
    assert comparison.iloc[0].brier == pytest.approx(.38)


def test_training_uses_last_snapshot_and_available_final_only(store):
    for hour in (10, 12):
        store.save_odds_snapshot('E0', odds(), f'2026-10-21T{hour}:00:00Z', str(hour))
    store.save_results('E0', result(), '2026-10-21T21:00:00Z')
    assert store.training_frame('2026-10-21T20:59:00Z').empty
    frame = store.training_frame('2026-10-22T00:00:00Z')
    assert len(frame) == 1
    assert pd.Timestamp(frame.iloc[0].captured_at) == pd.Timestamp('2026-10-21T12:00:00Z')
    assert len(store.snapshot_frame()) == 2


def test_final_id_does_not_match_other_fixture_with_same_teams(store):
    store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')
    store.save_results('E0', result('other'), '2026-10-21T21:00:00Z')
    assert store.training_frame('2026-10-22T00:00:00Z').empty


def test_capture_at_kickoff_is_ineligible(store):
    assert not store.save_odds_snapshot('E0', odds(), '2026-10-21T18:00:00Z', 'run')


def test_rescheduled_actual_kickoff_excludes_late_snapshot(store):
    store.save_odds_snapshot('E0', odds(), '2026-10-21T17:00:00Z', 'run')
    store.save_results('E0', result(kickoff='2026-10-21T16:00:00Z'), '2026-10-21T21:00:00Z')
    assert store.training_frame('2026-10-22T00:00:00Z').empty


def test_partial_retry_preserves_steps_completed_slot_is_noop(store):
    run = store.begin_run('2026-10-21T08:00:00+03:00', '2026-10-21T05:01:00Z')
    store.mark_step(run, 'E0', 'scores', 'completed')
    store.finish_run(run, 'partial', '2026-10-21T05:02:00Z', {})
    retry = store.begin_run('2026-10-21T08:00:00+03:00', '2026-10-21T05:03:00Z')
    assert retry == run
    assert store.step_done(retry, 'E0', 'scores')
    store.finish_run(run, 'completed', '2026-10-21T05:04:00Z', {})
    assert store.begin_run('2026-10-21T08:00:00+03:00', '2026-10-21T05:05:00Z') is None


def test_result_correction_updates_training_truth_without_mutating_forecast(store):
    row = store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')[0]
    store.save_prediction(row['snapshot_id'], 'market', [.5,.3,.2], 'baseline')
    assert store.save_results('E0', result(), '2026-10-21T21:00:00Z') == 1
    assert store.save_results('E0', result(), '2026-10-21T21:01:00Z') == 0
    assert store.save_results('E0', result(home='0'), '2026-10-22T01:00:00Z') == 1
    before = store.training_frame('2026-10-21T23:00:00Z')
    after = store.training_frame('2026-10-22T02:00:00Z')
    assert before.iloc[0].result == 'H'
    assert after.iloc[0].result == 'A'
    np.testing.assert_allclose(store.comparison_frame().iloc[0][['p_home','p_draw','p_away']].astype(float), [.5,.3,.2])


def test_retry_keeps_original_capture_and_cannot_backfill_future_model(store):
    row = store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')[0]
    again = store.save_odds_snapshot('E0', odds(), '2026-10-21T12:00:00Z', 'run')[0]
    assert again['snapshot_id'] == row['snapshot_id']
    assert again['captured_at'] == pd.Timestamp('2026-10-21T10:00:00Z')
    assert not store.save_prediction(row['snapshot_id'], 'later-model', [.5,.3,.2], 'candidate', trained_at='2026-10-21T11:00:00Z')
    assert store.comparison_frame().empty


def test_league_is_part_of_exact_event_identity(store):
    store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')
    store.save_results('D1', result(), '2026-10-21T21:00:00Z')
    assert store.training_frame('2026-10-22T00:00:00Z').empty


def test_future_kickoff_cannot_be_training_label(store):
    store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')
    assert store.save_results('E0', result(), '2026-10-21T11:00:00Z') == 0
    assert store.training_frame('2026-10-21T12:00:00Z').empty


def test_rescheduled_after_actual_kickoff_forecast_unscored(store):
    row = store.save_odds_snapshot('E0', odds(), '2026-10-21T17:00:00Z', 'run')[0]
    store.save_prediction(row['snapshot_id'], 'market', [.5,.3,.2], 'baseline')
    store.save_results('E0', result(kickoff='2026-10-21T16:00:00Z'), '2026-10-21T21:00:00Z')
    assert store.comparison_frame().log_loss.isna().all()


def test_naive_timestamp_rejected_and_empty_frames_safe(store):
    assert store.comparison_frame().empty
    assert store.training_frame('2026-10-21T12:00:00Z').empty
    with pytest.raises(ValueError):
        store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00', 'run')


def test_stale_provider_final_cannot_replace_newer_correction(store):
    row = store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')[0]
    store.save_prediction(row['snapshot_id'], 'market', [.5, .3, .2], 'baseline')

    def observe(home, updated, observed):
        payload = result(home=home)
        payload[0]['last_update'] = updated
        assert store.save_results('E0', payload, observed) == 1

    observe('2', '2026-10-21T20:00:00Z', '2026-10-21T21:00:00Z')
    observe('0', '2026-10-21T22:00:00Z', '2026-10-21T23:00:00Z')
    observe('2', '2026-10-21T20:00:00Z', '2026-10-22T00:00:00Z')
    assert store.training_frame('2026-10-22T00:30:00Z').iloc[0].result == 'A'
    assert store.comparison_frame().iloc[0].log_loss == pytest.approx(-np.log(.2))
    observe('1', '2026-10-22T01:00:00Z', '2026-10-22T02:00:00Z')
    assert len(store.results_frame()) == 4
    assert store.training_frame('2026-10-21T22:30:00Z').iloc[0].result == 'H'
    assert store.training_frame('2026-10-22T01:30:00Z').iloc[0].result == 'A'
    assert store.training_frame('2026-10-22T03:00:00Z').iloc[0].result == 'D'
    assert store.comparison_frame().iloc[0].log_loss == pytest.approx(-np.log(.3))
    assert not store.save_prediction(row['snapshot_id'], 'other', [.2, .2, .6], 'baseline')
    np.testing.assert_allclose(store.comparison_frame().iloc[0][['p_home','p_draw','p_away']].astype(float), [.5,.3,.2])


def test_missing_provider_update_uses_observation_fallback_without_overriding_known_version(store):
    store.save_odds_snapshot('E0', odds(), '2026-10-21T10:00:00Z', 'run')
    for home, observed in [('2', '21'), ('0', '22')]:
        payload = result(home=home)
        payload[0].pop('last_update')
        store.save_results('E0', payload, f'2026-10-21T{observed}:00:00Z')
    assert store.training_frame('2026-10-21T22:30:00Z').iloc[0].result == 'A'
    store.save_results('E0', result(home='1'), '2026-10-21T23:00:00Z')
    payload = result(home='2')
    payload[0].pop('last_update')
    store.save_results('E0', payload, '2026-10-22T00:00:00Z')
    assert store.training_frame('2026-10-22T01:00:00Z').iloc[0].result == 'D'
    assert len(store.results_frame()) == 4


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), -float('inf'), np.float32('nan'), np.float64('inf')])
def test_malformed_nonfinite_neighbor_is_archived_without_losing_good_fixture(store, nonfinite):
    import json
    broken = odds(event_id='malformed')[0]
    broken['bookmakers'][0]['markets'][0]['outcomes'][0]['price'] = nonfinite
    events = [broken, *odds()]
    rows = store.save_odds_snapshot('E0', events, '2026-10-21T10:00:00Z', 'run')
    assert [row['event_id'] for row in rows] == ['event-1']
    payload = store.conn.execute('SELECT payload FROM archives').fetchone()[0]
    saved = json.loads(payload, parse_constant=lambda value: pytest.fail(f'Invalid JSON constant {value}'))
    assert saved[0]['bookmakers'][0]['markets'][0]['outcomes'][0]['price'] is None
    assert saved[1]['id'] == 'event-1'
