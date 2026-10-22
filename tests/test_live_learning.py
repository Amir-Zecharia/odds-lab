import json
import numpy as np
import pandas as pd


def training(n=30):
    kickoff = pd.date_range('2026-01-01', periods=n, freq='D', tz='UTC')
    return pd.DataFrame(dict(event_id=[f'id-{i}' for i in range(n)], league='E0',
        home_team='Home', away_team='Away', kickoff=kickoff, captured_at=kickoff-pd.Timedelta(hours=2),
        result_observed_at=kickoff+pd.Timedelta(hours=3), p_home=.5, p_draw=.3, p_away=.2,
        result=['HDA'[i%3] for i in range(n)]))


def test_small_sample_fallback_is_json_and_returns_market():
    from src.live_learning import fit_candidate, predict_candidate
    frame = training()
    model = fit_candidate(pd.DataFrame(), frame, '2026-03-01T00:00:00Z')
    json.dumps(model, allow_nan=False)
    assert not model['validation']['promoted']
    assert model['validation']['reason']
    probabilities = predict_candidate(model, frame)
    assert probabilities.shape == (len(frame), 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(probabilities, frame[['p_home','p_draw','p_away']])


def test_training_lineage_excludes_future_observations():
    from src.live_learning import fit_candidate
    frame = training()
    frame.loc[0, 'result_observed_at'] = pd.Timestamp('2027-01-01T00:00:00Z')
    model = fit_candidate(pd.DataFrame(), frame, '2026-03-01T00:00:00Z')
    assert 'id-0' not in model['training_event_ids']
    assert pd.Timestamp(model['training_cutoff']) <= pd.Timestamp('2026-03-01T00:00:00Z')


def test_repeated_snapshots_do_not_inflate_training_count():
    from src.live_learning import fit_candidate
    frame = training()
    model = fit_candidate(pd.DataFrame(), pd.concat([frame]*3), '2026-03-01T00:00:00Z')
    ids = model['training_event_ids']
    assert len(ids) == len(set(ids))
    assert model['validation']['n_train'] + model['validation']['n_validation'] <= len(frame)


def large_training(n=1600):
    frame = training(n)
    rng = np.random.default_rng(11)
    groups = np.arange(n) % 3
    frame['home_team'] = [f'Team {g}' for g in groups]
    frame['away_team'] = 'Visitor'
    frame[['p_home', 'p_draw', 'p_away']] = 1 / 3
    frame['result'] = [rng.choice(list('HDA'), p=np.roll([.88, .06, .06], g)) for g in groups]
    return frame


def test_real_learning_promotes_and_serialized_team_features_change_forecasts():
    from src.live_learning import fit_candidate, predict_candidate
    frame = large_training()
    model = fit_candidate(pd.DataFrame(), frame, '2032-01-01T00:00:00Z')
    model = json.loads(json.dumps(model, allow_nan=False))
    assert model['validation']['promoted'], model['validation']
    assert model['validation']['ci']['hi'] < 0
    assert model['validation']['candidate_brier'] <= model['validation']['baseline_brier']
    probabilities = predict_candidate(model, frame.iloc[:3])
    assert (probabilities.diagonal() > .7).all()
    assert len(model['training_event_ids']) == len(frame)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert model == fit_candidate(pd.DataFrame(), frame, '2032-01-01T00:00:00Z')


def test_training_excludes_future_kickoff_capture_labels_and_duplicates():
    from src.live_learning import fit_candidate
    frame = training()
    frame.loc[0, 'captured_at'] = frame.loc[0, 'kickoff']
    frame.loc[1, 'result_observed_at'] = pd.Timestamp('2030-01-01T00:00:00Z')
    frame.loc[2, 'kickoff'] = pd.Timestamp('2030-01-01T00:00:00Z')
    model = fit_candidate(pd.DataFrame(), pd.concat([frame, frame]), '2026-03-01T00:00:00Z')
    assert len(model['training_event_ids']) == 27
    assert not set(['id-0', 'id-1', 'id-2']) & set(model['training_event_ids'])


def test_large_uninformative_candidate_is_not_promoted():
    from src.live_learning import fit_candidate
    frame = large_training()
    frame['home_team'] = 'Same'
    frame['result'] = ['HDA'[i % 3] for i in range(len(frame))]
    model = fit_candidate(pd.DataFrame(), frame, '2032-01-01T00:00:00Z')
    assert not model['validation']['promoted']
    assert model['validation']['reason']


def test_holdout_dates_and_label_availability_are_separate():
    from src.live_learning import fit_candidate
    frame = large_training()
    frame['kickoff'] = frame.kickoff.dt.floor('D')
    # A past fixture's late result must not train the validation estimator.
    frame.loc[0, 'result_observed_at'] = frame.kickoff.iloc[1250]
    model = fit_candidate(pd.DataFrame(), frame, '2032-01-01T00:00:00Z')
    split = pd.Timestamp(model['validation']['split_date'])
    assert model['validation']['n_train'] == ((frame.kickoff < split) & (frame.result_observed_at < split)).sum()
    assert 'id-0' in model['training_event_ids']  # available to subsequent full refit


def test_historical_uses_preclose_pinnacle_else_consensus():
    from src.live_learning import _prepare
    historical = pd.DataFrame(dict(date=['2025-01-01', '2025-01-02'], league='E0',
        home_team=['A', 'B'], away_team='C', result=['H', 'D'],
        p_PS_H=[.6, np.nan], p_PS_D=[.2, np.nan], p_PS_A=[.2, np.nan],
        p_cons_H=.4, p_cons_D=.3, p_cons_A=.3))
    prepared = _prepare(historical, pd.DataFrame(), pd.Timestamp('2026-01-01T00:00:00Z'))
    np.testing.assert_allclose(prepared.p_home, [.6, .4])


def test_newly_observed_results_enter_only_subsequent_refit():
    from src.live_learning import fit_candidate, predict_candidate
    frame = large_training()
    extra = training(90)
    extra['event_id'] = [f'new-{i}' for i in range(len(extra))]
    extra['kickoff'] = pd.date_range('2031-01-01', periods=len(extra), tz='UTC')
    extra['captured_at'] = extra.kickoff - pd.Timedelta(hours=2)
    extra['result_observed_at'] = extra.kickoff + pd.Timedelta(hours=3)
    extra['home_team'], extra['away_team'], extra['result'] = 'Team 0', 'Visitor', 'A'
    extra[['p_home', 'p_draw', 'p_away']] = 1 / 3
    all_rows = pd.concat([frame, extra], ignore_index=True)
    before = fit_candidate(pd.DataFrame(), all_rows, '2031-01-01T00:00:00Z')
    after = fit_candidate(pd.DataFrame(), all_rows, '2032-01-01T00:00:00Z')
    assert not any(e.startswith('new-') for e in before['training_event_ids'])
    assert 'new-89' in after['training_event_ids']
    assert after['version'] != before['version']
    assert predict_candidate(after, frame.iloc[:1])[0, 2] > predict_candidate(before, frame.iloc[:1])[0, 2]


def test_historical_same_day_label_is_unavailable_and_overlap_is_one_match():
    from src.live_learning import _prepare
    historical = pd.DataFrame(dict(date=['2026-01-01'], league='E0', home_team='Home',
        away_team='Away', result='H', p_cons_H=.5, p_cons_D=.3, p_cons_A=.2))
    assert _prepare(historical, pd.DataFrame(), pd.Timestamp('2026-01-01T23:59:00Z')).empty
    live = training(1)
    combined = _prepare(historical, live, pd.Timestamp('2026-01-03T00:00:00Z'))
    assert len(combined) == 1
    assert combined.source.iloc[0] == 'live'


def test_historical_lineage_is_compact_stable_and_sensitive_to_training_data():
    from src.live_learning import fit_candidate
    historical = pd.DataFrame(dict(date=['2025-01-01', '2025-01-02'], league='E0',
        home_team=['A', 'B'], away_team='C', result=['H', 'D'],
        p_cons_H=.4, p_cons_D=.3, p_cons_A=.3))
    live = training(3)
    cutoff = '2026-03-01T00:00:00Z'
    model = fit_candidate(historical, live, cutoff)
    lineage = model['historical_lineage']
    assert lineage['count'] == 2
    assert lineage['date_start'] == '2025-01-01T00:00:00+00:00'
    assert lineage['date_end'] == '2025-01-02T00:00:00+00:00'
    assert len(lineage['sha256']) == 64
    assert model['training_event_ids'] == live.event_id.tolist()
    assert len(model['event_lineage']) == len(live)
    assert all(row['source'] == 'live' for row in model['event_lineage'])
    assert 'historical:E0' not in json.dumps(model, allow_nan=False)
    shuffled = fit_candidate(historical.iloc[::-1], live, cutoff)
    assert shuffled['historical_lineage'] == lineage
    historical.loc[0, 'result'] = 'A'
    changed = fit_candidate(historical, live, cutoff)
    assert changed['historical_lineage']['sha256'] != lineage['sha256']
    historical.loc[0, 'p_cons_H'] = .5
    assert fit_candidate(historical, live, cutoff)['historical_lineage']['sha256'] != changed['historical_lineage']['sha256']
