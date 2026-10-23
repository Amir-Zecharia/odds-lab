"""Collector acceptance uses fake HTTP and the real database and learner."""
import json
import os

import pytest

from tests.test_api_football import Response, envelope, fake_http, fixture, league, odds


def options(tmp_path):
    return dict(db_path=tmp_path/'db.sqlite3', settings_path=tmp_path/'settings.json')


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.delenv('API_FOOTBALL_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')


def test_israel_no_key_has_no_http_and_defaults_remain_historical_six(tmp_path, monkeypatch):
    from src.collector import run_collection, load_settings
    from src.live import SPORT_KEYS
    monkeypatch.setattr('requests.get', lambda *a, **k: pytest.fail('unexpected HTTP'))
    assert set(load_settings(path=tmp_path/'missing.json')['leagues']) == set(SPORT_KEYS)
    out = run_collection(leagues=['IL1'], now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'needs_key'


def test_independent_private_key_save_clear_env_and_public_redaction(tmp_path, monkeypatch):
    from src.collector import load_settings, save_settings, public_status
    path = tmp_path/'settings.json'
    saved = save_settings({'api_key': 'odds-private-sentinel', 'api_football_key': 'football-private-sentinel'}, path=path)
    assert 'private-sentinel' not in json.dumps(saved)
    save_settings({'api_football_key': ''}, path=path)
    assert load_settings(path=path)['api_football_key'] == 'football-private-sentinel'
    assert os.stat(path).st_mode & 0o077 == 0
    monkeypatch.setenv('API_FOOTBALL_KEY', 'football-env-sentinel')
    assert load_settings(path=path)['api_football_key'] == 'football-env-sentinel'
    status = public_status(**options(tmp_path))
    assert status['key_configured'] is True
    assert status['api_football_key_configured'] is True
    assert 'private-sentinel' not in json.dumps(status) and 'env-sentinel' not in json.dumps(status)
    monkeypatch.delenv('API_FOOTBALL_KEY')
    save_settings({'clear_api_football_key': True}, path=path)
    settings = load_settings(path=path)
    assert not settings['api_football_key'] and settings['api_key'] == 'odds-private-sentinel'


@pytest.mark.parametrize('configured', ['odds_api', 'api_football'])
def test_one_missing_key_keeps_other_provider_healthy_and_retryable(tmp_path, monkeypatch, configured):
    from src.collector import run_collection
    calls = []
    def handler(endpoint, request):
        calls.append(endpoint)
        if configured == 'odds_api':
            assert endpoint in ('scores', 'odds')
            return Response([])
        assert endpoint in ('leagues', 'fixtures', 'odds')
        return Response(envelope([league()] if endpoint == 'leagues' else []))
    fake_http(monkeypatch, handler)
    keys = {'api_key': 'odds-key'} if configured == 'odds_api' else {'api_football_key': 'football-key'}
    out = run_collection(**keys, leagues=['E0', 'IL1'], now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'partial'
    assert calls, 'configured provider must still collect'
    # Supplying the missing key resumes the same slot rather than freezing the failure.
    fake_http(monkeypatch, lambda endpoint, request: Response(
        envelope([league()] if endpoint == 'leagues' else [])
        if 'x-apisports-key' in request.get('headers', {}) else []))
    out = run_collection(api_key='odds-key', api_football_key='football-key', leagues=['E0', 'IL1'],
                         now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'completed'


def test_israel_auth_failure_does_not_stop_european_requests_or_leak_body(tmp_path, monkeypatch):
    from src.collector import run_collection
    european = []
    def handler(endpoint, request):
        if 'x-apisports-key' in request.get('headers', {}):
            return Response(envelope([], errors={'token': 'football-secret-sentinel'}), 401)
        european.append(endpoint)
        return Response([])
    fake_http(monkeypatch, handler)
    out = run_collection(api_key='odds-key', api_football_key='football-secret-sentinel',
                         leagues=['IL1', 'E0'], now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'partial'
    assert european == ['scores', 'odds']
    assert 'football-secret-sentinel' not in json.dumps(out)
    assert b'football-secret-sentinel' not in (tmp_path/'db.sqlite3').read_bytes()


def test_two_slot_israel_forecast_final_learning_and_later_fixture(tmp_path, monkeypatch):
    import pandas as pd
    from src.collector import run_collection
    from src.live_store import LiveStore
    second = [False]
    def handler(endpoint, request):
        if endpoint == 'leagues': return Response(envelope([league()]))
        if endpoint == 'fixtures':
            rows = [fixture(42, 'FT'), fixture(43, date='2026-10-23T18:00:00+00:00')] if second[0] else [fixture()]
            return Response(envelope(rows))
        return Response(envelope([odds(43 if second[0] else 42)]))
    calls = fake_http(monkeypatch, handler)
    opts = dict(api_football_key='football-secret-sentinel', leagues=['IL1'], **options(tmp_path))
    assert run_collection(**opts, now='2026-10-21T10:00:00Z')['status'] == 'completed'
    store = LiveStore(tmp_path/'db.sqlite3')
    before = store.comparison_frame().copy()
    store.close()
    n_calls = len(calls)
    assert run_collection(**opts, now='2026-10-21T10:01:00Z')['status'] == 'already_completed'
    assert len(calls) == n_calls
    second[0] = True
    assert run_collection(**opts, now='2026-10-22T10:00:00Z')['status'] == 'completed'
    store = LiveStore(tmp_path/'db.sqlite3')
    try:
        after = store.comparison_frame()
        first = after[after.event_id == 'api-football:42']
        assert len(first) == 3 and first.log_loss.notna().all()
        assert first[['p_home', 'p_draw', 'p_away']].values.tolist() == before[['p_home', 'p_draw', 'p_away']].values.tolist()
        model = store.latest_model()
        israel = model['league_models']['IL1']
        assert 'api-football:42' in israel['training_event_ids']
        assert 'api-football:42' not in model['training_event_ids']
        assert israel['historical_lineage']['count'] == 0
        assert not israel['validation']['promoted']
        later = after[after.event_id == 'api-football:43']
        assert len(later) == 3 and later.log_loss.isna().all()
        assert israel['version'] in set(later.model_version)
        assert model['version'] != israel['version']
        assert set(store.training_frame('2026-10-22T10:00:00Z').league) == {'IL1'}
        json.dumps(model, allow_nan=False)
    finally:
        store.close()


def test_environment_key_is_effective_but_never_persisted(tmp_path, monkeypatch):
    from src.collector import load_settings, save_settings
    path = tmp_path/'settings.json'
    save_settings({'api_football_key': 'stored-key'}, path=path)
    monkeypatch.setenv('API_FOOTBALL_KEY', 'environment-key')
    save_settings({'enabled': True}, path=path)
    assert json.loads(path.read_text())['api_football_key'] == 'stored-key'
    save_settings({'clear_api_football_key': True}, path=path)
    assert json.loads(path.read_text())['api_football_key'] == ''
    assert load_settings(path=path)['api_football_key'] == 'environment-key'


def test_promoted_global_model_cannot_activate_unvalidated_israel(tmp_path, monkeypatch):
    import numpy as np
    from src.collector import run_collection
    from src.live_store import LiveStore
    from src.live_learning import fit_candidate
    calls = []
    def fit(historical, live, as_of):
        model = fit_candidate(historical, live, as_of)
        calls.append((historical.copy(), live.copy()))
        model['version'] = 'europe-promoted' if len(calls) == 1 else 'israel-unvalidated'
        model['validation']['promoted'] = len(calls) == 1
        return model
    monkeypatch.setattr('src.live_learning.fit_candidate', fit)
    monkeypatch.setattr('src.live_learning.predict_candidate', lambda model, rows: np.array([[.8, .1, .1]]))
    def handler(endpoint, request):
        if endpoint == 'leagues': return Response(envelope([league()]))
        if endpoint == 'fixtures': return Response(envelope([fixture()]))
        return Response(envelope([odds()]))
    fake_http(monkeypatch, handler)
    out = run_collection(api_football_key='test-key', leagues=['IL1'], now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'completed'
    store = LiveStore(tmp_path/'db.sqlite3')
    try:
        rows = store.comparison_frame()
        active = rows.loc[rows.kind == 'active'].iloc[0]
        baseline = rows.loc[rows.kind == 'baseline'].iloc[0]
        candidate = rows.loc[rows.kind == 'candidate'].iloc[0]
        assert active.model_version == 'market'
        assert candidate.model_version == 'israel-unvalidated'
        assert active[['p_home', 'p_draw', 'p_away']].tolist() == baseline[['p_home', 'p_draw', 'p_away']].tolist()
        assert candidate.p_home == .8
        assert calls[1][0].empty
    finally:
        store.close()


def test_public_estimates_exclude_israel_and_model_details_are_compact(tmp_path):
    from src.collector import public_status, save_settings
    from src.live_store import LiveStore
    save_settings({'leagues': ['E0', 'IL1']}, path=tmp_path/'settings.json')
    store = LiveStore(tmp_path/'db.sqlite3')
    store.save_model({'version': 'combined', 'trained_at': '2026-10-21T10:00:00Z', 'league_models': {'IL1': {
        'version': 'israel', 'training_event_ids': ['private-event'], 'coefficients': [1],
        'validation': {'promoted': False}}}})
    store.close()
    status = public_status(**options(tmp_path))
    assert status['estimated_credits_daily'] == 9
    assert status['estimated_credits_30_days'] == 270
    assert status['collection']['model']['league_models']['IL1']['version'] == 'israel'
    assert 'private-event' not in json.dumps(status)
    assert 'coefficients' not in json.dumps(status)


def test_failed_israel_request_persists_latest_quota_without_secrets(tmp_path, monkeypatch):
    from src.collector import run_collection, public_status
    from src.api_football import ApiFootballError
    class Client:
        def __init__(self, *args, **kwargs):
            self.quota = {'remaining': 10}
        def fetch_scores(self):
            return [], dict(self.quota)
        def fetch_events(self):
            self.quota = {'remaining': 0, 'minute_remaining': 0, 'limit': 100,
                          'minute_limit': 10, 'secret': 'sentinel-private-key'}
            raise ApiFootballError('quota')
    monkeypatch.setattr('src.api_football.ApiFootballClient', Client)
    out = run_collection(api_football_key='sentinel-private-key', leagues=['IL1'],
                         now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'partial'
    assert out['quota']['IL1:odds']['remaining'] == '0'
    assert out['errors']['IL1:odds']['quota']['minute_remaining'] == '0'
    status = public_status(**options(tmp_path))
    serialized = json.dumps(status)
    assert 'sentinel-private-key' not in serialized
    assert 'minute_remaining' in serialized
    assert b'sentinel-private-key' not in (tmp_path/'db.sqlite3').read_bytes()


def test_partial_odds_pagination_quota_is_persisted_and_never_publishes_partial_snapshot(tmp_path, monkeypatch):
    from src.collector import run_collection
    from src.live_store import LiveStore
    pages = []
    def handler(endpoint, request):
        if endpoint == 'leagues': return Response(envelope([league()]))
        if endpoint == 'fixtures': return Response(envelope([fixture()]))
        page = request['params']['page']
        pages.append(page)
        if page == 1:
            return Response(envelope([odds()], page=1, total=2))
        return Response(envelope([], errors={'quota': 'sentinel-private-key'}), 429,
                        headers={'x-ratelimit-requests-remaining': '0',
                                 'x-ratelimit-requests-limit': '100'})
    fake_http(monkeypatch, handler)
    out = run_collection(api_football_key='sentinel-private-key', leagues=['IL1'],
                         now='2026-10-21T10:00:00Z', **options(tmp_path))
    assert out['status'] == 'partial' and pages == [1, 2]
    assert out['quota']['IL1:odds']['remaining'] == '0'
    assert out['errors']['IL1:odds']['reason'] == 'quota'
    store = LiveStore(tmp_path/'db.sqlite3')
    try:
        assert store.snapshot_frame().empty
    finally:
        store.close()
    assert 'sentinel-private-key' not in json.dumps(out)
    assert b'sentinel-private-key' not in (tmp_path/'db.sqlite3').read_bytes()
