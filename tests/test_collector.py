import json
import os
import pytest


def test_missing_key_has_no_http_and_honest_state(tmp_path, monkeypatch):
    from src.collector import run_collection
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    monkeypatch.setattr('requests.get', lambda *a, **k: pytest.fail('unexpected HTTP'))
    out = run_collection(db_path=tmp_path/'db.sqlite3', settings_path=tmp_path/'settings.json',
                         now='2026-10-21T10:00:00Z')
    assert out['status'] == 'needs_key'


def test_settings_private_mode_and_empty_key_preserves_secret(tmp_path):
    from src.collector import save_settings, load_settings
    path = tmp_path/'settings.json'
    save_settings({'api_key':'private-sentinel', 'enabled':False}, path=path)
    save_settings({'api_key':'', 'enabled':True}, path=path)
    assert load_settings(path=path)['api_key'] == 'private-sentinel'
    assert os.stat(path).st_mode & 0o077 == 0


def test_new_settings_disabled(tmp_path, monkeypatch):
    from src.collector import load_settings
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    assert load_settings(path=tmp_path/'missing.json')['enabled'] is False


def _odds(identity='first', kickoff='2026-10-21T18:00:00Z'):
    return [{'id': identity, 'home_team': 'Home', 'away_team': 'Away', 'commence_time': kickoff,
             'bookmakers': [{'key': 'pinnacle', 'markets': [{'key': 'h2h', 'outcomes': [
                 {'name': 'Home', 'price': 2}, {'name': 'Draw', 'price': 3}, {'name': 'Away', 'price': 4}]}]}]}]


def test_real_store_learner_flow_and_slot_idempotence(tmp_path, monkeypatch):
    from src.collector import run_collection
    from src.live_store import LiveStore
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    calls = []
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: (calls.append('scores') or [], {}))
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: (calls.append('odds') or _odds(), {'remaining': '498'}))
    options = dict(api_key='secret', leagues=['E0'], db_path=tmp_path/'db', settings_path=tmp_path/'settings')
    assert run_collection(**options, now='2026-10-21T10:00:00Z')['status'] == 'completed'
    store = LiveStore(tmp_path/'db')
    before = store.comparison_frame().copy()
    store.close()
    assert run_collection(**options, now='2026-10-21T10:01:00Z')['status'] == 'already_completed'
    assert calls == ['scores', 'odds']
    final = dict(id='first', completed=True, home_team='Home', away_team='Away', commence_time='2026-10-21T18:00:00Z',
                 last_update='2026-10-21T20:00:00Z', scores=[dict(name='Home', score='2'), dict(name='Away', score='1')])
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: ([final], {}))
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: (_odds('later', '2026-10-23T18:00:00Z'), {}))
    assert run_collection(**options, now='2026-10-22T10:00:00Z')['status'] == 'completed'
    store = LiveStore(tmp_path/'db')
    try:
        after = store.comparison_frame()
        old = after[after.event_id == 'first']
        assert old.log_loss.notna().all()
        assert old[['p_home', 'p_draw', 'p_away']].values.tolist() == before[['p_home', 'p_draw', 'p_away']].values.tolist()
        assert 'first' in store.latest_model()['training_event_ids']
        assert len(after[after.event_id == 'later']) == 3
    finally:
        store.close()


def test_failures_redacted_and_successful_steps_resumed(tmp_path, monkeypatch):
    from src.collector import run_collection
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    calls = []
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: (calls.append('scores') or [], {}))
    def fail(*a, **k):
        raise RuntimeError('https://provider?apiKey=secret-sentinel')
    monkeypatch.setattr('src.live.fetch_events', fail)
    options = dict(api_key='secret-sentinel', leagues=['E0'], db_path=tmp_path/'db', settings_path=tmp_path/'settings', now='2026-10-21T10:00:00Z')
    assert run_collection(**options)['status'] == 'partial'
    assert b'secret-sentinel' not in (tmp_path/'db').read_bytes()
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: (_odds(), {}))
    assert run_collection(**options)['status'] == 'completed'
    assert calls == ['scores']


def test_overlap_is_rejected(tmp_path):
    import fcntl
    from src.collector import run_collection
    lock = open(tmp_path/'db.lock', 'a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert run_collection(db_path=tmp_path/'db', settings_path=tmp_path/'settings')['status'] == 'busy'
    finally:
        lock.close()


def test_actual_capture_clock_is_after_provider_call(tmp_path, monkeypatch):
    from src.collector import run_collection
    from src.live_store import LiveStore
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    clock = ['2026-10-21T10:00:00Z']
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: ([], {}))
    def odds(*a, **k):
        clock[0] = '2026-10-21T10:05:00Z'
        return _odds(), {}
    monkeypatch.setattr('src.live.fetch_events', odds)
    run_collection(api_key='key', leagues=['E0'], db_path=tmp_path/'db', settings_path=tmp_path/'settings', now=lambda: clock[0])
    store = LiveStore(tmp_path/'db')
    try:
        import pandas as pd
        assert pd.Timestamp(store.snapshot_frame().iloc[0].captured_at) == pd.Timestamp(clock[0])
    finally:
        store.close()


@pytest.mark.parametrize('code', [401, 429])
def test_global_auth_or_quota_failure_stops_all_requests(tmp_path, monkeypatch, code):
    import requests
    from src.collector import run_collection
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    calls = []
    def fail(*a, **k):
        calls.append('scores')
        response = requests.Response()
        response.status_code = code
        raise requests.HTTPError('key=secret', response=response)
    monkeypatch.setattr('src.live.fetch_scores', fail)
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: pytest.fail('must stop after global failure'))
    out = run_collection(api_key='secret', leagues=['E0', 'E1'], db_path=tmp_path/'db', settings_path=tmp_path/'settings')
    assert out['status'] == 'partial'
    assert calls == ['scores']
    assert 'secret' not in json.dumps(out)


def test_quota_cost_boundary_and_resume(tmp_path, monkeypatch):
    from src.collector import run_collection
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    calls = []
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: (calls.append(('scores', k['sport'])) or [], {'remaining': '1'}))
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: (calls.append(('odds', k['sport'])) or [], {'remaining': '0'}))
    opts = dict(api_key='secret', leagues=['E0', 'E1'], db_path=tmp_path/'db', settings_path=tmp_path/'settings', now='2026-10-21T10:00:00Z')
    assert run_collection(**opts)['status'] == 'partial'
    assert calls == [('scores', 'soccer_epl'), ('odds', 'soccer_epl')]
    calls.clear()
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: (calls.append(('scores', k['sport'])) or [], {'remaining': '10'}))
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: (calls.append(('odds', k['sport'])) or [], {'remaining': '9'}))
    assert run_collection(**opts)['status'] == 'completed'
    assert calls == [('scores', 'soccer_efl_champ'), ('odds', 'soccer_efl_champ')]


def test_default_seed_bootstraps_but_custom_directory_stays_isolated(tmp_path, monkeypatch):
    from src import config
    from src.collector import run_collection
    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(config, 'PROCESSED_DIR', tmp_path/'data/processed')
    calls = []
    def bootstrap():
        import pandas as pd
        calls.append('bootstrap')
        config.PROCESSED_DIR.mkdir(parents=True)
        pd.DataFrame({'seed': [1]}).to_parquet(config.PROCESSED_DIR/'enriched.parquet')
    monkeypatch.setattr('src.bundle.bootstrap', bootstrap)
    monkeypatch.setattr('src.live.fetch_scores', lambda *a, **k: ([], {}))
    monkeypatch.setattr('src.live.fetch_events', lambda *a, **k: ([], {}))
    original = __import__('src.live_learning', fromlist=['fit_candidate']).fit_candidate
    def fit(historical, live_training, as_of):
        calls.append(len(historical))
        return original(historical, live_training, as_of)
    monkeypatch.setattr('src.live_learning.fit_candidate', fit)
    opts = dict(api_key='secret', leagues=['E0'], db_path=tmp_path/'db', settings_path=tmp_path/'settings')
    assert run_collection(**opts, now='2026-10-21T10:00:00Z')['status'] == 'completed'
    assert calls == ['bootstrap', 1]
    calls.clear()
    monkeypatch.setattr(config, 'PROCESSED_DIR', tmp_path/'custom')
    assert run_collection(**opts, now='2026-10-22T10:00:00Z')['status'] == 'completed'
    assert calls == [0]


def test_public_status_model_is_concise(tmp_path):
    from src.collector import public_status
    from src.live_store import LiveStore
    store = LiveStore(tmp_path/'db')
    store.save_model({'version':'test', 'trained_at':'2026-10-21T10:00:00Z', 'validation':{'promoted':False},
                      'coefficients': [1,2,3], 'training_event_ids':['private-event']})
    store.close()
    status = public_status(db_path=tmp_path/'db', settings_path=tmp_path/'settings')
    assert status['collection']['model']['version'] == 'test'
    assert 'coefficients' not in json.dumps(status)
    assert 'private-event' not in json.dumps(status)


def test_fallback_day_slot_has_one_identity_and_next_schedule_uses_rules(monkeypatch):
    import os
    import time
    import pandas as pd
    from src.collector import scheduled_slot, next_schedule
    previous = os.environ.get('TZ')
    try:
        monkeypatch.setenv('TZ', 'Asia/Jerusalem')
        time.tzset()
        first = scheduled_slot('2026-10-24T21:30:00Z')
        later = scheduled_slot('2026-10-25T01:30:00Z')
        assert first == later
        assert pd.Timestamp(first) == pd.Timestamp('2026-10-24T21:00:00Z')
        upcoming = next_schedule('2026-10-24T21:30:00Z')
        assert pd.Timestamp(upcoming) == pd.Timestamp('2026-10-25T06:00:00Z')
        assert upcoming.endswith('+02:00')
    finally:
        if previous is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = previous
        time.tzset()
