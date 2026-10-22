import numpy as np
import pytest

from src.live import parse_events


def _book(key, price=2.0):
    return {'key': key, 'markets': [{'key': 'h2h', 'outcomes': [
        {'name': 'Home', 'price': price}, {'name': 'Draw', 'price': 3.5},
        {'name': 'Away', 'price': 4.0}]}]}


def _event():
    return {'home_team': 'Home', 'away_team': 'Away',
        'commence_time': '2026-08-15T14:00:00Z', 'bookmakers': [_book('pinnacle')]}


@pytest.mark.parametrize('price', [None, 'bad', True, float('nan'), float('inf'), 1.0])
def test_invalid_preferred_price_preserves_valid_fallback(price):
    event = _event()
    event['bookmakers'] = [_book('pinnacle', price), _book('unibet', 2.2)]
    out = parse_events([event])
    assert len(out) == 1
    assert out.iloc[0]['books'] == 1
    assert out.iloc[0]['source'] == 'average of 1 books'
    probabilities = out[['p_home', 'p_draw', 'p_away']].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


@pytest.mark.parametrize('timestamp', ['not-a-date', None])
def test_bad_timestamp_does_not_discard_valid_neighbor(timestamp):
    malformed = _event()
    malformed['commence_time'] = timestamp
    out = parse_events([malformed, _event()])
    assert len(out) == 1


@pytest.mark.parametrize('broken', [None, 3, {}, {'bookmakers': None},
    {'home_team': [], 'away_team': 'Away'}, {'commence_time': []}])
def test_malformed_event_preserves_neighbor(broken):
    out = parse_events([broken, _event()])
    assert len(out) == 1
    assert str(out.kickoff.dt.tz) == 'UTC'


def test_malformed_nested_bookmakers_preserve_fallback():
    event = _event()
    event['bookmakers'] = [None, {'key': 'bad', 'markets': [None, {'key': 'h2h', 'outcomes': [None]}]}, _book('valid')]
    assert parse_events([event]).iloc[0]['books'] == 1


@pytest.mark.parametrize('status', [401, 429, 500])
def test_fetch_http_failures_at_request_boundary(monkeypatch, status):
    import requests
    from src.live import fetch_events
    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr('src.live.requests.get', lambda *a, **kw: response)
    with pytest.raises(requests.HTTPError):
        fetch_events('test-key')


def test_fetch_rejects_bad_payload(monkeypatch):
    from src.live import fetch_events
    class Response:
        headers = {}
        def raise_for_status(self): pass
        def json(self): return {'message': 'not events'}
    monkeypatch.setattr('src.live.requests.get', lambda *a, **kw: Response())
    with pytest.raises(ValueError, match='invalid events'):
        fetch_events('test-key')


def test_fetch_timeout_at_request_boundary(monkeypatch):
    import requests
    from src.live import fetch_events
    def timeout(*a, **kw): raise requests.Timeout('timeout')
    monkeypatch.setattr('src.live.requests.get', timeout)
    with pytest.raises(requests.Timeout):
        fetch_events('test-key')


def test_record_selected_league_uses_matching_sport_and_ledger(monkeypatch, tmp_path):
    from src.ledger import record, load_ledger
    seen = []
    def fetch(key, sport):
        seen.append(sport)
        event = _event()
        event['commence_time'] = '2099-08-15T14:00:00Z'
        return [event], {}
    monkeypatch.setattr('src.live.fetch_events', fetch)
    path = tmp_path / 'ledger.parquet'
    record('test-key', path=path, league='F1')
    assert seen == ['soccer_france_ligue_one']
    assert load_ledger(path).league.tolist() == ['F1']
