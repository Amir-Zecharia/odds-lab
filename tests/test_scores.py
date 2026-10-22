import copy
import pandas as pd
import pytest


def final(event_id='provider-1', home='2', away='1'):
    return dict(id=event_id, completed=True, commence_time='2026-10-20T16:00:00Z',
                home_team='Home', away_team='Away', last_update='2026-10-20T18:00:00Z',
                scores=[dict(name='Away', score=away), dict(name='Home', score=home)])


def test_scores_join_by_named_team_and_keep_provider_identity():
    from src.live import parse_completed_scores
    row = parse_completed_scores([final()]).iloc[0]
    assert (row.event_id, row.home_goals, row.away_goals, row.result) == ('provider-1', 2, 1, 'H')
    assert row.kickoff == pd.Timestamp('2026-10-20T16:00:00Z')
    assert row.provider_updated_at == pd.Timestamp('2026-10-20T18:00:00Z')


@pytest.mark.parametrize('change', [dict(completed=False), dict(id=None), dict(scores=None),
    dict(scores=[dict(name='Home', score='-1'), dict(name='Away', score='0')]),
    dict(scores=[dict(name='Home', score='1.5'), dict(name='Away', score='0')]),
    dict(scores=[dict(name='Home', score=True), dict(name='Away', score='0')])])
def test_invalid_final_preserves_valid_neighbor(change):
    from src.live import parse_completed_scores
    bad = final(); bad.update(change)
    out = parse_completed_scores([bad, final('good')])
    assert out.event_id.tolist() == ['good']


def test_fetch_scores_endpoint_quota_and_bounded_timeout(monkeypatch):
    from src.live import fetch_scores
    calls = []
    class Response:
        headers = {'x-requests-remaining': '12', 'x-requests-used': '4'}
        def raise_for_status(self): pass
        def json(self): return [final()]
    def get(url, **kwargs):
        calls.append((url, kwargs)); return Response()
    monkeypatch.setattr('src.live.requests.get', get)
    events, quota = fetch_scores('private', sport='soccer_epl')
    assert events[0]['id'] == 'provider-1'
    url, options = calls[0]
    assert url.endswith('/sports/soccer_epl/scores')
    assert options['params']['daysFrom'] == 3
    assert 0 < options['timeout'] <= 60
    assert quota


@pytest.mark.parametrize('days', [0, 4])
def test_scores_lookback_rejected_before_request(monkeypatch, days):
    from src.live import fetch_scores
    monkeypatch.setattr('src.live.requests.get', lambda *a, **k: pytest.fail('unexpected HTTP'))
    with pytest.raises(ValueError): fetch_scores('private', days_from=days)
