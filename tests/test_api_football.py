"""Direct-provider acceptance tests. Every provider response is synthetic."""
import copy
import json

import pytest

NOW = '2026-10-21T10:00:00Z'
SECRET = 'synthetic-api-football-secret'


def envelope(rows, page=1, total=1, errors=None):
    return {'errors': errors or [], 'results': len(rows),
            'paging': {'current': page, 'total': total}, 'response': rows}


def league(coverage=True):
    return {'league': {'id': 9123, 'name': "Ligat Ha'al", 'type': 'League'},
            'country': {'name': 'Israel'},
            'seasons': [{'year': 2026, 'current': True, 'coverage': {'odds': coverage}}]}


def fixture(identity=42, status='NS', date='2026-10-21T18:00:00+00:00'):
    return {'fixture': {'id': identity, 'date': date, 'status': {'short': status}},
            'league': {'id': 9123, 'season': 2026},
            'teams': {'home': {'id': 100, 'name': 'Home'}, 'away': {'id': 101, 'name': 'Away'}},
            'goals': {'home': 8, 'away': 7},
            'score': {'fulltime': {'home': 2, 'away': 1}}}


def odds(identity=42):
    return {'league': {'id': 9123, 'season': 2026}, 'fixture': {'id': identity},
            'bookmakers': [{'id': 4, 'name': 'Pinnacle', 'bets': [
                {'id': 1, 'name': 'Match Winner', 'values': [
                    {'value': 'Home', 'odd': '2.0'}, {'value': 'Draw', 'odd': '3.5'},
                    {'value': 'Away', 'odd': '4.0'}]}]}]}


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code = payload, status
        self.headers = headers or {'x-ratelimit-requests-remaining': '90',
                                   'x-ratelimit-requests-limit': '100'}
    def json(self):
        return copy.deepcopy(self.payload)
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(SECRET, response=self)


def fake_http(monkeypatch, handler):
    calls = []
    def get(url, **kwargs):
        calls.append((url, copy.deepcopy(kwargs)))
        return handler(url.rsplit('/', 1)[-1], kwargs)
    monkeypatch.setattr('requests.get', get)
    return calls


def standard_http(monkeypatch, fixtures=None, odds_rows=None):
    def handle(endpoint, options):
        return Response(envelope({'leagues': [league()], 'fixtures': fixtures if fixtures is not None else [fixture()],
                                  'odds': odds_rows if odds_rows is not None else [odds()]}[endpoint]))
    return fake_http(monkeypatch, handle)


def test_registry_keeps_israel_out_of_historical_and_odds_api_leagues():
    from src.live_leagues import ISRAEL_LEAGUE, LIVE_LEAGUES, live_league_name
    from src.live import SPORT_KEYS
    from src.config import LEAGUES
    assert ISRAEL_LEAGUE == 'IL1'
    assert set(LIVE_LEAGUES) == set(SPORT_KEYS) | {'IL1'}
    assert LIVE_LEAGUES['IL1']['provider'] == 'api_football'
    assert all(LIVE_LEAGUES[code]['provider'] == 'odds_api' for code in SPORT_KEYS)
    assert live_league_name('IL1') == LIVE_LEAGUES['IL1']['name']
    assert 'IL1' not in LEAGUES and 'IL1' not in SPORT_KEYS


def test_lazy_discovery_header_auth_utc_window_and_instance_cache(monkeypatch):
    from src.api_football import ApiFootballClient
    calls = standard_http(monkeypatch)
    client = ApiFootballClient(SECRET, now=lambda: NOW)
    assert calls == []
    client.fetch_scores()
    events, quota = client.fetch_events()
    assert events[0]['id'] == 'api-football:42'
    assert str(quota['remaining']) == '90'
    assert sum(url.endswith('/leagues') for url, _ in calls) == 1
    assert calls[0][1]['params'] == {'country': 'Israel', 'current': 'true', 'type': 'League'}
    for url, options in calls:
        assert url.startswith('https://v3.football.api-sports.io/')
        assert options['headers']['x-apisports-key'] == SECRET
        assert SECRET not in url and SECRET not in json.dumps(options.get('params', {}))
        assert options['timeout'] == 30 and options['allow_redirects'] is False
    params = next(options['params'] for url, options in calls if url.endswith('/fixtures'))
    assert params == {'league': 9123, 'season': 2026, 'from': '2026-10-18', 'to': '2026-11-04', 'timezone': 'UTC'}
    params = next(options['params'] for url, options in calls if url.endswith('/odds'))
    assert params == {'league': 9123, 'season': 2026, 'bet': 1, 'page': 1}


@pytest.mark.parametrize('catalog', [[league(False)], [league(), league()],
                                    [dict(league(), country={'name': 'England'})]])
def test_discovery_fails_closed_without_verified_unique_current_coverage(monkeypatch, catalog):
    from src.api_football import ApiFootballClient, ApiFootballError
    calls = fake_http(monkeypatch, lambda endpoint, options: Response(envelope(catalog)))
    with pytest.raises(ApiFootballError) as caught:
        ApiFootballClient(SECRET, now=NOW).fetch_events()
    assert caught.value.reason
    assert len(calls) == 1
    assert SECRET not in repr(caught.value)


def test_scores_accept_only_regulation_finals_and_skip_bad_neighbors(monkeypatch):
    from src.api_football import ApiFootballClient
    from src.live import parse_completed_scores
    rows = [fixture(42, 'FT'), fixture(43, 'AET'), fixture(44, 'PEN'), fixture(45, 'NS')]
    broken = fixture(46, 'FT'); broken['score']['fulltime']['home'] = True
    wrong_league = fixture(47, 'FT'); wrong_league['league']['id'] = 1
    same_team = fixture(48, 'FT'); same_team['teams']['away']['id'] = 100
    standard_http(monkeypatch, rows + [broken, wrong_league, same_team])
    events, _ = ApiFootballClient(SECRET, now=NOW).fetch_scores()
    assert [event['id'] for event in events] == ['api-football:42']
    assert events[0].get('last_update') is None
    parsed = parse_completed_scores(events)
    assert (parsed.iloc[0].home_goals, parsed.iloc[0].away_goals) == (2, 1)
    assert parsed.iloc[0].home_team == 'Home'


def test_odds_only_complete_match_winner_known_fixture_and_valid_prices(monkeypatch):
    from src.api_football import ApiFootballClient
    from src.live import parse_events
    good = odds()
    unsupported = copy.deepcopy(good['bookmakers'][0]['bets'][0]); unsupported['id'] = 5
    good['bookmakers'][0]['bets'].insert(0, unsupported)
    malformed = odds(43); malformed['bookmakers'][0]['bets'][0]['values'][0]['odd'] = True
    incomplete = odds(44); incomplete['bookmakers'][0]['bets'][0]['values'].pop()
    nonfinite = odds(45); nonfinite['bookmakers'][0]['bets'][0]['values'][0]['odd'] = 'NaN'
    standard_http(monkeypatch, [fixture(i) for i in range(42, 46)], [good, malformed, incomplete, nonfinite, odds(999)])
    events, _ = ApiFootballClient(SECRET, now=NOW).fetch_events()
    parsed = parse_events(events)
    assert parsed.event_id.tolist() == ['api-football:42']
    assert parsed.iloc[0]['source'] == 'Pinnacle'
    assert events[0]['bookmakers'][0]['key'] == 'pinnacle'


def test_pagination_joins_all_pages_and_rejects_partial_second_page(monkeypatch):
    from src.api_football import ApiFootballClient, ApiFootballError
    broken = [False]
    def handler(endpoint, options):
        if endpoint == 'leagues': return Response(envelope([league()]))
        if endpoint == 'fixtures': return Response(envelope([fixture(42), fixture(43)]))
        page = options['params']['page']
        if broken[0] and page == 2: return Response(envelope([], page=1, total=2))
        return Response(envelope([odds(41 + page)], page=page, total=2))
    fake_http(monkeypatch, handler)
    assert len(ApiFootballClient(SECRET, now=NOW).fetch_events()[0]) == 2
    broken[0] = True
    with pytest.raises(ApiFootballError): ApiFootballClient(SECRET, now=NOW).fetch_events()


def test_exhausted_quota_prevents_subsequent_request(monkeypatch):
    from src.api_football import ApiFootballClient, ApiFootballError
    calls = fake_http(monkeypatch, lambda endpoint, options: Response(envelope([league()]), headers={
        'x-ratelimit-requests-remaining': '0', 'x-ratelimit-requests-limit': '100'}))
    with pytest.raises(ApiFootballError) as caught:
        ApiFootballClient(SECRET, now=NOW).fetch_events()
    assert caught.value.reason == 'quota'
    assert len(calls) == 1


@pytest.mark.parametrize('response', [Response(envelope([], errors={'token': SECRET})),
                                    Response([], 200), Response(envelope([]), 302)])
def test_envelope_and_redirect_failures_are_safe(monkeypatch, response):
    from src.api_football import ApiFootballClient, ApiFootballError
    calls = fake_http(monkeypatch, lambda endpoint, options: response)
    with pytest.raises(ApiFootballError) as caught:
        ApiFootballClient(SECRET, now=NOW).fetch_scores()
    assert len(calls) == 1
    assert SECRET not in str(caught.value) + repr(vars(caught.value))


@pytest.mark.parametrize('paging', [None, {'current': True, 'total': 1},
                                    {'current': 1, 'total': 21}, {'current': 1, 'total': 0}])
def test_invalid_or_unbounded_paging_rejected(monkeypatch, paging):
    from src.api_football import ApiFootballClient, ApiFootballError
    payload = envelope([league()]); payload['paging'] = paging
    fake_http(monkeypatch, lambda endpoint, options: Response(payload))
    with pytest.raises(ApiFootballError):
        ApiFootballClient(SECRET, now=NOW).fetch_events()


def test_minute_quota_is_independent_of_daily_remaining(monkeypatch):
    from src.api_football import ApiFootballClient, ApiFootballError
    calls = fake_http(monkeypatch, lambda endpoint, options: Response(envelope([league()]), headers={
        'x-ratelimit-requests-remaining': '90', 'x-ratelimit-remaining': '0'}))
    with pytest.raises(ApiFootballError) as caught:
        ApiFootballClient(SECRET, now=NOW).fetch_events()
    assert caught.value.reason == 'quota' and len(calls) == 1


def test_transport_and_json_exceptions_do_not_retain_secret_context(monkeypatch):
    import requests
    from src.api_football import ApiFootballClient, ApiFootballError
    def fail(*args, **kwargs):
        raise requests.ConnectionError(SECRET)
    monkeypatch.setattr(requests, 'get', fail)
    with pytest.raises(ApiFootballError) as caught:
        ApiFootballClient(SECRET, now=NOW).fetch_events()
    assert caught.value.__context__ is None
    assert SECRET not in str(caught.value)


def test_inconsistent_results_count_rejected(monkeypatch):
    from src.api_football import ApiFootballClient, ApiFootballError
    payload = envelope([league()]); payload['results'] = 2
    fake_http(monkeypatch, lambda endpoint, options: Response(payload))
    with pytest.raises(ApiFootballError):
        ApiFootballClient(SECRET, now=NOW).fetch_events()


def test_fixture_identity_and_bookmaker_namespace_fail_closed(monkeypatch):
    from src.api_football import ApiFootballClient
    duplicate = fixture(43)
    wrong_time = fixture(44, date='2026-10-21T18:00:00+02:00')
    wrong_season = fixture(45); wrong_season['league']['season'] = True
    other = odds(); other['bookmakers'][0].update(id=99, name='Other')
    standard_http(monkeypatch, [fixture(), duplicate, duplicate, wrong_time, wrong_season],
                  [other, odds(43), odds(44), odds(45)])
    events, _ = ApiFootballClient(SECRET, now=NOW).fetch_events()
    assert [event['id'] for event in events] == ['api-football:42']
    assert events[0]['bookmakers'][0]['key'] == 'api-football:99'


def test_events_only_future_scheduled_fixtures_scores_still_accept_finals(monkeypatch):
    from src.api_football import ApiFootballClient
    rows = [fixture(42, 'FT', '2026-10-20T18:00:00Z'),
            fixture(43, '1H'), fixture(44, '2H'), fixture(45, 'CANC'),
            fixture(46, 'PST'), fixture(47, 'TBD'), fixture(48, 'UNKNOWN'),
            fixture(49, 'NS'), fixture(50, 'NS', NOW)]
    standard_http(monkeypatch, rows, [odds(i) for i in range(42, 51)])
    client = ApiFootballClient(SECRET, now=NOW)
    assert [event['id'] for event in client.fetch_scores()[0]] == ['api-football:42']
    assert [event['id'] for event in client.fetch_events()[0]] == ['api-football:49']


def test_cached_fixtures_expire_against_clock_after_odds_request(monkeypatch):
    from src.api_football import ApiFootballClient
    clock = [NOW]
    def handler(endpoint, options):
        if endpoint == 'leagues': return Response(envelope([league()]))
        if endpoint == 'fixtures': return Response(envelope([fixture()]))
        clock[0] = '2026-10-21T18:00:00Z'
        return Response(envelope([odds()]))
    fake_http(monkeypatch, handler)
    client = ApiFootballClient(SECRET, now=lambda: clock[0])
    client.fetch_scores()
    assert client.fetch_events()[0] == []


@pytest.mark.parametrize('bad_status', [None, [], {'short': None}, {'short': []}])
def test_malformed_status_is_rejected_before_fixture_cache(monkeypatch, bad_status):
    from src.api_football import ApiFootballClient
    broken = fixture(43); broken['fixture']['status'] = bad_status
    standard_http(monkeypatch, [broken, fixture()], [odds(43), odds()])
    client = ApiFootballClient(SECRET, now=NOW)
    assert [event['id'] for event in client.fetch_events()[0]] == ['api-football:42']
    assert set(client._fixtures) == {42}
