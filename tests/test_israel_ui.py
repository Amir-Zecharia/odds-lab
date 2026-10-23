"""Both live surfaces expose opt-in Israel without startup network calls."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def offline(monkeypatch, tmp_path):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.delenv('API_FOOTBALL_KEY', raising=False)
    monkeypatch.setattr('src.config.PROCESSED_DIR', tmp_path/'processed')
    monkeypatch.setattr('requests.get', lambda *a, **k: pytest.fail('UI must remain offline'))


def test_upcoming_israel_selection_and_opt_in_without_key_are_offline(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    offline(monkeypatch, tmp_path)
    app = AppTest.from_file(str(ROOT/'app/pages/15_Upcoming_Matches.py'), default_timeout=30).run()
    assert not app.exception
    selector = next((widget for widget in list(app.selectbox) + list(app.radio) if any(
        'israel' in str(option).lower() or 'ligat' in str(option).lower() for option in widget.options)), None)
    assert selector is not None, 'Upcoming Matches must offer named Israeli Premier League'
    selector.set_value('IL1').run()
    assert not app.exception
    live = next(widget for widget in app.checkbox if 'live' in widget.label.lower())
    live.check().run()
    assert not app.exception
    assert any('key' in str(item.value).lower() for item in list(app.info) + list(app.warning))


def test_collection_includes_named_israel_and_separate_private_provider_controls(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    offline(monkeypatch, tmp_path)
    pages = list((ROOT/'app/pages').glob('*Live_Collection.py'))
    assert len(pages) == 1
    app = AppTest.from_file(str(pages[0]), default_timeout=30).run()
    assert not app.exception
    selector = next((widget for widget in app.multiselect if any(
        'israel' in str(option).lower() or 'ligat' in str(option).lower() for option in widget.options)), None)
    assert selector is not None
    assert 'IL1' not in selector.value, 'Israel is explicit opt-in'
    selector.select('IL1').run()
    assert not app.exception
    password_labels = [widget.label.lower() for widget in app.text_input if widget.proto.type == widget.proto.PASSWORD]
    assert any('football' in label for label in password_labels)
    labels = [button.label.lower() for button in app.button]
    assert any('clear' in label and 'football' in label for label in labels)
    rendered = ' '.join(str(item.value) for item in list(app.caption) + list(app.info) + list(app.markdown))
    assert 'unverified' in rendered.lower()


@pytest.mark.parametrize('league', ['E0', 'IL1'])
def test_upcoming_cached_odds_drop_started_fixtures_on_each_render(monkeypatch, tmp_path, league):
    import datetime as dt
    import pandas as pd
    from streamlit.testing.v1 import AppTest
    offline(monkeypatch, tmp_path)
    real_datetime = dt.datetime
    clock = [real_datetime.now(dt.timezone.utc)]

    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return pd.Timestamp(clock[0]).tz_convert(tz) if tz else pd.Timestamp(clock[0]).tz_localize(None)

    def event(name, minutes):
        return {'id': name, 'home_team': name, 'away_team': 'Away',
                'commence_time': (clock[0] + dt.timedelta(minutes=minutes)).isoformat(),
                'bookmakers': [{'key': 'pinnacle', 'markets': [{'key': 'h2h', 'outcomes': [
                    {'name': name, 'price': 2.0}, {'name': 'Draw', 'price': 3.0},
                    {'name': 'Away', 'price': 4.0}]}]}]}

    payload = [event('Expired', -1), event('Soon', 5), event('Later', 90)]
    calls = []

    def fetch(*args):
        calls.append(True)
        return payload, {'remaining': 100}

    monkeypatch.setattr('src.live.fetch_events', fetch)
    monkeypatch.setattr('src.api_football.ApiFootballClient.fetch_events', fetch)
    monkeypatch.setenv('API_FOOTBALL_KEY' if league == 'IL1' else 'ODDS_API_KEY', 'test-only-key')
    # AppTest does not register sibling multipage navigation metadata.
    monkeypatch.setattr('streamlit.page_link', lambda *args, **kwargs: None)
    monkeypatch.setattr(dt, 'datetime', Clock)
    app = AppTest.from_file(str(ROOT/'app/pages/15_Upcoming_Matches.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value(league).run()
    next(widget for widget in app.checkbox if 'live odds' in widget.label.lower()).check().run()
    assert not app.exception
    assert list(app.dataframe[0].value['home']) == ['Soon', 'Later']
    assert len(calls) == 1
    clock[0] += dt.timedelta(minutes=10)
    app.run()
    assert not app.exception
    assert list(app.dataframe[0].value['home']) == ['Later']
    assert len(calls) == 1, 'Rerender must filter the cached response without a provider request'
