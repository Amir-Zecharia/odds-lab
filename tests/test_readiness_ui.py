"""Runtime checks for optional live startup and historical result caching."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_upcoming_startup_never_requests_live_odds(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('ODDS_API_KEY', 'unit-test-key')
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Startup must not fetch')
    monkeypatch.setattr('src.live.requests.get', forbidden)
    app = AppTest.from_file(str(ROOT / 'app/pages/15_Upcoming_Matches.py')).run()
    assert not app.exception
    assert not calls
    assert app.checkbox[0].label == 'Enable optional live odds'


def test_simulator_cache_recomputes_after_dataset_timestamp_changes(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    from src import config, models, season_sim
    monkeypatch.syspath_prepend(str(ROOT / 'app'))
    import common

    st.cache_data.clear()
    dataset = tmp_path / 'enriched.parquet'
    dataset.touch()
    monkeypatch.setattr(config, 'PROCESSED_DIR', tmp_path)
    # A small current season and enough prior matches to reach the real cache.
    past = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=500), 'league': 'E0',
        'season': '2023-24', 'home_team': 'Home', 'away_team': 'Away',
        'home_goals': 2, 'away_goals': 1,
    })
    current = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=4), 'league': 'E0',
        'season': '2024-25', 'home_team': 'Home', 'away_team': 'Away',
        'home_goals': 2, 'away_goals': 1,
    })
    data = pd.concat([past, current], ignore_index=True)
    monkeypatch.setattr(common, 'load_matches', lambda: data.copy())
    monkeypatch.setattr(common, 'page_setup', lambda *a, **kw: None)
    monkeypatch.setattr(common, 'league_selector', lambda: 'E0')
    fits = []
    class Model:
        def fit(self, history):
            fits.append(len(history))
            return self
        def outcome_probs(self, home, away):
            return [0.5, 0.25, 0.25]
    monkeypatch.setattr(models, 'DixonColes', Model)
    # Keep the real page/cache and summary rendering; avoid Monte Carlo cost.
    monkeypatch.setattr(season_sim, 'simulate_season',
                        lambda *a, **kw: [season_sim.league_table(current)])
    try:
        app = AppTest.from_file(str(ROOT / 'app/pages/26_Season_Simulator.py')).run()
        assert not app.exception
        assert fits == [500]
        app.run()
        assert not app.exception
        assert fits == [500], 'unchanged file must reuse the fitted result'
        before = dataset.stat()
        os.utime(dataset, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        app.run()
        assert not app.exception
        assert fits == [500, 500], 'changed dataset must invalidate the model result'
    finally:
        st.cache_data.clear()
