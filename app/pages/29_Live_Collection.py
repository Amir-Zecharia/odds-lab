"""Read-only startup; paid requests require an explicit collection action."""
import pandas as pd
import streamlit as st
from common import page_setup
from src.collector import load_settings, save_settings, public_status, run_collection
from src.live import SPORT_KEYS
from src.live_store import LiveStore

page_setup('Live Collection', icon='🕒')
st.write('Collect live odds and final scores at 00:00, 08:00 and 16:00 in this Mac’s local timezone. The Mac must be available. Each run checks the last three days of scores; older missing results remain unresolved.')
settings = load_settings()
status = public_status()
st.metric('Private API key', 'Configured' if status['key_configured'] else 'Awaiting key')
st.write('Scheduling enabled' if settings['enabled'] else 'Scheduling disabled')
st.write('Next local slot:', status['next_schedule'])
st.caption('Install the local scheduler once with `python -m src.collector schedule install`. Collection continues with the dashboard closed. Saving enabled controls an installed scheduler; it does not install one.')
with st.form('collection_settings'):
    selected = st.multiselect('Leagues', list(SPORT_KEYS), default=settings['leagues'])
    enabled = st.checkbox('Enable scheduled collection', value=settings['enabled'])
    key = st.text_input('Private Odds API key (blank preserves existing key)', type='password', value='')
    save = st.form_submit_button('Save private settings')
    if save:
        save_settings({'leagues': selected, 'enabled': enabled, 'api_key': key})
        st.success('Private settings saved. Environment key overrides the stored key.')
if st.button('Clear stored key'):
    save_settings({'clear_key': True})
    st.success('Stored key cleared; an environment override may still be configured.')
st.caption(f'Estimated usage: {len(selected)*9} credits/day; {len(selected)*270} per 30 days. Free plan: 500 credits/month. Six leagues cost approximately 54/day or 1620/month; empty responses may cost less. No plan purchases are made.')
if st.button('Run current scheduled slot now'):
    with st.spinner('Collecting scores, fitting candidate and collecting odds…'):
        st.json(run_collection())
st.subheader('Collection status')
st.json(status['collection'])
store = LiveStore()
try:
    runs = store.runs_frame()
    if not runs.empty:
        st.dataframe(runs, hide_index=True)
    model = store.latest_model()
    st.subheader('Learning and promotion')
    if model:
        st.json({'version': model.get('version'), 'trained_at': model.get('trained_at'), 'validation': model.get('validation')})
    else:
        st.info('No model fitted yet. Active forecasts use the market baseline until the candidate passes the chronological validation gate.')
    snapshots = store.snapshot_frame()
    comparison = store.comparison_frame()
    results = store.results_frame()
    if snapshots.empty:
        st.info('No live snapshots collected. Configure a key and run a slot, or wait for an enabled installed scheduler.')
    else:
        st.subheader('One last eligible forecast per completed match')
        if not comparison.empty:
            eligible = comparison.dropna(subset=['log_loss', 'brier']).copy()
            if not eligible.empty:
                eligible['captured_at'] = pd.to_datetime(eligible['captured_at'], utc=True)
                latest = eligible.sort_values('captured_at').groupby(['league', 'event_id', 'kind'], as_index=False).tail(1)
                st.dataframe(latest.groupby('kind')[['log_loss', 'brier']].mean(), use_container_width=True)
        identities = snapshots[['league', 'event_id', 'home_team', 'away_team']].drop_duplicates(['league', 'event_id'])
        labels = {f"{r.league}: {r.home_team} – {r.away_team} ({r.event_id})": (r.league, r.event_id) for r in identities.itertuples()}
        selected_event = st.selectbox('Match history', list(labels))
        league, event_id = labels[selected_event]
        for title, frame in [('Odds snapshots', snapshots), ('Frozen baseline, candidate and active forecasts / losses', comparison), ('Final observations and corrections', results)]:
            st.write(title)
            if not frame.empty:
                st.dataframe(frame[(frame['league'] == league) & (frame['event_id'] == event_id)], hide_index=True)
finally:
    store.close()
