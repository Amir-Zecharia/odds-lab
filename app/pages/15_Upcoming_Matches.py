import os
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from common import page_setup
from src.live import SPORT_KEYS, fetch_events, parse_events
from src import config

page_setup("Upcoming Matches", icon="🔮")
st.caption(
    "Live probabilities for upcoming fixtures, using the recipe this "
    "platform validated: Shin de-margin every book, prefer Pinnacle, "
    "fall back to the cross-book average."
)

league = st.sidebar.radio("League", list(SPORT_KEYS), format_func=config.league_display_name)
league_name = config.league_display_name(league)
sport_key = SPORT_KEYS[league]

api_key = os.environ.get("ODDS_API_KEY", "")
if not api_key:
    try:
        api_key = st.secrets.get("ODDS_API_KEY", "")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        api_key = ""
if not api_key:
    api_key = st.text_input(
        "The Odds API key",
        type="password",
        help="Use your own key from the-odds-api.com. "
        "Set ODDS_API_KEY in your environment to skip this box.",
    )
if not api_key:
    st.info(
        "Get a free key at [the-odds-api.com](https://the-odds-api.com/) "
        "paste it above, use Streamlit secrets, or run the app with "
        "`ODDS_API_KEY=<key> streamlit run app/Home.py`."
    )
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Fetching live odds ...")
def cached_fetch(key: str, sport: str):
    events, quota = fetch_events(key, sport)
    return events, quota, datetime.now(timezone.utc)


if not st.checkbox("Enable optional live odds", value=False):
    st.info("Historical research works offline. Enable live odds to contact The Odds API with your key.")
    st.stop()
if st.button("Refresh live odds", help="Requests fresh odds and consumes your API quota."):
    cached_fetch.clear(api_key, sport_key)
try:
    events, quota, fetched_at = cached_fetch(api_key, sport_key)
except requests.HTTPError as exc:
    status = exc.response.status_code if exc.response is not None else "?"
    if status == 401:
        st.error("The API rejected this key (401). Check it and try again.")
    elif status == 429:
        st.error("Monthly request quota exhausted (429).")
    else:
        st.error(f"The Odds API returned an error (HTTP {status}).")
    st.stop()
except ValueError:
    st.error("The Odds API returned an unreadable response. Try refreshing later.")
    st.stop()
except requests.RequestException:
    st.error("Could not reach The Odds API — check your connection.")
    st.stop()

table = parse_events(events)
if table.empty:
    st.warning(
        f"No upcoming {league_name} fixtures with odds right now "
        "(off-season or between rounds). Odds usually appear a few days "
        "before kickoff."
    )
    st.stop()

st.markdown(
    f"**{len(table)} upcoming fixtures.** Odds cached for an hour; "
    f"Cache age: {max(0, int((datetime.now(timezone.utc) - fetched_at).total_seconds() // 60))} minutes. "
    f"API quota remaining: {quota.get('remaining', '?')}."
)

show = table.copy()
show["kickoff"] = show["kickoff"].dt.strftime("%a %d %b %H:%M")
show = show.rename(
    columns={
        "kickoff": "kickoff (UTC)", "home_team": "home", "away_team": "away",
        "p_home": "P(home)", "p_draw": "P(draw)", "p_away": "P(away)",
    }
)
st.dataframe(
    show.set_index("kickoff (UTC)").style.format(
        {"P(home)": "{:.1%}", "P(draw)": "{:.1%}", "P(away)": "{:.1%}"}
    ),
    use_container_width=True,
    height=min(80 + 36 * len(show), 700),
)

st.caption(
    "These are the market's probabilities, de-margined — the best publicly "
    "available forecast, and (as every other page here shows) extremely "
    "hard to improve on. They are not betting advice: the margin means the "
    "expected value of betting them is negative."
)

st.subheader("Record these predictions")
st.markdown(
    "Commit these forecasts to the **Live Ledger** before kickoff. Once a "
    "match starts its row is frozen, so what gets scored later is exactly "
    "what was predicted in advance."
)
if st.button("📓 Record to ledger", type="primary"):
    from src.ledger import LEDGER_PARQUET, append_predictions, load_ledger

    ledger, stats = append_predictions(load_ledger(), table, league=league)
    LEDGER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(LEDGER_PARQUET, index=False)
    st.success(
        f"Recorded: **{stats['added']} new**, {stats['updated']} updated, "
        f"{stats['frozen']} already frozen (kicked off). "
        f"Ledger now holds {len(ledger)} predictions — see the Live Ledger page."
    )
