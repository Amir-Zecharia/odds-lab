import os
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from common import page_setup
from src.live import SPORT_KEYS, fetch_events, parse_events
from src.collector import load_settings
from src.api_football import ApiFootballClient, ApiFootballError
from src.live_leagues import LIVE_LEAGUES, ISRAEL_LEAGUE, live_league_name

page_setup("Upcoming Matches", icon="🔮")
st.caption(
    "Live probabilities for upcoming fixtures, using the recipe this "
    "platform validated: Shin de-margin every book, prefer Pinnacle, "
    "fall back to the cross-book average."
)

league = st.sidebar.radio("League", list(LIVE_LEAGUES), format_func=live_league_name)
league_name = live_league_name(league)
is_israel = league == ISRAEL_LEAGUE
provider_name = "API-Football" if is_israel else "The Odds API"
provider_code = LIVE_LEAGUES[league]["provider"]
sport_key = SPORT_KEYS.get(league, league)
env_name = "API_FOOTBALL_KEY" if is_israel else "ODDS_API_KEY"
settings_key = "api_football_key" if is_israel else "api_key"
api_key = os.environ.get(env_name, "")
if not api_key:
    try:
        api_key = st.secrets.get(env_name, "")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        api_key = ""
if not api_key:
    api_key = load_settings().get(settings_key, "")
if not api_key:
    api_key = st.text_input(f"{provider_name} key", type="password",
                            help=f"Set {env_name} or save a private key on Live Collection.")
if is_israel:
    st.info("Israel current-season odds coverage and plan access remain unverified until you authenticate. "
            "There is no Israeli historical bundle. API-Football uses a separate direct-provider key and quota.")
live_enabled = st.checkbox("Enable optional live odds", value=False)
if not api_key:
    st.info(f"Supply your own {provider_name} key above, via {env_name}, or on Live Collection.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner="Fetching live odds ...")
def cached_fetch(key: str, sport: str, provider: str):
    events, quota = (ApiFootballClient(key).fetch_events() if provider == "api_football"
                     else fetch_events(key, sport))
    return events, quota, datetime.now(timezone.utc)


if not live_enabled:
    st.info("Historical research works offline. Enable live odds to contact the selected provider with your key.")
    st.stop()
if st.button("Refresh live odds", help="Requests fresh odds and consumes your API quota."):
    cached_fetch.clear(api_key, sport_key, provider_code)
try:
    events, quota, fetched_at = cached_fetch(api_key, sport_key, provider_code)
except ApiFootballError as exc:
    st.error(f"API-Football could not provide current Israel odds ({exc.reason}). Coverage or plan access may be unavailable; try again later.")
    st.stop()
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
# A cached response can outlive kickoff; recheck eligibility on every render.
if not table.empty:
    table = table.loc[table["kickoff"] > datetime.now(timezone.utc)].copy()
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

if is_israel:
    st.subheader("Collect and freeze Israel forecasts")
    st.info("Use Live Collection with IL1 selected to save provider event identities, frozen forecasts and live finals. "
            "Israel is not supported by the historical Live Ledger settlement workflow.")
    st.page_link("pages/29_Live_Collection.py", label="Open Live Collection")
    st.stop()

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
