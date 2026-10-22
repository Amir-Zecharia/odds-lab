import streamlit as st

from common import explain, league_filter, load_analysis, load_matches, page_setup

page_setup("Odds Lab", icon="🎯")

st.markdown(
    "**The betting market under a microscope.** Six leagues, 26 seasons, "
    "61,000 matches, 16 bookmakers — how well do the odds predict football, "
    "where do they crack, and who is copying whom?"
)

full = load_matches()
dates = full["date"]
st.info(
    f"Working historical data: {dates.min():%d %b %Y} – {dates.max():%d %b %Y}. "
    "The bundled baseline ends May 2026; working coverage above reflects any local refresh. "
    "All research pages work offline. Live odds are optional and require your own API key."
)
for path, label in [
    ("pages/1_Data_Explorer.py", "Explore historical data"),
    ("pages/28_Match_Forecast.py", "Replay a match forecast"),
    ("pages/26_Season_Simulator.py", "Replay a season simulation"),
    ("pages/15_Upcoming_Matches.py", "Connect optional live odds"),
    ("pages/29_Live_Collection.py", "Configure scheduled live collection and learning"),
]:
    st.page_link(path, label=label)
df = league_filter(full)
analysis = load_analysis()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches", f"{len(df):,}", help=f"{len(full):,} across all leagues")
c2.metric("Leagues", full["league"].nunique() if "league" in full else 1)
c3.metric("Seasons", df["season"].nunique())
eff = analysis["efficiency"]
c4.metric(
    "Calibration slope",
    f"{eff['slope']:.3f}",
    help="1.000 = when the market says 60%, it happens exactly 60% of the "
    "time. Computed for the league selected in the sidebar.",
)

st.subheader("Start here — three steps")
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        "**1 · Meet the data** 🗺️\n\nOpen **Data Explorer** for base rates, "
        "bookmaker margins and the coverage matrix that bounds every claim "
        "on this platform."
    )
with s2:
    st.markdown(
        "**2 · See the market's honesty** 🎚️\n\n**Calibration** shows the "
        "signature chart: implied probability vs what actually happened, "
        "hugging the diagonal for two decades."
    )
with s3:
    st.markdown(
        "**3 · Hunt the cracks** 🔬\n\n**Mismatch Explorer** lets you dredge "
        "61k matches for mispriced pockets — with the statistics that keep "
        "you honest about what you find."
    )

st.subheader("What we've established so far")
st.markdown(
    """
| Finding | Where to see it |
| --- | --- |
| The market is calibrated (slope ≈ 1.02, all leagues pooled) | Calibration |
| Nothing beats the closing odds — not Elo, form, ML, or blends | Beat the Market, Research Log |
| Pinnacle closing is the single sharpest price | Bookmaker Leaderboard, Consensus Builder |
| Optimal bookmaker weights ≈ any weights (books agree at r > 0.98) | Consensus Builder |
| Predicting early costs +0.003 nats; the line move is real information | Line Movement |
| Big odds moves do **not** flag upset-prone matches | Drift vs Surprises |
| A deliberate mispricing dredge found 0 confirmed pockets | Mismatch Explorer |
| Soft books copy Pinnacle (herding ≈ 0.7, one-way leadership z ≈ +24) | Bookmaker Influence |
| Every betting rule loses ≈ the margin (ROI ≈ −3%) | Betting Simulator |
"""
)

st.subheader("All pages, by what you want to do")

st.markdown("**🎯 Make a prediction**")
st.markdown(
    """
| Page | Question |
| --- | --- |
| **Match Forecast** ⭐ | One fixture, every prediction the platform can make |
| **Season Simulator** | Who wins the league? Monte Carlo the run-in, thousands of times |
| **Upcoming Matches** | Live de-margined probabilities for the next fixtures |
| **Live Ledger** | Forward testing: predictions recorded before kickoff, scored after |
| **Pricing Engine** | One grid, every market — how a desk originates prices |
"""
)

st.markdown("**🔍 Explore the data**")
st.markdown(
    """
| Page | Question |
| --- | --- |
| **Data Explorer** | What does two decades of odds data look like? |
| **Descriptive Statistics** | Slice the data with your own filters |
| **Half Time** | The second outcome in every match: leads, comebacks, two different games |
| **Corners, Cards & Referees** | The markets no closing line disciplines — and what referees do |
| **Structural Patterns** | Home advantage, draws, and an odds-free model |
"""
)

st.markdown("**⚖️ Is the market honest?**")
st.markdown(
    """
| Page | Question |
| --- | --- |
| **Calibration** | Are market probabilities honest? (the signature chart) |
| **Mismatch Explorer** | Where do odds and reality actually disagree? |
| **Asian Handicap** | Is the cleanest no-draw market priced honestly? |
| **Totals Market** | Is over/under 2.5 goals priced as sharply as 1X2? |
| **Round by Round** | Does accuracy vary across the season? |
"""
)

st.markdown("**🥊 Can anything beat it?**")
st.markdown(
    """
| Page | Question |
| --- | --- |
| **Beat the Market** | Do models or extra features add anything? |
| **Model Stack** | Five rating systems + a meta-model: the best odds-free predictor |
| **Club Elo** | Does an external results-only rating add anything? |
| **Consensus Builder** | How should bookmakers be weighted in a prediction? |
| **Betting Simulator** | Backtest any rule and watch the margin grind |
"""
)

st.markdown("**🏦 How the market works**")
st.markdown(
    """
| Page | Question |
| --- | --- |
| **Bookmaker Leaderboard** | Which book prices matches sharpest? |
| **Bookmaker Influence** | Who leads, who follows, how strong is the pack? |
| **Bookmaker Analysis** | One book under the microscope |
| **Line Movement** | What does the early→closing drift know? |
| **Odds Gaps** | What happens when bookmakers significantly disagree? |
| **Disagreement & Error** | Does bookmaker disagreement signal uncertainty? |
| **Drift vs Surprises** | Do big moves predict results defying the consensus? |
| **Research Log** | Every study run here, with its verdict |
"""
)

explain(
    "Glossary — the six ideas everything here builds on",
    """
- **Decimal odds → implied probability**: 1/odds. Odds of 2.50 = 40%.
- **Overround (margin)**: implied probabilities sum past 100%; the excess
  is the bookmaker's built-in fee (2-12% depending on book and era).
- **Shin de-margining**: removing the margin knowing bookmakers hide more
  of it in longshots than favorites; plain division would distort.
- **Closing line**: the last price before kickoff — the market's final,
  best-informed estimate. Everything sharpens toward it.
- **Log loss**: the grade for probability forecasts; heavily punishes
  confident misses. Lower = sharper. ln(3) ≈ 1.099 = knowing nothing.
- **FDR (false discovery rate)**: when scanning many segments, the
  correction that keeps lucky noise from being reported as findings.
"""
)

st.caption(
    f"Data: football-data.co.uk · Analyses last computed "
    f"{analysis.get('computed_at', '?')[:19]} · Source data terms: see DATA_SOURCES.md · not betting advice"
)
