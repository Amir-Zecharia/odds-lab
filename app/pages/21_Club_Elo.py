import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import explain, load_matches, page_setup, season_filter
from src import config
from src.clubelo import CACHE_DIR, COVERED_LEAGUES, RESULTS_JSON, attach_club_elo

page_setup("Club Elo", icon="🌍")
st.caption(
    "First EXTERNAL information source tested on the platform: clubelo.com "
    "publishes results-only Elo ratings built with no odds input. If any "
    "public rating could add to the market consensus, this is the candidate."
)


@st.cache_data(show_spinner="Attaching cached Club Elo ratings ...")
def _with_clubelo(parquet_mtime: float, cache_marker: float) -> pd.DataFrame:
    # live=False: the dashboard only ever reads the disk cache, never the
    # network. cache_marker (results json mtime) invalidates after a re-run.
    frame, _ = attach_club_elo(load_matches(), live=False)
    return frame


def _load_results() -> dict | None:
    try:
        return json.loads(RESULTS_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return None


results = _load_results()
if results is None:
    st.warning("No study results yet. Run `python -m src.clubelo` first.")
    st.stop()
if not CACHE_DIR.exists():
    st.warning("No cached Club Elo data. Run `python -m src.clubelo` first.")
    st.stop()

marker = RESULTS_JSON.stat().st_mtime
from src.run_analyses import ENRICHED_PARQUET  # noqa: E402

df = season_filter(_with_clubelo(ENRICHED_PARQUET.stat().st_mtime, marker))
league = st.session_state.get("league", config.DEFAULT_LEAGUE)

both = df["elo_home_clubelo"].notna() & df["elo_away_clubelo"].notna()
c1, c2, c3 = st.columns(3)
c1.metric("Matches in selection", f"{len(df):,}")
c2.metric("With both Club Elo ratings", f"{both.sum():,}",
          help="Home AND away rating valid the day before kickoff.")
c3.metric("Coverage", f"{both.mean():.1%}" if len(df) else "—",
          help="E1 (Championship) is only partially covered by Club Elo; "
          "top divisions are near-complete.")

unmapped = results.get("unmapped_teams", [])
if unmapped:
    with st.expander(f"Unmapped teams ({len(unmapped)})"):
        st.markdown(
            "No Club Elo history found under the mapped or raw name — their "
            "matches carry NaN ratings and drop out of the study universe: "
            + ", ".join(unmapped)
        )

st.subheader("Club Elo difference vs market home probability")
sub = df.loc[both & df["p_cons_H"].notna(),
             ["elo_diff_clubelo", "p_cons_H", "result"]]
if len(sub) > 8000:
    sub = sub.sample(8000, random_state=0)
fig = go.Figure(
    go.Scattergl(
        x=sub["elo_diff_clubelo"], y=sub["p_cons_H"], mode="markers",
        marker=dict(size=4, opacity=0.25),
        name="matches",
    )
)
if len(sub) > 50:
    # Binned mean overlay: the market's read of the same rating gap.
    bins = pd.qcut(sub["elo_diff_clubelo"], 25, duplicates="drop")
    binned = sub.groupby(bins, observed=True).agg(
        x=("elo_diff_clubelo", "mean"), y=("p_cons_H", "mean")
    )
    fig.add_trace(go.Scatter(x=binned["x"], y=binned["y"],
                             mode="lines+markers", name="binned mean"))
fig.update_layout(
    xaxis_title="Club Elo difference (home − away)",
    yaxis=dict(title="consensus P(home win)", tickformat=".0%"),
    height=480,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The tight monotone band is the whole story: the market has already "
    "priced whatever the external rating knows. The out-of-sample test "
    "below asks whether the residual scatter carries any signal."
)

st.subheader("Out-of-sample: consensus vs consensus + Club Elo")
per_league = results.get("leagues", {})
pooled = results.get("pooled")
block = pooled if league == config.ALL_LEAGUES else per_league.get(league)
if block is None:
    st.caption("Not enough covered seasons in this league to run the test.")
else:
    label = ("pooled, 5 covered leagues" if league == config.ALL_LEAGUES
             else config.league_display_name(league))
    vs = block["candidate_vs_consensus"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test matches", f"{block['n_test']:,}", help=block["test_seasons"])
    m2.metric("Consensus log loss", f"{block['consensus_log_loss']:.4f}")
    m3.metric("+ Club Elo log loss", f"{block['candidate_log_loss']:.4f}")
    m4.metric(
        "Diff (95% CI)",
        f"{vs['diff']:+.4f}",
        help=f"[{vs['lo']:+.4f}, {vs['hi']:+.4f}] on identical matches "
        f"({label}). Negative = Club Elo helps; significant only if the CI "
        "excludes zero.",
    )
    if vs["significantly_better"]:
        st.success("Club Elo adds significant out-of-sample signal here.")
    elif vs["significantly_worse"]:
        st.error("Adding Club Elo is significantly WORSE than the raw consensus.")
    else:
        st.info(
            "No significant difference: the external rating adds nothing "
            "the market had not already priced."
        )

rows = [("ALL (pooled)", pooled)] if pooled else []
rows += [(config.league_display_name(lg), per_league[lg])
         for lg in COVERED_LEAGUES + ["E1"] if lg in per_league]
if rows:
    names = [name for name, _ in rows]
    diffs = [b["candidate_vs_consensus"]["diff"] for _, b in rows]
    los = [b["candidate_vs_consensus"]["lo"] for _, b in rows]
    his = [b["candidate_vs_consensus"]["hi"] for _, b in rows]
    fig = go.Figure(
        go.Scatter(
            x=diffs, y=names, mode="markers", marker=dict(size=11),
            error_x=dict(type="data", symmetric=False,
                         array=np.array(his) - np.array(diffs),
                         arrayminus=np.array(diffs) - np.array(los)),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="log-loss diff vs raw consensus (negative = Club Elo helps)",
        yaxis=dict(autorange="reversed"),
        height=90 + 55 * len(rows),
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("In-sample LR test and naive sign check"):
    for name, b in rows:
        lr, nv = b["lr_test"], b["naive"]
        st.markdown(
            f"- **{name}**: LR stat {lr['lr_stat']:.2f} (df {lr['df']}, "
            f"p = {lr['p_value']:.3g}); higher-Elo side wins "
            f"{nv['clubelo_sign_accuracy']:.1%} vs always-home "
            f"{nv['always_home_accuracy']:.1%} ({nv['n']:,} matches)."
        )

explain(
    "How to read this",
    """
**The source.** [clubelo.com](http://clubelo.com) maintains daily Elo
ratings for European clubs computed purely from match results — no odds,
no market input. Every previous candidate feature on this platform was
derived from the same football-data.co.uk feed as the odds themselves;
this is the first genuinely **external** information source tested.

**As-of discipline.** Each match uses the rating valid on the day *before*
kickoff (the Club Elo `From ≤ d ≤ To` validity row). A match's own result
can therefore never leak into the rating that predicts it.

**The test.** Baseline: multinomial logit on the two consensus log-odds
ratios. Candidate: baseline + the standardized Club Elo rating difference.
Both are refitted each season on all prior seasons (expanding window) and
scored on the identical out-of-sample matches. The headline number is the
paired-bootstrap 95% CI of the candidate's log loss minus the **raw market
consensus** log loss: negative and excluding zero would mean the external
rating adds real signal.

**Why the naive check matters.** The higher-Elo side does beat an
always-home rule on picks — Club Elo genuinely measures strength. The
point is that measuring strength is not enough: the market has already
absorbed it, with margin to spare.
""",
)
