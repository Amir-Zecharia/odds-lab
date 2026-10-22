import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import explain, page_setup
from src import calibration
from src.ledger import BACKTEST_BASELINE, load_ledger, scoreboard

page_setup("Live Ledger", icon="📓")
st.caption(
    "Forward testing: every prediction written down **before kickoff**, "
    "frozen when the match starts, scored after. The one result on this "
    "platform that hindsight cannot touch."
)

st.info(
    "Settlement reads local historical results, not a live scores feed. "
    "The bundled snapshot ends May 2026. To settle later fixtures, explicitly "
    "update the season coverage in src/config.py and refresh local results "
    "before running python -m src.ledger settle. Unmatched fixtures remain open."
)

ledger = load_ledger()
board = scoreboard(ledger)

if board["n_settled"] == 0 and board["n_open"] == 0:
    st.info(
        "The ledger is empty. Record your first predictions:\n\n"
        "```bash\nODDS_API_KEY=<key> python -m src.ledger record\n```\n\n"
        "…then after the matches finish:\n\n"
        "```bash\npython -m src.ingest && python -m src.ledger settle\n```\n\n"
        "Or use the **Record these predictions** button on the Upcoming "
        "Matches page. Use your own The Odds API key; request limits depend on your plan."
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Settled predictions", f"{board['n_settled']:,}")
c2.metric("Awaiting results", f"{board['n_open']:,}")
if board["n_settled"]:
    c3.metric(
        "Live log loss",
        f"{board['log_loss']:.4f}",
        help=f"95% CI [{board['log_loss_lo']:.4f}, {board['log_loss_hi']:.4f}]. "
        f"Backtest baseline over 32,780 historical matches: {BACKTEST_BASELINE:.4f}.",
    )
    c4.metric("Favorite hit rate", f"{board['favorite_hit_rate']:.1%}",
              help=f"Mean favorite probability {board['mean_favorite_prob']:.1%}")

if board["n_settled"] and not board.get("sample_is_meaningful", False):
    st.warning(
        f"**Small sample:** {board['n_settled']} settled predictions. The "
        "confidence interval above is wide, and any gap to the backtest "
        "baseline is currently noise. This page gets meaningful around "
        "200+ matches — it is designed to be patient."
    )

settled = ledger.dropna(subset=["result"]) if not ledger.empty else ledger

if len(settled) >= 20:
    st.subheader("Running log loss")
    chron = settled.sort_values("kickoff").reset_index(drop=True)
    chron["cumulative"] = chron["log_loss"].expanding().mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chron["kickoff"], y=chron["cumulative"],
                             mode="lines+markers", name="live cumulative"))
    fig.add_hline(y=BACKTEST_BASELINE, line_dash="dash", line_color="gray",
                  annotation_text="backtest baseline")
    fig.update_layout(height=400, yaxis_title="cumulative mean log loss")
    st.plotly_chart(fig, use_container_width=True)

if len(settled) >= 50:
    st.subheader("Live calibration")
    st.caption(
        "The signature chart, rebuilt from predictions made in advance "
        "rather than from history."
    )
    live = settled.rename(
        columns={"p_home": "p_cons_H", "p_draw": "p_cons_D", "p_away": "p_cons_A"}
    )
    try:
        rel = calibration.reliability_curve(live, "p_cons", n_bins=6)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                 line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(
            x=rel["predicted"], y=rel["observed"], mode="markers+lines",
            name="observed",
            error_y=dict(type="data", symmetric=False,
                         array=rel["ci_hi"] - rel["observed"],
                         arrayminus=rel["observed"] - rel["ci_lo"]),
        ))
        fig.update_layout(
            xaxis=dict(title="predicted probability", tickformat=".0%"),
            yaxis=dict(title="observed frequency", tickformat=".0%"),
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Not enough spread across probability bins yet.")

st.subheader("The ledger")
tab_open, tab_settled = st.tabs(["Awaiting results", "Settled"])
show_cols = ["kickoff", "home_team", "away_team", "p_home", "p_draw", "p_away", "source"]
with tab_open:
    open_rows = ledger[ledger["result"].isna()]
    if open_rows.empty:
        st.caption("No open predictions — record some before the next matchday.")
    else:
        st.dataframe(
            open_rows[show_cols].style.format(
                {"p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}"}
            ),
            use_container_width=True,
        )
with tab_settled:
    if settled.empty:
        st.caption("Nothing settled yet.")
    else:
        st.dataframe(
            settled[show_cols + ["result", "log_loss"]]
            .sort_values("kickoff", ascending=False)
            .style.format(
                {"p_home": "{:.1%}", "p_draw": "{:.1%}", "p_away": "{:.1%}",
                 "log_loss": "{:.3f}"}
            ),
            use_container_width=True,
        )

explain(
    "Why this page is the strongest evidence here",
    """
- **Every other study is retrospective.** They are done carefully — strict
  time-ordered splits, bootstrap CIs, FDR control — but a backtest can
  always be quietly shaped by decisions made after seeing the data.
- **This one cannot.** Predictions are written to disk before kickoff and
  the row is **frozen** the moment the match starts: a later run can never
  revise a forecast it now knows the answer to. That rule is enforced in
  code and covered by a test that tries to tamper with a started match.
- **The reference line** is the market consensus' backtest log loss
  (0.9832 over 32,780 historical matches). If the live number tracks it,
  the platform's central finding survives contact with the future.
- **Be patient with it.** Below ~200 settled predictions the interval is
  too wide to conclude anything, and the page says so rather than
  implying precision it doesn't have.
""",
)
