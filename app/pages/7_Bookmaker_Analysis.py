import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import calibration, config
from src.evaluate import (
    bootstrap_ci,
    brier_per_match,
    log_loss_per_match,
    paired_bootstrap_diff,
)

page_setup("Bookmaker Analysis", icon="🔍")
st.caption(
    "One bookmaker under the microscope: coverage, margin, calibration, "
    "sharpness vs the market consensus, and where its probabilities lean."
)

df_all = season_filter(load_matches())

candidates = [
    b for b in BOOKS
    if b != "cons"
    and all(f"p_{b}_{o}" in df_all.columns for o in config.OUTCOMES)
]
book = st.selectbox("Bookmaker", candidates, format_func=BOOKS.get)

p_cols = [f"p_{book}_{o}" for o in config.OUTCOMES]
cons_cols = [f"p_cons_{o}" for o in config.OUTCOMES]
df = df_all.dropna(subset=p_cols)
if df.empty:
    st.warning("This book has no coverage in the selected seasons.")
    st.stop()

probs = df[p_cols].to_numpy()
ll = log_loss_per_match(probs, df["result"])
ll_mean, ll_lo, ll_hi = bootstrap_ci(ll)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matches covered", f"{len(df):,}")
c2.metric("Coverage", f"{len(df) / len(df_all):.0%}")
c3.metric("Mean margin", f"{df[f'margin_{book}'].mean():.2%}")
c4.metric("Log loss", f"{ll_mean:.4f}", help=f"95% CI [{ll_lo:.4f}, {ll_hi:.4f}]")
c5.metric("Brier score", f"{brier_per_match(probs, df['result']).mean():.4f}")

# ---- Sharpness vs consensus ------------------------------------------------
st.subheader("Against the market consensus")
common = df.dropna(subset=cons_cols)
book_ll = log_loss_per_match(common[p_cols].to_numpy(), common["result"])
cons_ll = log_loss_per_match(common[cons_cols].to_numpy(), common["result"])
diff = paired_bootstrap_diff(book_ll, cons_ll)
better = diff["diff"] < 0
st.markdown(
    f"On the {len(common):,} shared matches, **{BOOKS[book]}** scores "
    f"**{diff['diff']:+.4f}** log loss vs the weighted consensus "
    f"(95% CI [{diff['lo']:+.4f}, {diff['hi']:+.4f}]) — "
    + (
        "**sharper than the consensus**."
        if better and diff["hi"] < 0
        else "sharper on average, but not conclusively."
        if better
        else "no sharper than the consensus."
    )
)

season_ll = pd.DataFrame(
    {
        "season": common["season"].to_numpy(),
        BOOKS[book]: book_ll,
        "consensus": cons_ll,
    }
).groupby("season").mean()
fig = px.line(
    season_ll.reset_index().melt(id_vars="season", var_name="source", value_name="log loss"),
    x="season", y="log loss", color="source", markers=True,
)
fig.update_layout(height=380, yaxis_title="mean log loss (lower = sharper)")
st.plotly_chart(fig, use_container_width=True)

# ---- Margin over time -------------------------------------------------------
st.subheader("Margin by season")
margin = df.groupby("season")[f"margin_{book}"].mean().reset_index()
fig = px.bar(margin, x="season", y=f"margin_{book}", labels={f"margin_{book}": "overround"})
fig.update_yaxes(tickformat=".1%")
fig.update_layout(height=340)
st.plotly_chart(fig, use_container_width=True)

# ---- Calibration ------------------------------------------------------------
st.subheader("Calibration")
rel = calibration.reliability_curve(df, f"p_{book}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                         line=dict(dash="dash", color="gray")))
fig.add_trace(
    go.Scatter(
        x=rel["predicted"], y=rel["observed"], mode="markers+lines",
        name="observed",
        error_y=dict(type="data", symmetric=False,
                     array=rel["ci_hi"] - rel["observed"],
                     arrayminus=rel["observed"] - rel["ci_lo"]),
    )
)
fig.update_layout(
    xaxis=dict(title="implied probability (de-margined)", tickformat=".0%"),
    yaxis=dict(title="observed frequency", tickformat=".0%"),
    height=460,
)
st.plotly_chart(fig, use_container_width=True)

try:
    eff = calibration.efficiency_regression(df, f"p_{book}")
    e1, e2 = st.columns(2)
    e1.metric("Calibration slope", f"{eff['slope']:.3f} ± {eff['slope_se']:.3f}",
              help="1.0 = perfectly calibrated")
    e2.metric("Intercept", f"{eff['intercept']:.3f} ± {eff['intercept_se']:.3f}")
except Exception:
    st.caption("Too little data for the efficiency regression here.")

# ---- Bias vs consensus -------------------------------------------------------
st.subheader("Where does this book lean?")
st.markdown(
    "Mean difference between this book's probability and the consensus, per "
    "outcome. Recreational books typically shade popular home favorites "
    "(positive H bias) and pad longshots."
)
bias = pd.DataFrame(
    {
        "outcome": ["Home", "Draw", "Away"],
        "bias": [
            (common[f"p_{book}_{o}"] - common[f"p_cons_{o}"]).mean()
            for o in config.OUTCOMES
        ],
    }
)
fig = px.bar(bias, x="outcome", y="bias", labels={"bias": "mean(p_book − p_consensus)"})
fig.update_yaxes(tickformat="+.2%")
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)
