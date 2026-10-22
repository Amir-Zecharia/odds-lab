import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import config
from src.evaluate import log_loss_per_match, paired_bootstrap_diff

page_setup("Line Movement", icon="📈")
st.caption(
    "The odds you see on Tuesday are not the odds at kickoff. How far does "
    "the line drift, does the drift carry information, and what does "
    "predicting early cost you?"
)

PAIRS = {  # early column prefix -> closing column prefix
    "PS": "PSC",
    "B365": "B365C",
    "WH": "WHC",
    "Avg": "AvgC",
    "Max": "MaxC",
}

df = season_filter(load_matches())
early = st.selectbox(
    "Book", list(PAIRS), format_func=lambda b: f"{BOOKS[b]} → {BOOKS[PAIRS[b]]}"
)
close = PAIRS[early]
cols_e = [f"p_{early}_{o}" for o in config.OUTCOMES]
cols_c = [f"p_{close}_{o}" for o in config.OUTCOMES]
sub = df.dropna(subset=cols_e + cols_c).copy()
if len(sub) < 200:
    st.warning("Too little paired early/closing data in this selection.")
    st.stop()

sub["drift"] = sub[f"p_{close}_H"] - sub[f"p_{early}_H"]
c1, c2, c3 = st.columns(3)
c1.metric("Matches with both prices", f"{len(sub):,}")
c2.metric("Median |drift| (home prob)", f"{sub['drift'].abs().median():.2%}")
c3.metric("Drift beyond 2 points", f"{(sub['drift'].abs() > 0.02).mean():.1%}")

st.subheader("How far does the line move?")
fig = px.histogram(sub, x="drift", nbins=80,
                   labels={"drift": "closing − early home probability"})
fig.update_xaxes(tickformat="+.0%")
fig.update_layout(height=380, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("The cost of predicting early")
ll_early = log_loss_per_match(sub[cols_e].to_numpy(), sub["result"])
ll_close = log_loss_per_match(sub[cols_c].to_numpy(), sub["result"])
boot = paired_bootstrap_diff(ll_early, ll_close)
st.markdown(
    f"On the same {len(sub):,} matches, the **early** price scores "
    f"{ll_early.mean():.4f} log loss vs **{ll_close.mean():.4f}** at the "
    f"close — predicting early costs **{boot['diff']:+.4f} nats/match** "
    f"(95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}])."
    + (
        " The closing line is significantly sharper: information arriving "
        "between the two snapshots (lineups, injuries, sharp money) is real."
        if boot["lo"] > 0
        else " No significant difference in this selection."
    )
)

st.subheader("Does the move itself predict?")
st.markdown(
    "Binned: how often the home team wins, given how the line moved. If the "
    "market merely wandered, the line would be flat; a slope means moves "
    "carry real information — the classic *follow the steam* effect."
)
sub["drift_bin"] = pd.qcut(sub["drift"], q=10, duplicates="drop")
binned = sub.groupby("drift_bin", observed=True).agg(
    drift=("drift", "mean"),
    home_win=("result", lambda s: (s == "H").mean()),
    early_implied=(f"p_{early}_H", "mean"),
    n=("result", "size"),
)
binned["excess"] = binned["home_win"] - binned["early_implied"]
fig = px.scatter(
    binned, x="drift", y="excess", size="n",
    labels={
        "drift": "closing − early home probability",
        "excess": "home wins − early implied",
    },
)
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_xaxes(tickformat="+.1%")
fig.update_yaxes(tickformat="+.1%")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

# Formal check: does drift add information beyond the early price?
p_early = np.clip(sub[f"p_{early}_H"].to_numpy(), 1e-9, 1 - 1e-9)
X = sm.add_constant(
    np.column_stack([np.log(p_early / (1 - p_early)), sub["drift"].to_numpy()])
)
res = sm.GLM(
    (sub["result"] == "H").astype(int).to_numpy(), X, family=sm.families.Binomial()
).fit()
st.markdown(
    f"Logistic regression of home wins on the early price **plus the "
    f"drift**: drift coefficient {res.params[2]:.2f} "
    f"(z = {res.tvalues[2]:.1f}, p = {res.pvalues[2]:.2g}). "
    + (
        "**The move adds real information beyond the early price** — "
        "whoever moves the line knows something. For your prediction: "
        "always take the latest available odds."
        if res.pvalues[2] < 0.05
        else "No significant information in the move for this selection."
    )
)
