import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_analysis, load_matches, page_setup, season_filter
from src import config
from src.evaluate import brier_per_match

page_setup("Disagreement & Error", icon="🌪️")
st.caption(
    "When bookmakers disagree about a match, is the market actually more "
    "wrong about it? OLS of per-match consensus error on cross-book dispersion."
)

analysis = load_analysis()
reg = analysis["dispersion_error"]

c1, c2, c3 = st.columns(3)
c1.metric("Dispersion coefficient", f"{reg['coef_dispersion']:+.3f}")
c2.metric("p-value", f"{reg['p_dispersion']:.3f}")
c3.metric("R²", f"{reg['r_squared']:.4f}")
verdict = (
    "Dispersion **does** predict market error at the 5% level."
    if reg["p_dispersion"] < 0.05
    else "No significant relationship — bookmaker disagreement does **not** "
    "signal larger market errors, consistent with disagreement being noise "
    "around a shared efficient estimate."
)
st.markdown(
    f"{verdict} (Controls for match lopsidedness; HC1 robust errors; "
    f"n = {reg['n']:,}.)"
)

df = season_filter(load_matches())
cols = [f"p_cons_{o}" for o in config.OUTCOMES]
sub = df.dropna(subset=cols + ["disp_std"]).copy()
sub["consensus_brier"] = brier_per_match(sub[cols].to_numpy(), sub["result"])

st.subheader("Binned view")
sub["disp_bin"] = pd.qcut(sub["disp_std"], q=12, duplicates="drop")
binned = sub.groupby("disp_bin", observed=True).agg(
    dispersion=("disp_std", "mean"),
    mean_error=("consensus_brier", "mean"),
    n=("consensus_brier", "size"),
    se=("consensus_brier", lambda s: s.std() / np.sqrt(len(s))),
)
fig = px.scatter(
    binned,
    x="dispersion",
    y="mean_error",
    error_y="se",
    size="n",
    labels={
        "dispersion": "cross-book dispersion (std of home prob)",
        "mean_error": "mean consensus Brier score",
    },
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Each point is a dispersion quantile bin (size = matches). A flat line "
    "means disagreement carries no information about market accuracy."
)
