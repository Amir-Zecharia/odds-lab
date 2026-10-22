import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import explain, page_setup
from src.model_stack import RESULTS_JSON

page_setup("Model Stack", icon="🗼")
st.caption(
    "The model of models: five odds-free predictors — Dixon-Coles, Elo, "
    "Pi-ratings, Massey, Colley — and a meta-model that learns how much to "
    "trust each. Then the market joins the stack, and the learned weights "
    "reveal the information hierarchy."
)

if not RESULTS_JSON.exists():
    st.info("Run `python -m src.model_stack` to compute this study (~20 min).")
    st.stop()

results = json.loads(RESULTS_JSON.read_text())
st.markdown(
    f"Universe: **{results['n_universe']:,} matches** where every source "
    f"speaks; **{results['n_test']:,}** out-of-sample test matches "
    f"({results['test_seasons']}), expanding-window."
)

# ---- the race -----------------------------------------------------------------
st.subheader("The race")
table = pd.DataFrame(results["race"])
fig = go.Figure(
    go.Scatter(
        x=table["diff_vs_market"],
        y=table["model"],
        mode="markers",
        marker=dict(
            size=11,
            color=["#22c55e" if "STACK" in m else ("#9ca3af" if "consensus" in m else "#60a5fa") for m in table["model"]],
        ),
        error_x=dict(type="data", symmetric=False,
                     array=table["hi"] - table["diff_vs_market"],
                     arrayminus=table["diff_vs_market"] - table["lo"]),
    )
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    xaxis_title="log-loss difference vs raw market consensus "
    "(negative = beats the market), 95% paired-bootstrap CI",
    yaxis=dict(autorange="reversed"),
    height=90 + 52 * len(table),
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(
    table.set_index("model").style.format(
        {"log_loss": "{:.4f}", "diff_vs_market": "{:+.4f}",
         "lo": "{:+.4f}", "hi": "{:+.4f}"}
    ),
    use_container_width=True,
)

# ---- the hierarchy ---------------------------------------------------------------
st.subheader("What the meta-model learned to trust")
weights = pd.DataFrame(results["weights_with_market"])
if not weights.empty:
    nice = {
        "dc_logit_H": "Dixon-Coles", "elo_diff": "Elo", "pi_diff": "Pi-ratings",
        "massey_diff": "Massey", "colley_diff": "Colley",
        "mkt_logit_H": "Market (home)", "mkt_logit_D": "Market (draw)",
    }
    melted = weights.melt(id_vars="test_season", var_name="source", value_name="weight")
    melted["source"] = melted["source"].map(nice)
    fig = px.area(
        melted, x="test_season", y="weight", color="source",
        labels={"test_season": "season being predicted", "weight": "share of learned weight"},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)
    market_share = weights[["mkt_logit_H", "mkt_logit_D"]].sum(axis=1)
    st.caption(
        f"With the market inside the stack, the market's share of the "
        f"learned weight averages **{market_share.mean():.0%}** across "
        "refits — the information hierarchy in one number."
    )

explain(
    "How to read this page",
    """
- **Stacking** is the "model of models": each level-0 system outputs its
  signal, and a level-1 logistic regression — trained only on strictly
  earlier seasons — learns the trust weights. No source sees the future,
  and the meta-model never grades itself on matches it trained on.
- **Why the zoo helps**: Elo and Pi listen to recency, Massey to score
  margins over a whole window, Colley only to wins. Their *disagreements*
  carry information a single system can't, which is why the odds-free
  stack should edge out its best individual member.
- **The collapse test**: when the market's own probabilities join the
  stack, a rational meta-model should shift its weight onto them exactly
  to the extent the market already contains the zoo's knowledge. The
  area chart shows that shift directly.
- Full methodology and every other study: see the Research Log.
""",
)
