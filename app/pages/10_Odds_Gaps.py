import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import BOOKS, load_matches, page_setup, season_filter
from src import calibration, config
from src.evaluate import log_loss_per_match, paired_bootstrap_diff

page_setup("Odds Gaps", icon="↔️")
st.caption(
    "What happens when bookmakers significantly disagree? Gap = spread "
    "(max − min) of the de-margined home-win probability across all books "
    "quoting the match."
)

df = season_filter(load_matches())
df = df[(df["n_books"] >= 3) & df["disp_range"].notna()]

q90 = float(np.nanquantile(df["disp_range"], 0.90))
threshold = st.sidebar.slider(
    "Gap threshold (probability spread)",
    min_value=0.02,
    max_value=0.15,
    value=round(q90 * 200) / 200,  # nearest half-point
    step=0.005,
    format="%.3f",
    help="A match counts as a gap match when the home-win probability "
    "spread across books is at least this large. Default ≈ top decile.",
)
gap = df[df["disp_range"] >= threshold]
rest = df[df["disp_range"] < threshold]

c1, c2, c3 = st.columns(3)
c1.metric("Matches with odds", f"{len(df):,}")
c2.metric("Gap matches", f"{len(gap):,} ({len(gap) / len(df):.1%})")
c3.metric("Median spread", f"{df['disp_range'].median():.3f}")

# ---- 1. Frequency ------------------------------------------------------------
st.subheader("1 · How often, and are gaps dying out?")
col_a, col_b = st.columns(2)
with col_a:
    fig = px.histogram(df, x="disp_range", nbins=80,
                       labels={"disp_range": "home-prob spread across books"})
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text="threshold")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    share = (df["disp_range"] >= threshold).groupby(df["season"]).mean()
    fig = px.bar(share.reset_index(), x="season", y="disp_range",
                 labels={"disp_range": "share of gap matches"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Left: distribution of disagreement with your threshold. Right: share of "
    "matches over the threshold per season — how synchronized the books have "
    "become over two decades."
)

st.markdown("**How big are the gaps, year by year?**")
per_season = df.groupby("season").agg(
    median_spread=("disp_range", "median"),
    p90_spread=("disp_range", lambda s: s.quantile(0.90)),
    books=("n_books", "mean"),
)
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=per_season.index, y=per_season["books"], name="books quoting (avg)",
        yaxis="y2", marker_color="rgba(128,128,128,0.25)",
    )
)
fig.add_trace(
    go.Scatter(x=per_season.index, y=per_season["median_spread"],
               mode="lines+markers", name="median spread")
)
fig.add_trace(
    go.Scatter(x=per_season.index, y=per_season["p90_spread"],
               mode="lines+markers", name="90th percentile spread",
               line=dict(dash="dot"))
)
fig.update_layout(
    height=420,
    yaxis=dict(title="home-prob spread across books", tickformat=".1%"),
    yaxis2=dict(title="books in panel", overlaying="y", side="right",
                showgrid=False),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Both the typical (median) and extreme (90th percentile) disagreement "
    "shrink over the years. Read them against the gray bars: a wider panel "
    "mechanically produces larger max-min spreads, so the 2005-2019 era "
    "(~30-40 books) overstates disagreement relative to the ~8-book panels "
    "before and after. That the spread still falls while the panel widens "
    "makes the convergence conclusion conservative, not weaker."
)

if len(gap) < 50:
    st.warning("Fewer than 50 gap matches — lower the threshold or widen the seasons.")
    st.stop()

# ---- 2. Who is right? ---------------------------------------------------------
st.subheader("2 · Who is right when they disagree?")
scored = []
for book in ("PS", "B365", "WH", "Avg", "Max", "cons"):
    cols = [f"p_{book}_{o}" for o in config.OUTCOMES]
    if not all(c in gap.columns for c in cols):
        continue
    sub = gap.dropna(subset=cols + [f"p_cons_{o}" for o in config.OUTCOMES])
    if len(sub) < 50:
        continue
    ll = log_loss_per_match(sub[cols].to_numpy(), sub["result"])
    cons_ll = log_loss_per_match(
        sub[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy(), sub["result"]
    )
    boot = paired_bootstrap_diff(ll, cons_ll)
    scored.append(
        {
            "book": BOOKS[book],
            "log_loss": ll.mean(),
            "vs consensus": boot["diff"],
            "ci_lo": boot["lo"],
            "ci_hi": boot["hi"],
            "n": len(sub),
        }
    )
board = pd.DataFrame(scored).sort_values("log_loss")
fig = go.Figure(
    go.Scatter(
        x=board["vs consensus"], y=board["book"], mode="markers",
        marker=dict(size=11),
        error_x=dict(type="data", symmetric=False,
                     array=board["ci_hi"] - board["vs consensus"],
                     arrayminus=board["vs consensus"] - board["ci_lo"]),
    )
)
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(
    xaxis_title="log-loss difference vs consensus on gap matches "
    "(negative = sharper), 95% paired-bootstrap CI",
    yaxis=dict(autorange="reversed"),
    height=80 + 55 * len(board),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Follow the sharp?** Pinnacle vs Bet365 direction on gap matches:")
ss = gap.dropna(subset=["sharp_soft_gap", "p_PS_H", "p_B365_H"])
rows = []
for label, side in (
    ("Pinnacle higher on home", ss[ss["sharp_soft_gap"] > 0]),
    ("Bet365 higher on home", ss[ss["sharp_soft_gap"] < 0]),
):
    if len(side) < 20:
        continue
    rows.append(
        {
            "direction": label,
            "matches": len(side),
            "Pinnacle implied": side["p_PS_H"].mean(),
            "Bet365 implied": side["p_B365_H"].mean(),
            "realized home wins": (side["result"] == "H").mean(),
        }
    )
if rows:
    st.dataframe(
        pd.DataFrame(rows).set_index("direction").style.format(
            {
                "Pinnacle implied": "{:.1%}",
                "Bet365 implied": "{:.1%}",
                "realized home wins": "{:.1%}",
            }
        ),
        use_container_width=True,
    )
    st.caption(
        "If the realized rate sits closer to Pinnacle's number in both rows, "
        "the sharp book wins the argument. (Pinnacle coverage starts 2012-13, "
        "so this table uses that era only.)"
    )
else:
    st.info("Not enough Pinnacle/Bet365 overlap in this selection.")

# ---- 3. Do gap matches stay predictable? --------------------------------------
st.subheader("3 · Do gap matches stay predictable?")
cols = st.columns(2)
for col, (label, subset) in zip(
    cols, (("gap matches", gap), ("all other matches", rest))
):
    with col:
        st.markdown(f"**{label}** ({len(subset):,})")
        try:
            eff = calibration.efficiency_regression(subset, "p_cons")
            st.metric("Calibration slope", f"{eff['slope']:.3f} ± {eff['slope_se']:.3f}")
        except Exception:
            st.caption("regression failed (too few matches)")
        p_cols = [f"p_cons_{o}" for o in config.OUTCOMES]
        with_p = subset.dropna(subset=p_cols)
        fav_idx = with_p[p_cols].to_numpy().argmax(axis=1)
        fav_won = (
            np.array([config.OUTCOMES[i] for i in fav_idx]) == with_p["result"].to_numpy()
        )
        st.metric("Favorite win rate", f"{fav_won.mean():.1%}")
        st.metric(
            "Mean favorite probability",
            f"{with_p[p_cols].to_numpy().max(axis=1).mean():.1%}",
        )
st.caption(
    "If the slope stays ≈1 on gap matches, the consensus remains honest even "
    "where books disagree. Compare the favorite win rate to its implied "
    "probability, not across columns — gap matches skew toward different "
    "match profiles."
)

# ---- 4. Paper arbitrage --------------------------------------------------------
st.subheader("4 · The paper-arbitrage corner")
st.markdown(
    "When the spread is wide enough, the **best available prices** across "
    "books imply probabilities summing to less than 100% (`margin_Max < 0`) — "
    "a risk-free profit on paper."
)
arb = df[df["margin_Max"] < 0].copy()
arb["size"] = -arb["margin_Max"]
# Best-price columns occasionally carry palpable errors (a stale or
# fat-fingered price at one book). A risk-free margin above 10% is
# essentially always such an error, not a real market state.
plausible = arb[arb["size"] <= 0.10]
a1, a2, a3 = st.columns(3)
a1.metric(
    "Paper-arb matches",
    f"{len(plausible):,} ({len(plausible) / len(df):.1%})",
)
a2.metric("Median size", f"{plausible['size'].median():.2%}" if len(plausible) else "—")
a3.metric(
    "Excluded as data errors",
    f"{len(arb) - len(plausible):,}",
    help="Matches with a best-price margin below -10% — e.g. Arsenal-Charlton "
    "2006 lists a max home price of 4.00 against a market average of 1.40, "
    "an impossible quote that only ever existed in a broken feed.",
)
if len(plausible):
    arb_share = (
        plausible.groupby("season").size().reindex(
            sorted(df["season"].unique()), fill_value=0
        )
        / df.groupby("season").size()
    )
    fig = px.bar(arb_share.reset_index(name="share"), x="season", y="share",
                 labels={"share": "share with negative best-price margin"})
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The mid-2000s to 2019 spike is an artifact of breadth, not "
        "inefficiency: in that era the source's best price scanned ~40 books "
        "(Betbrain), so *some* pair of books nearly always disagreed by more "
        "than the smallest margin. From 2019-20 the panel shrinks to ~8 books."
    )
st.caption(
    "Educational, not actionable: these are snapshot prices — by the time "
    "both legs could be placed, the stale side has moved. Their small size "
    "and quick disappearance are themselves evidence of market efficiency."
)
