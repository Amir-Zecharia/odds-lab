import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from common import BOOKS, explain, load_matches, page_setup, season_filter
from src.mismatch import (
    DIMENSIONS,
    RESULTS_JSON,
    scan_segments,
    stacked_universe,
)

page_setup("Mismatch Explorer", icon="🔬")
st.caption(
    "The microscope: where do implied probabilities and reality actually "
    "disagree? Slice the 61k-match universe any way you like — the scanner "
    "recomputes edges, significance and FDR flags live."
)

df = season_filter(load_matches())

source = st.sidebar.selectbox(
    "Probability source",
    [b for b in ("cons", "PSC", "PS", "B365", "Avg") if f"p_{b}_H" in df.columns],
    format_func=BOOKS.get,
)
min_n = st.sidebar.slider("Minimum matches per cell", 100, 2000, 300, 50)
lo, hi = st.sidebar.slider(
    "Implied probability range", 0.0, 1.0, (0.0, 1.0), 0.05,
    help="Keep only (match, outcome) rows whose implied probability falls "
    "in this range — e.g. 0-0.15 to study extreme longshots only.",
)

long = stacked_universe(df, source)
if lo > 0 or hi < 1:
    long = long[long["prob"].between(lo, hi)]

dims = st.multiselect(
    "Slice by",
    DIMENSIONS,
    default=["outcome", "band"],
    help="Every combination of the chosen dimensions becomes one cell; "
    "each cell is tested for a real gap between implied and realized.",
)
if not dims:
    st.info("Pick at least one dimension.")
    st.stop()

table = scan_segments(long, dims, min_n=min_n)
if table.empty:
    st.warning("No cells reach the minimum size — lower it or widen filters.")
    st.stop()

n_sig = int(table["fdr_significant"].sum())
c1, c2, c3 = st.columns(3)
c1.metric("Cells scanned", len(table))
c2.metric("FDR-significant cells", n_sig)
c3.metric("Rows in scan", f"{len(long):,}")

if len(dims) == 2:
    pivot = table.pivot(index=dims[0], columns=dims[1], values="edge")
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels=dict(color="edge (realized − implied)"),
        aspect="auto",
        text_auto=".1%",
    )
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

show = table.copy()
show["flag"] = show["fdr_significant"].map({True: "⭐", False: ""})
st.dataframe(
    show.drop(columns=["fdr_significant"]).style.format(
        {
            "implied": "{:.1%}", "realized": "{:.1%}", "edge": "{:+.2%}",
            "z": "{:+.2f}", "p_value": "{:.4f}",
        }
    ),
    use_container_width=True,
    height=400,
)

st.subheader("Confirmed pockets (the honest list)")
st.markdown(
    "Cells above are *exploration* — scan enough slices and some will glow "
    "by luck. The list below survived the full protocol: FDR-significant "
    "in the discovery era (≤2017-18) **and** replicated with the same sign "
    "on data it never saw."
)
if RESULTS_JSON.exists():
    pockets = json.loads(RESULTS_JSON.read_text())
    st.caption(
        f"{pockets['candidates_after_fdr']} candidates survived FDR in "
        f"discovery; {len(pockets['confirmed_pockets'])} replicated."
    )
    if pockets["confirmed_pockets"]:
        rows = [
            {
                **p["segment"],
                "discovery edge": p["discovery"]["edge"],
                "confirmed edge": p["confirmation"]["edge"],
                "confirmation p": p["confirmation"]["p_value"],
                "confirmation n": p["confirmation"]["n"],
            }
            for p in pockets["confirmed_pockets"]
        ]
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {
                    "discovery edge": "{:+.2%}",
                    "confirmed edge": "{:+.2%}",
                    "confirmation p": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.success(
            "No pocket replicated — every apparent mispricing in the "
            "discovery era evaporated on fresh data. That *is* the finding: "
            "the market's calibration holds even under a deliberate dredge."
        )
else:
    st.caption("Run `python -m src.mismatch` to generate the confirmed list.")

explain(
    "How to read this page",
    """
- **Edge** = how often the outcome actually happened minus how often the
  odds said it would. +2% on draws means draws hit 2 points more often
  than priced.
- **Why most cells should be noise:** scan 200 cells at the 5% level and
  ~10 will look significant by pure luck. The ⭐ flag uses
  **Benjamini-Hochberg FDR** so that at most ~10% of flagged cells are
  expected to be flukes — and even ⭐ cells you find while exploring are
  *candidates*, not findings.
- **The confirmed list is the standard of proof:** discovered before
  2017-18, then replicated on later seasons. Anything that passes both is
  a genuine, durable mispricing — and would be front-page news in the
  market-efficiency literature.
- Filters up top slice the universe (league and seasons in the sidebar,
  probability range for longshots/favorites); the scan always recomputes
  on exactly what you selected.
""",
)
