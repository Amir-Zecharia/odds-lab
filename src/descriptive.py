"""Descriptive statistics layer: the platform's "explore the data" section.

Doubles as data validation — every table here bounds what the analyses
downstream can claim (especially the coverage matrix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.probabilities import available_books


def outcome_base_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of H/D/A per season (the naive baseline)."""
    rates = (
        df.groupby("season")["result"]
        .value_counts(normalize=True)
        .unstack()
        .reindex(columns=config.OUTCOMES)
    )
    rates.columns = ["home_win", "draw", "away_win"]
    return rates


def margin_by_season(df: pd.DataFrame, books: list[str] | None = None) -> pd.DataFrame:
    """Mean overround per book per season (the two-decade declining trend)."""
    books = books or [config.SHARP_BOOK, config.SOFT_BOOK, "Avg", "Max"]
    cols = {b: f"margin_{b}" for b in books if f"margin_{b}" in df.columns}
    out = df.groupby("season")[list(cols.values())].mean()
    out.columns = list(cols.keys())
    return out


def coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of matches per season with complete H/D/A odds per book."""
    books = available_books(df) + [
        a for a in ("Avg", "Max") if f"{a}H" in df.columns
    ]
    cov = {}
    for prefix in books:
        cols = [f"{prefix}{o}" for o in config.OUTCOMES]
        cov[prefix] = df[cols].notna().all(axis=1).groupby(df["season"]).mean()
    out = pd.DataFrame(cov)
    # Order books by overall coverage, best first.
    return out[out.mean().sort_values(ascending=False).index]


def favorite_win_rate_by_band(
    df: pd.DataFrame, prob_col_prefix: str = "p_cons", n_bands: int = 10
) -> pd.DataFrame:
    """The raw table behind calibration: favorite implied prob vs win rate."""
    cols = [f"{prob_col_prefix}_{o}" for o in config.OUTCOMES]
    probs = df[cols].to_numpy(float)
    valid = ~np.isnan(probs).any(axis=1)
    sub = df.loc[valid]
    probs = probs[valid]

    fav_idx = probs.argmax(axis=1)
    fav_prob = probs.max(axis=1)
    fav_won = np.array(
        [config.OUTCOMES[i] for i in fav_idx]
    ) == sub["result"].to_numpy()

    bands = pd.cut(fav_prob, bins=np.linspace(1 / 3, 1.0, n_bands + 1))
    table = pd.DataFrame(
        {"band": bands, "implied": fav_prob, "won": fav_won.astype(float)}
    )
    out = table.groupby("band", observed=True).agg(
        matches=("won", "size"),
        mean_implied=("implied", "mean"),
        win_rate=("won", "mean"),
    )
    out["gap"] = out["win_rate"] - out["mean_implied"]
    return out


def dispersion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of cross-book disagreement levels per season."""
    return df.groupby("season")["disp_std"].describe()[
        ["count", "mean", "50%", "75%", "max"]
    ]


def draw_rate_by_season(df: pd.DataFrame) -> pd.Series:
    return (df["result"] == "D").groupby(df["season"]).mean().rename("draw_rate")
