"""The over/under 2.5 goals market: a second, binary market to test.

Same pipeline as 1X2 — implied probabilities, Shin de-margining (the
formula is outcome-count agnostic), calibration, per-book scoring — but
with a two-outcome market and a different ground truth (total goals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.probabilities import DEMARGIN

# Books quoting totals in the source (Pinnacle is "P" here, not "PS").
TOTAL_BOOKS = ["B365", "P", "Avg", "Max", "B365C", "PC", "AvgC", "MaxC"]

TOTAL_BOOK_NAMES = {
    "B365": "Bet365",
    "P": "Pinnacle",
    "Avg": "Market average",
    "Max": "Best price",
    "B365C": "Bet365 closing",
    "PC": "Pinnacle closing",
    "AvgC": "Closing average",
    "MaxC": "Closing best price",
}


def totals_cols(book: str) -> list[str]:
    return [f"{book}>2.5", f"{book}<2.5"]


def available_total_books(df: pd.DataFrame) -> list[str]:
    return [b for b in TOTAL_BOOKS if all(c in df.columns for c in totals_cols(b))]


def add_totals_features(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Attach p_over_<book>, margin_tot_<book>, and the realized outcome."""
    new: dict[str, pd.Series] = {}
    for book in available_total_books(df):
        odds = df[totals_cols(book)].to_numpy(float)
        valid = ~np.isnan(odds).any(axis=1)
        p_over = np.full(len(df), np.nan)
        if valid.any():
            p_over[valid] = DEMARGIN[method](odds[valid])[:, 0]
        new[f"p_over_{book}"] = pd.Series(p_over, index=df.index)
        new[f"margin_tot_{book}"] = pd.Series(
            np.where(valid, (1 / odds).sum(axis=1) - 1, np.nan), index=df.index
        )
    new["went_over"] = (df["home_goals"] + df["away_goals"]) > 2.5
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def binary_log_loss(p_over: np.ndarray, went_over: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_over, float), 1e-12, 1 - 1e-12)
    y = np.asarray(went_over, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def reliability_binary(
    df: pd.DataFrame, book: str, n_bins: int = 12
) -> pd.DataFrame:
    """Binned predicted P(over) vs realized over rate with Wilson CIs."""
    sub = df.dropna(subset=[f"p_over_{book}"])
    frame = pd.DataFrame(
        {"prob": sub[f"p_over_{book}"], "hit": sub["went_over"].astype(int)}
    )
    frame["bin"] = pd.cut(frame["prob"], bins=np.linspace(0, 1, n_bins + 1))
    grouped = frame.groupby("bin", observed=True).agg(
        n=("hit", "size"), predicted=("prob", "mean"), observed=("hit", "mean")
    )
    grouped = grouped[grouped["n"] >= 20]
    lo, hi = [], []
    for _, row in grouped.iterrows():
        ci = stats.binomtest(
            int(round(row["observed"] * row["n"])), int(row["n"])
        ).proportion_ci(0.95, method="wilson")
        lo.append(ci.low)
        hi.append(ci.high)
    grouped["ci_lo"], grouped["ci_hi"] = lo, hi
    return grouped.reset_index()


def efficiency_binary(df: pd.DataFrame, book: str) -> dict:
    """Logistic regression went_over ~ logit(p_over): slope 1 = efficient."""
    sub = df.dropna(subset=[f"p_over_{book}"])
    p = np.clip(sub[f"p_over_{book}"].to_numpy(), 1e-9, 1 - 1e-9)
    X = sm.add_constant(np.log(p / (1 - p)))
    res = sm.GLM(
        sub["went_over"].astype(int).to_numpy(), X, family=sm.families.Binomial()
    ).fit()
    return {
        "intercept": float(res.params[0]),
        "intercept_se": float(res.bse[0]),
        "slope": float(res.params[1]),
        "slope_se": float(res.bse[1]),
        "n": int(len(sub)),
    }
