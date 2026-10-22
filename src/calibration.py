"""Market calibration: reliability curves and efficiency regressions.

The efficiency test regresses outcomes on the logit of implied probability:
slope ~= 1 and intercept ~= 0 indicate an efficient, well-calibrated market;
deviations at the extremes reveal the favorite-longshot bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import config


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _stack_outcomes(df: pd.DataFrame, prob_prefix: str) -> pd.DataFrame:
    """Long frame with one row per (match, outcome): prob and realized 0/1.

    Pooling the three outcomes triples the sample and covers the full
    probability range (favorites and longshots).
    """
    frames = []
    for o in config.OUTCOMES:
        col = f"{prob_prefix}_{o}"
        sub = df[[col, "result", "season", "date"]].dropna(subset=[col])
        frames.append(
            pd.DataFrame(
                {
                    "prob": sub[col],
                    "hit": (sub["result"] == o).astype(int),
                    "outcome": o,
                    "season": sub["season"],
                    "date": sub["date"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def reliability_curve(
    df: pd.DataFrame, prob_prefix: str = "p_cons", n_bins: int = 15
) -> pd.DataFrame:
    """Binned predicted vs observed frequency with Wilson 95% CIs."""
    long = _stack_outcomes(df, prob_prefix)
    long["bin"] = pd.cut(long["prob"], bins=np.linspace(0, 1, n_bins + 1))
    grouped = long.groupby("bin", observed=True).agg(
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


def efficiency_regression(df: pd.DataFrame, prob_prefix: str = "p_cons") -> dict:
    """Logistic regression hit ~ logit(prob); efficiency => slope 1, intercept 0.

    Standard errors are cluster-robust by match-date (the three stacked
    outcomes of one match are not independent).
    """
    long = _stack_outcomes(df, prob_prefix)
    X = sm.add_constant(_logit(long["prob"].to_numpy()))
    model = sm.GLM(long["hit"].to_numpy(), X, family=sm.families.Binomial())
    res = model.fit(
        cov_type="cluster", cov_kwds={"groups": long["date"].to_numpy()}
    )
    intercept, slope = res.params
    se_i, se_s = res.bse
    return {
        "intercept": float(intercept),
        "intercept_se": float(se_i),
        "slope": float(slope),
        "slope_se": float(se_s),
        # Wald tests against the efficient-market values (0, 1).
        "z_intercept_vs_0": float(intercept / se_i),
        "z_slope_vs_1": float((slope - 1.0) / se_s),
        "n": int(len(long)),
    }


def calibration_by_period(
    df: pd.DataFrame,
    prob_prefix: str = "p_cons",
    split_month: int = 12,
) -> pd.DataFrame:
    """Efficiency regression early vs late season.

    Early = August through ``split_month`` (December), late = January on.
    """
    early_mask = df["date"].dt.month.between(8, split_month)
    rows = []
    for label, sub in (("early", df[early_mask]), ("late", df[~early_mask])):
        r = efficiency_regression(sub, prob_prefix)
        r["period"] = label
        rows.append(r)
    return pd.DataFrame(rows).set_index("period")


def home_advantage_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Realized home-win rate vs market-expected home probability per season.

    The 2020-21 empty-stadium season is the natural experiment: realized
    home advantage dropped while the market only partially adjusted.
    """
    out = df.groupby("season").agg(
        matches=("result", "size"),
        home_win_rate=("result", lambda s: (s == "H").mean()),
        market_home_prob=("p_cons_H", "mean"),
        home_goal_diff=("home_goals", "mean"),
    )
    away_goals = df.groupby("season")["away_goals"].mean()
    out["home_goal_diff"] = out["home_goal_diff"] - away_goals
    out["excess_home_wins"] = out["home_win_rate"] - out["market_home_prob"]
    return out


def draw_pricing_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Market draw probability vs realized draw rate per season."""
    return df.groupby("season").agg(
        market_draw_prob=("p_cons_D", "mean"),
        realized_draw_rate=("result", lambda s: (s == "D").mean()),
    )
