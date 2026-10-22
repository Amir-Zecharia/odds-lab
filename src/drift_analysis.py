"""Do big pre-close -> close odds moves predict results that disagree
with the bookmaker consensus?

The consensus (p_cons_*) is built from pre-closing prices, so a large move
means information arrived after the consensus formed. If moves are real
information, matches with big |drift| should see the consensus favorite
lose more often and the consensus score a worse log loss — while the
closing-line favorite, which absorbed the move, should be surprised less.

"Alignment" is operationalized two ways per match:
- favorite-miss (binary): the consensus favorite did not win;
- consensus per-match log loss (continuous error).

Coverage note: the source records a single price per match before 2012-13,
so no early/closing pair exists there. Pinnacle (PS -> PSC) is the longest
series (2012-13 onward); other books pair up from 2019-20.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import config
from src.evaluate import bootstrap_ci, log_loss_per_match

RESULTS_JSON = config.PROCESSED_DIR / "drift_surprise.json"

# early prefix -> closing prefix, ordered by series length.
BOOK_PAIRS = {
    "PS": "PSC",
    "B365": "B365C",
    "WH": "WHC",
    "Avg": "AvgC",
    "Max": "MaxC",
}

SIGNIFICANT_QUANTILE = 0.90  # top decile of |drift| = "significant move"


def build_drift_frame(df: pd.DataFrame, early: str, close: str) -> pd.DataFrame:
    """One row per match with both quotes and a consensus: drift magnitude,
    consensus favorite outcome/probability, miss indicators, consensus error."""
    cols_e = [f"p_{early}_{o}" for o in config.OUTCOMES]
    cols_c = [f"p_{close}_{o}" for o in config.OUTCOMES]
    cons_cols = [f"p_cons_{o}" for o in config.OUTCOMES]
    sub = df.dropna(subset=cols_e + cols_c + cons_cols).copy()

    drifts = np.abs(sub[cols_c].to_numpy() - sub[cols_e].to_numpy())
    cons = sub[cons_cols].to_numpy()
    closing = sub[cols_c].to_numpy()
    fav_idx = cons.argmax(axis=1)
    close_fav_idx = closing.argmax(axis=1)
    outcomes = np.array(config.OUTCOMES)
    results = sub["result"].to_numpy()

    return pd.DataFrame(
        {
            "season": sub["season"].to_numpy(),
            "date": sub["date"].to_numpy(),
            "drift": drifts.max(axis=1),
            "fav_prob": cons.max(axis=1),
            "fav_missed": (outcomes[fav_idx] != results).astype(int),
            "closing_fav_missed": (outcomes[close_fav_idx] != results).astype(int),
            "cons_log_loss": log_loss_per_match(cons, sub["result"]),
        },
        index=sub.index,
    )


def miss_rate_by_bucket(frame: pd.DataFrame, n_buckets: int = 8) -> pd.DataFrame:
    """Favorite-miss rate (Wilson CI) and consensus log loss per |drift| bucket."""
    frame = frame.copy()
    frame["bucket"] = pd.qcut(frame["drift"], q=n_buckets, duplicates="drop")
    rows = []
    for bucket, grp in frame.groupby("bucket", observed=True):
        misses = int(grp["fav_missed"].sum())
        ci = stats.binomtest(misses, len(grp)).proportion_ci(0.95, method="wilson")
        ll_mean, ll_lo, ll_hi = bootstrap_ci(grp["cons_log_loss"].to_numpy())
        rows.append(
            {
                "drift_mid": float(grp["drift"].mean()),
                "n": int(len(grp)),
                "miss_rate": float(grp["fav_missed"].mean()),
                "miss_lo": float(ci.low),
                "miss_hi": float(ci.high),
                "log_loss": ll_mean,
                "ll_lo": ll_lo,
                "ll_hi": ll_hi,
                "mean_fav_prob": float(grp["fav_prob"].mean()),
            }
        )
    return pd.DataFrame(rows)


def drift_surprise_regression(frame: pd.DataFrame) -> dict:
    """Does |drift| predict consensus misses beyond favorite strength?

    Logistic fav_missed ~ drift + fav_prob (the control absorbs the
    mechanical "long shots miss more" effect), plus the analogous OLS on
    the continuous consensus log loss, HC1 robust errors.
    """
    X = sm.add_constant(frame[["drift", "fav_prob"]].to_numpy())
    logit = sm.GLM(
        frame["fav_missed"].to_numpy(), X, family=sm.families.Binomial()
    ).fit(cov_type="HC1")
    ols = sm.OLS(frame["cons_log_loss"].to_numpy(), X).fit(cov_type="HC1")
    return {
        "n": int(len(frame)),
        "logit_drift_coef": float(logit.params[1]),
        "logit_drift_z": float(logit.tvalues[1]),
        "logit_drift_p": float(logit.pvalues[1]),
        "ols_drift_coef": float(ols.params[1]),
        "ols_drift_t": float(ols.tvalues[1]),
        "ols_drift_p": float(ols.pvalues[1]),
    }


def stability_over_time(frame: pd.DataFrame, block_years: int = 3) -> pd.DataFrame:
    """The big-move-vs-rest miss-rate gap per season block.

    Blocks (rather than single seasons) keep enough top-decile matches per
    point for the gap to mean something.
    """
    frame = frame.copy()
    threshold = frame["drift"].quantile(SIGNIFICANT_QUANTILE)
    start_years = frame["season"].str[:4].astype(int)
    block_start = (start_years // block_years) * block_years
    frame["block"] = block_start.astype(str) + "-" + (block_start + block_years - 1).astype(str)

    rows = []
    for block, grp in frame.groupby("block"):
        big = grp[grp["drift"] >= threshold]
        rest = grp[grp["drift"] < threshold]
        if len(big) < 15 or len(rest) < 15:
            continue
        rows.append(
            {
                "block": block,
                "n_big": int(len(big)),
                "miss_rate_big": float(big["fav_missed"].mean()),
                "miss_rate_rest": float(rest["fav_missed"].mean()),
                "gap": float(big["fav_missed"].mean() - rest["fav_missed"].mean()),
            }
        )
    return pd.DataFrame(rows)


def consensus_vs_closing(frame: pd.DataFrame) -> dict:
    """In significant-move matches: who gets surprised, the stale consensus
    favorite or the closing-line favorite?"""
    threshold = float(frame["drift"].quantile(SIGNIFICANT_QUANTILE))
    big = frame[frame["drift"] >= threshold]
    disagree = big[big["fav_missed"] != big["closing_fav_missed"]]
    return {
        "threshold": threshold,
        "n_big_moves": int(len(big)),
        "consensus_fav_miss_rate": float(big["fav_missed"].mean()),
        "closing_fav_miss_rate": float(big["closing_fav_missed"].mean()),
        "n_favorites_disagree": int(
            (big["fav_missed"] != big["closing_fav_missed"]).sum()
        ),
        "disagreements_won_by_closing": float(
            (disagree["closing_fav_missed"] == 0).mean()
        )
        if len(disagree)
        else float("nan"),
    }


def run_all(df: pd.DataFrame) -> dict:
    """Every book pair with data; writes data/processed/drift_surprise.json."""
    out: dict = {"pairs": {}}
    for early, close in BOOK_PAIRS.items():
        frame = build_drift_frame(df, early, close)
        if len(frame) < 300:
            continue
        seasons = sorted(frame["season"].unique())
        out["pairs"][early] = {
            "closing": close,
            "n_matches": int(len(frame)),
            "seasons": f"{seasons[0]} to {seasons[-1]}",
            "regression": drift_surprise_regression(frame),
            "consensus_vs_closing": consensus_vs_closing(frame),
        }
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    data = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    if "league" in data.columns:  # this study is EPL-scoped
        data = data[data["league"] == config.DEFAULT_LEAGUE]
    results = run_all(data)
    for book, r in results["pairs"].items():
        reg, cvc = r["regression"], r["consensus_vs_closing"]
        print(
            f"{book:5s} n={r['n_matches']:5d} ({r['seasons']})  "
            f"logit drift z={reg['logit_drift_z']:+.2f} p={reg['logit_drift_p']:.2g}  "
            f"big-move miss: cons {cvc['consensus_fav_miss_rate']:.1%} "
            f"vs closing {cvc['closing_fav_miss_rate']:.1%}"
        )
    print(f"wrote {RESULTS_JSON}")
