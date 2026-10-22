"""The Asian-handicap market: the cleanest test of odds efficiency.

The over/under market is binary but its 2.5 line sits near a coin flip, so
there is little probability range to calibrate against. Asian handicaps
remove the draw entirely — the bookmaker chooses the line so the two-way
price hovers near even money at *any* matchup strength — which spreads the
de-margined P(home covers) across the whole 0-1 range. That makes AH the
sharpest lens on "is the market well calibrated?": no third outcome to
absorb error, and a line deliberately centred on the true 50/50 point.

Quarter lines (e.g. -0.75) are settled as two equal half-stakes at the two
neighbouring half/full lines, yielding settlement fractions in
{-1, -0.5, 0, +0.5, +1}. Mirrors src/totals.py: implied probabilities,
Shin de-margining (outcome-count agnostic), calibration, per-book scoring.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src import config
from src.evaluate import bootstrap_ci
from src.probabilities import DEMARGIN, overround
from src.totals import binary_log_loss

# Books quoting Asian handicaps in the source (Pinnacle is "P", not "PS").
AH_BOOKS = ["B365", "P", "Avg", "Max", "B365C", "PC", "AvgC", "MaxC"]

AH_BOOK_NAMES = {
    "B365": "Bet365",
    "P": "Pinnacle",
    "Avg": "Market average",
    "Max": "Best price",
    "B365C": "Bet365 closing",
    "PC": "Pinnacle closing",
    "AvgC": "Closing average",
    "MaxC": "Closing best price",
}


def ah_price_cols(book: str) -> list[str]:
    """Two-way [home, away] Asian-handicap price columns for one book."""
    return [f"{book}AHH", f"{book}AHA"]


def _is_closing(book: str) -> bool:
    return book.endswith("C")


def _truth_col(book: str) -> str:
    """Closing books settle at the closing line; openers at the opening line."""
    return "ah_covered_close" if _is_closing(book) else "ah_covered"


def _settle_half_line(goal_diff: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Settle a single (half/full) line: win/push/loss -> +1/0/-1."""
    margin = goal_diff + line
    return np.where(margin > 0, 1.0, np.where(margin < 0, -1.0, 0.0))


def settle_home_ah_array(goal_diff, line) -> np.ndarray:
    """Vectorised home-side settlement fraction for Asian lines.

    Returns the profit per 1 unit staked *at even terms* — i.e. the cover
    fraction in {-1, -0.5, 0, +0.5, +1}. A quarter line (odd multiple of
    0.25) splits into two half-stakes at ``line ± 0.25`` (both half/full
    lines) and averages their outcomes; the book's actual price is applied
    separately when a realised bet profit is needed.
    """
    gd = np.asarray(goal_diff, dtype=float)
    ln = np.asarray(line, dtype=float)
    is_quarter = np.mod(np.round(ln * 4), 2) != 0
    half = _settle_half_line(gd, ln)
    quarter = 0.5 * (
        _settle_half_line(gd, ln - 0.25) + _settle_half_line(gd, ln + 0.25)
    )
    out = np.where(is_quarter, quarter, half)
    return np.where(np.isnan(gd) | np.isnan(ln), np.nan, out)


def settle_home_ah(goal_diff: float, line: float) -> float:
    """Scalar home-side settlement fraction (see ``settle_home_ah_array``)."""
    return float(settle_home_ah_array(goal_diff, line))


def p_home_cover(df: pd.DataFrame, book: str, method: str = "shin") -> pd.Series:
    """De-margin the two-way [home, away] AH price -> P(home covers).

    The de-margining is outcome-count agnostic, so the same Shin/power/
    proportional machinery used for 1X2 and totals applies to the two-way
    handicap market. NaN rows (missing quotes) are preserved.
    """
    odds = df[ah_price_cols(book)].to_numpy(dtype=float)
    valid = ~np.isnan(odds).any(axis=1)
    p = np.full(len(df), np.nan)
    if valid.any():
        p[valid] = DEMARGIN[method](odds[valid])[:, 0]
    return pd.Series(p, index=df.index)


def add_ah_features(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Attach handicap line, settlement, cover target and P(cover) per book.

    Adds ``ah_line``/``ah_result``/``ah_covered`` (opening line = ``AHh``)
    and their ``_close`` variants (closing line = ``AHCh``), plus
    ``p_ahcover_<book>`` for every book with prices present. Cleanly no-ops
    when the raw AH columns are absent (e.g. an enriched frame built before
    ingest learned to keep them): the truth columns come through as NaN and
    no P(cover) columns are added. Uses pd.concat to avoid the fragmentation
    warning add_market_features guards against.
    """
    new: dict[str, pd.Series] = {}
    goal_diff = (df["home_goals"] - df["away_goals"]).to_numpy(dtype=float)

    for line_col, suffix in (("AHh", ""), ("AHCh", "_close")):
        if line_col in df.columns:
            line = df[line_col].to_numpy(dtype=float)
            result = settle_home_ah_array(goal_diff, line)
        else:
            line = np.full(len(df), np.nan)
            result = line
        new[f"ah_line{suffix}"] = pd.Series(line, index=df.index)
        new[f"ah_result{suffix}"] = pd.Series(result, index=df.index)
        # Binary cover target: win -> 1, loss -> 0, exact push -> NaN (dropped
        # by calibration/scoring). Half-wins/half-losses count as win/loss.
        covered = np.where(
            np.isnan(result), np.nan, np.where(result > 0, 1.0, np.where(result < 0, 0.0, np.nan))
        )
        new[f"ah_covered{suffix}"] = pd.Series(covered, index=df.index)

    for book in AH_BOOKS:
        if all(c in df.columns for c in ah_price_cols(book)):
            new[f"p_ahcover_{book}"] = p_home_cover(df, book, method)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def available_ah_books(df: pd.DataFrame) -> list[str]:
    """Books whose P(home cover) column is present with at least one quote."""
    return [
        b for b in AH_BOOKS
        if f"p_ahcover_{b}" in df.columns and df[f"p_ahcover_{b}"].notna().any()
    ]


def reliability_ah(df: pd.DataFrame, book: str, n_bins: int = 10) -> pd.DataFrame:
    """Binned predicted P(home cover) vs realized cover rate with Wilson CIs.

    Pushes are already NaN in the cover column and dropped here, so bins hold
    only decisive bets.
    """
    truth = _truth_col(book)
    sub = df.dropna(subset=[f"p_ahcover_{book}", truth])
    frame = pd.DataFrame(
        {"prob": sub[f"p_ahcover_{book}"], "hit": sub[truth].astype(int)}
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


def efficiency_ah(df: pd.DataFrame, book: str) -> dict:
    """Logistic regression covered ~ logit(P cover): slope 1 = efficient.

    A slope of 1 with intercept 0 means the de-margined price is the true
    probability that the home side covers; deviations quantify miscalibration.
    """
    truth = _truth_col(book)
    sub = df.dropna(subset=[f"p_ahcover_{book}", truth])
    p = np.clip(sub[f"p_ahcover_{book}"].to_numpy(), 1e-9, 1 - 1e-9)
    X = sm.add_constant(np.log(p / (1 - p)))
    res = sm.GLM(
        sub[truth].astype(int).to_numpy(), X, family=sm.families.Binomial()
    ).fit()
    return {
        "intercept": float(res.params[0]),
        "intercept_se": float(res.bse[0]),
        "slope": float(res.params[1]),
        "slope_se": float(res.bse[1]),
        "n": int(len(sub)),
    }


def ah_margin(df: pd.DataFrame, book: str) -> pd.Series:
    """Two-way overround of a book's AH price (sum of 1/odds minus 1)."""
    odds = df[ah_price_cols(book)].to_numpy(dtype=float)
    valid = ~np.isnan(odds).any(axis=1)
    return pd.Series(
        np.where(valid, overround(odds), np.nan), index=df.index
    )


def leaderboard_ah(df: pd.DataFrame, books: list[str]) -> pd.DataFrame:
    """Binary log loss per book on the subset every book quotes (pushes out)."""
    common = df.dropna(subset=[f"p_ahcover_{b}" for b in books])
    rows = []
    for b in books:
        truth = _truth_col(b)
        sub = common.dropna(subset=[truth])
        ll = binary_log_loss(
            sub[f"p_ahcover_{b}"].to_numpy(), sub[truth].to_numpy()
        )
        mean, lo, hi = bootstrap_ci(ll)
        rows.append(
            {
                "book": b,
                "name": AH_BOOK_NAMES[b],
                "log_loss": mean,
                "lo": lo,
                "hi": hi,
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


RESULTS_JSON = config.PROCESSED_DIR / "asian_handicap.json"


def run(df: pd.DataFrame) -> dict:
    """Per-book scoring + calibration and per-league efficiency for the JSON."""
    if "p_ahcover_Avg" not in df.columns and "ah_result" not in df.columns:
        df = add_ah_features(df)
    books = available_ah_books(df)
    covered = df.dropna(subset=["p_ahcover_Avg"]) if "p_ahcover_Avg" in df.columns else df

    board = leaderboard_ah(df, books) if books else pd.DataFrame()
    per_book: dict = {}
    for b in books:
        eff = efficiency_ah(df, b)
        row = board[board["book"] == b].iloc[0]
        per_book[b] = {
            "name": AH_BOOK_NAMES[b],
            "log_loss": row["log_loss"],
            "lo": row["lo"],
            "hi": row["hi"],
            "n": eff["n"],
            "slope": eff["slope"],
            "slope_se": eff["slope_se"],
            "intercept": eff["intercept"],
            "intercept_se": eff["intercept_se"],
            "mean_margin": float(ah_margin(df, b).mean()),
        }

    # Primary lens: market average opening line, the deepest-coverage book.
    primary = "Avg" if "Avg" in per_book else (books[0] if books else None)
    by_league: dict = {}
    if primary and "league" in df.columns:
        for league, group in df.groupby("league"):
            sub = group.dropna(subset=[f"p_ahcover_{primary}", _truth_col(primary)])
            if len(sub) >= 500:
                by_league[league] = efficiency_ah(group, primary)["slope"]

    return {
        "n_matches": int(len(df)),
        "coverage": int(len(covered)),
        "home_cover_rate": (
            float(covered["ah_covered"].mean()) if "ah_covered" in covered else float("nan")
        ),
        "primary_book": primary,
        "books": per_book,
        "efficiency_by_league": by_league,
        "computed_at": pd.Timestamp.now().isoformat(),
    }


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    results = run(df)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    p = results["primary_book"]
    if p:
        b = results["books"][p]
        print(
            f"AH efficiency ({b['name']}): slope {b['slope']:.3f} "
            f"± {b['slope_se']:.3f}, intercept {b['intercept']:+.3f}, "
            f"log loss {b['log_loss']:.4f} on {b['n']:,} bets "
            f"(coverage {results['coverage']:,} matches) -> {RESULTS_JSON}"
        )
    else:
        print(f"No AH odds present in enriched.parquet -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
