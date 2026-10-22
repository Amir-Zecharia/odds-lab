"""Do bookmakers change their odds because of other bookmakers?

Resolution caveat, stated wherever these results appear: the source
provides two snapshots per book (early-week and closing), not intraday
ticks. "A leads B" here therefore means *A's earlier prices contain
information B's closing prices later incorporate* — the standard price-
discovery notion — not literal minute-by-minute reaction.

Three instruments, all on logit home-win probabilities:

- convergence: a book's early→close move regressed on its early deviation
  from the cross-book consensus. Coefficient 1 = fully corrects toward the
  pack (herding); 0 = ignores the pack.
- leadership: for an ordered pair (A, B), does A's early price explain B's
  closing price beyond B's own early price? The cross-coefficient's z is
  the arrow strength in the who-leads-whom matrix.
- follow-the-sharp: how much of a soft book's move is explained by where
  Pinnacle already stood at the early snapshot.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config

RESULTS_JSON = config.PROCESSED_DIR / "influence.json"

# Books with an early/closing pair in the source.
PAIRED_BOOKS = {"PS": "PSC", "B365": "B365C", "WH": "WHC"}


def _logit(p: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _pair_frame(df: pd.DataFrame, books: list[str]) -> pd.DataFrame:
    """Matches where every requested book has both snapshots (home prob)."""
    needed = ["p_cons_H"]
    for b in books:
        needed += [f"p_{b}_H", f"p_{PAIRED_BOOKS[b]}_H"]
    sub = df.dropna(subset=needed).copy()
    out = pd.DataFrame(index=sub.index)
    out["league"] = sub.get("league", config.DEFAULT_LEAGUE)
    out["season"] = sub["season"].to_numpy()
    out["cons_early"] = _logit(sub["p_cons_H"])
    for b in books:
        out[f"{b}_early"] = _logit(sub[f"p_{b}_H"])
        out[f"{b}_close"] = _logit(sub[f"p_{PAIRED_BOOKS[b]}_H"])
        out[f"{b}_move"] = out[f"{b}_close"] - out[f"{b}_early"]
    return out


def convergence_regression(df: pd.DataFrame, book: str) -> dict:
    """move_b ~ (consensus_early − book_early): herding toward the pack."""
    frame = _pair_frame(df, [book])
    deviation = frame["cons_early"] - frame[f"{book}_early"]
    X = sm.add_constant(deviation.to_numpy())
    res = sm.OLS(frame[f"{book}_move"].to_numpy(), X).fit(cov_type="HC1")
    return {
        "book": book,
        "n": int(len(frame)),
        "herding_coef": float(res.params[1]),
        "se": float(res.bse[1]),
        "t": float(res.tvalues[1]),
        "r_squared": float(res.rsquared),
    }


def leadership_matrix(df: pd.DataFrame, books: list[str] | None = None) -> pd.DataFrame:
    """z-scores for 'A's early price explains B's close beyond B's early'."""
    books = books or list(PAIRED_BOOKS)
    frame = _pair_frame(df, books)
    rows = []
    for leader in books:
        for follower in books:
            if leader == follower:
                continue
            X = sm.add_constant(
                np.column_stack(
                    [frame[f"{follower}_early"], frame[f"{leader}_early"]]
                )
            )
            res = sm.OLS(frame[f"{follower}_close"].to_numpy(), X).fit(
                cov_type="HC1"
            )
            rows.append(
                {
                    "leader": leader,
                    "follower": follower,
                    "coef": float(res.params[2]),
                    "z": float(res.tvalues[2]),
                    "n": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def follow_the_sharp(df: pd.DataFrame, soft_book: str, sharp: str = "PS") -> dict:
    """Decompose the soft book's move: own inertia vs the sharp book's
    early position. Share of move variance explained by the sharp gap."""
    frame = _pair_frame(df, [soft_book, sharp])
    gap = frame[f"{sharp}_early"] - frame[f"{soft_book}_early"]
    X = sm.add_constant(gap.to_numpy())
    res = sm.OLS(frame[f"{soft_book}_move"].to_numpy(), X).fit(cov_type="HC1")
    # Where does the soft close land relative to its own early price and
    # the sharp book's early price?
    d_own = (frame[f"{soft_book}_close"] - frame[f"{soft_book}_early"]).abs()
    d_sharp = (frame[f"{soft_book}_close"] - frame[f"{sharp}_early"]).abs()
    return {
        "soft_book": soft_book,
        "n": int(len(frame)),
        "gap_coef": float(res.params[1]),
        "gap_t": float(res.tvalues[1]),
        "move_r_squared": float(res.rsquared),
        "closes_nearer_sharp_early": float((d_sharp < d_own).mean()),
    }


def run(df: pd.DataFrame) -> dict:
    results = {
        "convergence": [
            convergence_regression(df, b) for b in PAIRED_BOOKS
        ],
        "leadership": leadership_matrix(df).to_dict("records"),
        "follow_the_sharp": [
            follow_the_sharp(df, b) for b in PAIRED_BOOKS if b != "PS"
        ],
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    data = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    out = run(data)
    for c in out["convergence"]:
        print(
            f"convergence {c['book']:5s} coef={c['herding_coef']:+.2f} "
            f"(t={c['t']:+.1f}, n={c['n']:,})"
        )
    for row in out["leadership"]:
        print(
            f"lead {row['leader']:5s} -> {row['follower']:5s} "
            f"coef={row['coef']:+.2f} z={row['z']:+.1f}"
        )
    for f in out["follow_the_sharp"]:
        print(
            f"{f['soft_book']} closes nearer Pinnacle's EARLY price in "
            f"{f['closes_nearer_sharp_early']:.0%} of matches"
        )
