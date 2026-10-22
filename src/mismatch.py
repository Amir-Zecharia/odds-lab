"""The mismatch microscope: where do odds and reality actually disagree?

Global calibration is near-perfect (slope ~1.02 across 60k matches), but
that average could hide *pockets* — segments of matches where implied
probabilities systematically miss. Hunting pockets across many segments
invites data dredging: scan 500 cells and ~25 will look "significant" at
5% by luck. Two defenses, both mandatory here:

1. **Benjamini-Hochberg FDR** across every scanned cell, so the expected
   share of false discoveries among flagged cells is controlled at 10%.
2. **Era-split confirmation**: pockets are *discovered* on the first half
   of the data and only count as findings if they replicate with the same
   sign (p < 0.05) on the untouched second half.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config

RESULTS_JSON = config.PROCESSED_DIR / "mismatch_pockets.json"

DISCOVERY_LAST_SEASON = "2017-18"
FDR_LEVEL = 0.10
MIN_CELL_N = 200

PROB_BANDS = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
BAND_LABELS = [
    "<10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-65%", "65-80%", "80%+",
]

OUTCOME_NAMES = {"H": "home win", "D": "draw", "A": "away win"}

# Dimensions available to the scanner and the interactive page.
DIMENSIONS = ["league", "outcome", "band", "era", "phase"]


def stacked_universe(df: pd.DataFrame, source: str = "cons") -> pd.DataFrame:
    """One row per (match, outcome): implied probability vs what happened.

    Carries every segmentation dimension: league, season, era (3-season
    block), round phase, probability band, favorite flag.
    """
    cols = [f"p_{source}_{o}" for o in config.OUTCOMES]
    sub = df.dropna(subset=cols + ["result"]).copy()
    if "league" not in sub.columns:
        sub["league"] = config.DEFAULT_LEAGUE

    frames = []
    fav_idx = sub[cols].to_numpy().argmax(axis=1)
    for i, o in enumerate(config.OUTCOMES):
        frames.append(
            pd.DataFrame(
                {
                    "league": sub["league"].to_numpy(),
                    "season": sub["season"].to_numpy(),
                    "round": sub["round"].to_numpy()
                    if "round" in sub.columns
                    else np.nan,
                    "outcome": OUTCOME_NAMES[o],
                    "prob": sub[cols[i]].to_numpy(),
                    "hit": (sub["result"] == o).to_numpy().astype(int),
                    "is_favorite": fav_idx == i,
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)

    start_year = long["season"].str[:4].astype(int)
    block = (start_year // 3) * 3
    long["era"] = block.astype(str) + "-" + (block + 2).astype(str)
    long["band"] = pd.cut(
        long["prob"], bins=PROB_BANDS, labels=BAND_LABELS, include_lowest=True
    )
    long["phase"] = np.select(
        [long["round"] <= 10, long["round"] >= 29],
        ["early season", "late season"],
        default="mid season",
    )
    return long


def scan_segments(
    long: pd.DataFrame,
    dims: list[str],
    min_n: int = MIN_CELL_N,
    fdr_level: float = FDR_LEVEL,
) -> pd.DataFrame:
    """Edge (realized − implied) per segment cell, with FDR-adjusted flags.

    The binomial test treats each (match, outcome) row's implied probability
    as its own null; the cell statistic compares total hits against the sum
    of implied probabilities (a Poisson-binomial, approximated by its normal
    limit — fine at the enforced minimum cell size).
    """
    grouped = long.groupby(dims, observed=True)
    rows = []
    for key, grp in grouped:
        if len(grp) < min_n:
            continue
        implied = grp["prob"].to_numpy()
        hits = int(grp["hit"].sum())
        expected = float(implied.sum())
        variance = float((implied * (1 - implied)).sum())
        if variance <= 0:
            continue
        z = (hits - expected) / np.sqrt(variance)
        cell = {
            dim: (key[i] if isinstance(key, tuple) else key)
            for i, dim in enumerate(dims)
        }
        cell.update(
            {
                "n": int(len(grp)),
                "implied": float(implied.mean()),
                "realized": float(grp["hit"].mean()),
                "edge": float(grp["hit"].mean() - implied.mean()),
                "z": float(z),
                "p_value": float(2 * stats.norm.sf(abs(z))),
            }
        )
        rows.append(cell)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["fdr_significant"] = benjamini_hochberg(
        table["p_value"].to_numpy(), fdr_level
    )
    return table.sort_values("p_value").reset_index(drop=True)


def benjamini_hochberg(p_values: np.ndarray, level: float) -> np.ndarray:
    """Boolean mask of discoveries under BH false-discovery-rate control."""
    m = len(p_values)
    order = np.argsort(p_values)
    thresholds = level * (np.arange(1, m + 1) / m)
    passed = p_values[order] <= thresholds
    cutoff = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:cutoff]] = True
    return mask


DISCOVERY_GRID = [
    ["outcome", "band"],
    ["league", "outcome"],
    ["league", "band"],
    ["outcome", "phase"],
    ["league", "outcome", "band"],
]


def discover_and_confirm(
    long: pd.DataFrame, discovery_last: str = DISCOVERY_LAST_SEASON
) -> dict:
    """Two-era protocol: dredge freely in the past, then make each pocket
    prove itself on data it has never seen."""
    discovery = long[long["season"] <= discovery_last]
    confirmation = long[long["season"] > discovery_last]

    confirmed, candidates = [], 0
    for dims in DISCOVERY_GRID:
        found = scan_segments(discovery, dims)
        if found.empty:
            continue
        for _, row in found[found["fdr_significant"]].iterrows():
            candidates += 1
            mask = np.ones(len(confirmation), dtype=bool)
            for dim in dims:
                mask &= (confirmation[dim] == row[dim]).to_numpy()
            cell = confirmation[mask]
            check = scan_segments(cell.assign(_all="x"), ["_all"], min_n=100)
            if check.empty:
                continue
            c = check.iloc[0]
            if c["p_value"] < 0.05 and np.sign(c["edge"]) == np.sign(row["edge"]):
                confirmed.append(
                    {
                        "segment": {d: str(row[d]) for d in dims},
                        "discovery": {
                            "n": int(row["n"]),
                            "edge": row["edge"],
                            "p_value": row["p_value"],
                        },
                        "confirmation": {
                            "n": int(c["n"]),
                            "edge": float(c["edge"]),
                            "p_value": float(c["p_value"]),
                        },
                    }
                )
    return {
        "discovery_through": discovery_last,
        "candidates_after_fdr": candidates,
        "confirmed_pockets": confirmed,
        "n_discovery_rows": int(len(discovery)),
        "n_confirmation_rows": int(len(confirmation)),
    }


def run(df: pd.DataFrame, source: str = "cons") -> dict:
    long = stacked_universe(df, source)
    results = {"source": source, **discover_and_confirm(long)}
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    data = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    out = run(data)
    print(
        f"{out['candidates_after_fdr']} FDR-significant candidates in "
        f"discovery, {len(out['confirmed_pockets'])} confirmed out-of-era"
    )
    for pocket in out["confirmed_pockets"]:
        seg = ", ".join(f"{k}={v}" for k, v in pocket["segment"].items())
        print(
            f"  {seg}: discovery edge {pocket['discovery']['edge']:+.3f}, "
            f"confirmed edge {pocket['confirmation']['edge']:+.3f} "
            f"(p={pocket['confirmation']['p_value']:.3g}, "
            f"n={pocket['confirmation']['n']})"
        )
