"""Confirm or kill the Serie A lead.

The all-league battery produced one anomaly: Serie A's calibration slope
(1.103 ± 0.024) and a nominally significant out-of-sample win for the
Elo-augmented model over the raw consensus. Both were found while looking
at ~48 league × model comparisons, so before either counts as a finding
it must pass the same era-split protocol the Mismatch Explorer enforces:
does it hold separately on seasons the discovery never touched?

Discovery era: test seasons up to 2017-18. Confirmation era: 2018-19
onward. The model is refit each season either way (expanding window);
only the *evaluation* is partitioned.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import calibration, config
from src.evaluate import expanding_season_splits, log_loss_per_match, paired_bootstrap_diff
from src.models import BASELINE_FEATURES, _fit_mnlogit, _predict_mnlogit, model_dataset

RESULTS_JSON = config.PROCESSED_DIR / "seriea_lead.json"
CUTOFF_SEASON = "2017-18"
LEAGUE = "I1"


def era_verdict(
    model_losses: np.ndarray,
    base_losses: np.ndarray,
    seasons: np.ndarray,
    cutoff: str = CUTOFF_SEASON,
) -> dict:
    """Paired bootstrap of model-vs-baseline, separately per era."""
    out = {}
    for era, mask in (
        ("discovery", seasons <= cutoff),
        ("confirmation", seasons > cutoff),
    ):
        boot = paired_bootstrap_diff(model_losses[mask], base_losses[mask])
        out[era] = {
            "n": int(mask.sum()),
            "diff": boot["diff"],
            "lo": boot["lo"],
            "hi": boot["hi"],
            "significantly_better": boot["hi"] < 0,
        }
    out["confirmed"] = (
        out["discovery"]["significantly_better"]
        and out["confirmation"]["significantly_better"]
    )
    return out


def run(df: pd.DataFrame) -> dict:
    sub = df[df["league"] == LEAGUE] if "league" in df.columns else df

    # Part 1: does the slope deviation hold in both eras?
    slopes = {}
    for era, frame in (
        ("discovery", sub[sub["season"] <= CUTOFF_SEASON]),
        ("confirmation", sub[sub["season"] > CUTOFF_SEASON]),
    ):
        eff = calibration.efficiency_regression(frame)
        slopes[era] = {
            "slope": eff["slope"],
            "se": eff["slope_se"],
            "z_vs_1": eff["z_slope_vs_1"],
            "n": eff["n"],
        }
    slopes["confirmed"] = (
        slopes["discovery"]["z_vs_1"] > 2 and slopes["confirmation"]["z_vs_1"] > 2
    )

    # Part 2: does +Elo beat the raw consensus in both eras?
    data = model_dataset(sub)
    splits = expanding_season_splits(data, min_train_seasons=5)
    features = BASELINE_FEATURES + ["elo_diff"]
    chunks, test_frames = [], []
    for train_ix, test_ix in splits:
        res = _fit_mnlogit(data.loc[train_ix], features)
        chunks.append(_predict_mnlogit(res, data.loc[test_ix], features))
        test_frames.append(data.loc[test_ix])
    test_all = pd.concat(test_frames)
    elo_losses = log_loss_per_match(np.vstack(chunks), test_all["result"])
    base_losses = log_loss_per_match(
        test_all[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy(),
        test_all["result"],
    )
    elo = era_verdict(elo_losses, base_losses, test_all["season"].to_numpy())

    results = {
        "league": LEAGUE,
        "cutoff": CUTOFF_SEASON,
        "slope": slopes,
        "elo_vs_consensus": elo,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    data = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    r = run(data)
    s = r["slope"]
    print(
        f"slope: discovery {s['discovery']['slope']:.3f} "
        f"(z vs 1: {s['discovery']['z_vs_1']:+.1f}) | "
        f"confirmation {s['confirmation']['slope']:.3f} "
        f"(z vs 1: {s['confirmation']['z_vs_1']:+.1f}) | "
        f"confirmed: {s['confirmed']}"
    )
    e = r["elo_vs_consensus"]
    for era in ("discovery", "confirmation"):
        d = e[era]
        print(
            f"elo {era:12s}: diff {d['diff']:+.4f} "
            f"[{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n']:,} "
            f"better={d['significantly_better']}"
        )
    print(f"elo confirmed: {e['confirmed']}")
