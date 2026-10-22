"""Precompute the heavy analyses so the dashboard loads instantly.

Writes to data/processed/:
- ``enriched.parquet``: matches + market features + odds-free features,
  all leagues in one frame (Elo/form/rounds computed per league — team
  strength pools must not mix leagues)
- ``analysis.json``: per-league model comparison, LR tests, Dixon-Coles
  benchmark, dispersion-error regression, efficiency regressions

Run after ``python -m src.ingest``:  ``python -m src.run_analyses``
"""

from __future__ import annotations

import json
import time
import warnings

import pandas as pd

from src import calibration, config, models
from src.asian_handicap import add_ah_features
from src.evaluate import paired_bootstrap_diff
from src.features import add_features, add_rounds
from src.probabilities import add_market_features
from src.totals import add_totals_features

ENRICHED_PARQUET = config.PROCESSED_DIR / "enriched.parquet"
ANALYSIS_JSON = config.PROCESSED_DIR / "analysis.json"


def build_enriched() -> pd.DataFrame:
    df = pd.read_parquet(config.MATCHES_PARQUET)
    if "league" not in df.columns:  # pre-multi-league parquet
        df["league"] = config.DEFAULT_LEAGUE
    df = add_market_features(df)
    df = add_totals_features(df)
    df = add_ah_features(df)
    # Elo, form, rest and rounds are league-internal: a promoted team starts
    # fresh in the other league's panel rather than importing its rating.
    per_league = []
    for league, group in df.groupby("league"):
        group = add_features(group)
        group = add_rounds(group)
        per_league.append(group)
    df = pd.concat(per_league, ignore_index=True)
    df = df.sort_values(["league", "date", "home_team"]).reset_index(drop=True)
    df.to_parquet(ENRICHED_PARQUET, index=False)
    return df


def analyze_league(df: pd.DataFrame, dc_window: int | None = None) -> dict:
    """The full battery for one league's matches."""
    results: dict = {}

    results["efficiency"] = calibration.efficiency_regression(df)
    results["efficiency_by_period"] = (
        calibration.calibration_by_period(df).reset_index().to_dict("records")
    )
    results["dispersion_error"] = models.dispersion_error_regression(df)

    data = models.model_dataset(df)
    results["lr_tests"] = (
        models.likelihood_ratio_tests(data).reset_index().to_dict("records")
    )
    results["model_rows"] = int(len(data))

    summary, losses = models.out_of_sample_comparison(df)
    baseline = losses["market consensus (no fit)"]
    comparison = summary.reset_index().rename(columns={"index": "model"})
    boots = [
        paired_bootstrap_diff(losses[m], baseline)
        if m != "market consensus (no fit)"
        else {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_better": 0.5}
        for m in comparison["model"]
    ]
    comparison["diff_vs_market"] = [b["diff"] for b in boots]
    comparison["ci_lo"] = [b["lo"] for b in boots]
    comparison["ci_hi"] = [b["hi"] for b in boots]
    results["oos_comparison"] = comparison.to_dict("records")

    dc_summary, dc_losses = models.dixon_coles_benchmark(
        df, max_train_seasons=dc_window
    )
    dc = dc_summary.reset_index().rename(columns={"index": "model"})
    boot = paired_bootstrap_diff(
        dc_losses["Dixon-Coles (odds-free)"], dc_losses["market consensus"]
    )
    results["dixon_coles"] = {
        "summary": dc.to_dict("records"),
        "diff_vs_market": boot["diff"],
        "ci_lo": boot["lo"],
        "ci_hi": boot["hi"],
    }
    return results


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()

    print("enriching matches ...", flush=True)
    df = build_enriched()

    results: dict = {"leagues": {}}
    for league in config.LEAGUES:
        sub = df[df["league"] == league]
        if sub.empty:
            continue
        print(f"analyzing {league} ({len(sub):,} matches) ...", flush=True)
        results["leagues"][league] = analyze_league(sub)

    # Pooled battery across every configured league; Dixon-Coles fits one
    # combined team pool (divisions linked by promotion/relegation), with a
    # trailing 6-season window to keep the ~60k-match fits tractable.
    print(f"analyzing {config.ALL_LEAGUES} ({len(df):,} matches) ...", flush=True)
    results["leagues"][config.ALL_LEAGUES] = analyze_league(df, dc_window=6)

    results["computed_at"] = pd.Timestamp.now().isoformat()
    results["runtime_seconds"] = round(time.time() - t0, 1)
    ANALYSIS_JSON.write_text(json.dumps(results, indent=2))
    print(f"done in {results['runtime_seconds']}s -> {ANALYSIS_JSON}")


if __name__ == "__main__":
    main()
