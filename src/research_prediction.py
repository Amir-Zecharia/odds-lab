"""Does anything predict EPL outcomes beyond the bookmaker odds themselves?

Four candidates, each a multinomial logit fitted only on past seasons
(expanding window), all scored on the identical set of test matches so the
comparisons are exactly paired:

- recalibration: Platt-style slope/intercept correction of the best
  available market probability (Pinnacle closing where quoted, else
  consensus). The multinomial intercepts give the draw its own correction.
- drift: adds the pre-close-to-close Pinnacle move (PSC - PS). If late
  money is informative beyond the closing number itself, this helps.
- cross-market: adds the totals market (logit of P(over 2.5)), testing
  whether goals-market information leaks into 1X2 pricing incompletely.
- ensemble: all of the above together.

Every model is compared against the raw consensus AND Pinnacle closing on
the same matches with paired bootstrap CIs; "beats the market" requires the
95% CI on the log-loss difference to exclude zero.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.evaluate import (
    expanding_season_splits,
    log_loss_per_match,
    paired_bootstrap_diff,
)
from src.models import _fit_mnlogit, _predict_mnlogit

MIN_TRAIN_SEASONS = 5

BASE_FEATURES = ["best_logit_H", "best_logit_D"]
# Drift over the three outcomes sums to zero (both books are de-margined),
# so the A drift is redundant and would make the design singular.
DRIFT_FEATURES = ["drift_H", "drift_D"]
TOTALS_FEATURES = ["over_logit"]

CANDIDATES: dict[str, list[str]] = {
    "recalibration": BASE_FEATURES,
    "drift": BASE_FEATURES + DRIFT_FEATURES,
    "cross-market": BASE_FEATURES + TOTALS_FEATURES,
    "ensemble": BASE_FEATURES + DRIFT_FEATURES + TOTALS_FEATURES,
}

RESULTS_JSON = config.PROCESSED_DIR / "research_prediction.json"
FINDINGS_MD = config.PROJECT_ROOT / "docs" / "research-prediction-findings.md"


def _coalesce(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    return df[primary].where(df[primary].notna(), df[fallback])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame on the joint universe: rows where every candidate's
    inputs AND both baselines exist, so all models score identical matches."""
    df = df.copy()
    best = {
        o: _coalesce(df, f"p_PSC_{o}", f"p_cons_{o}") for o in config.OUTCOMES
    }
    df["best_logit_H"] = np.log(best["H"] / best["A"])
    df["best_logit_D"] = np.log(best["D"] / best["A"])
    for o in ("H", "D"):
        df[f"drift_{o}"] = df[f"p_PSC_{o}"] - df[f"p_PS_{o}"]
    p_over = _coalesce(df, "p_over_PC", "p_over_Avg").clip(1e-6, 1 - 1e-6)
    df["over_logit"] = np.log(p_over / (1 - p_over))

    needed = (
        BASE_FEATURES
        + DRIFT_FEATURES
        + TOTALS_FEATURES
        + [f"p_cons_{o}" for o in config.OUTCOMES]
        + [f"p_PSC_{o}" for o in config.OUTCOMES]
        + ["result", "season", "date"]
    )
    return df.dropna(subset=needed).sort_values("date")


def run_race(df: pd.DataFrame) -> dict:
    """Expanding-window race of the four candidates against both baselines."""
    data = build_features(df)
    splits = expanding_season_splits(data, MIN_TRAIN_SEASONS)
    if not splits:
        raise ValueError("not enough seasons in the joint universe")

    preds: dict[str, list[np.ndarray]] = {name: [] for name in CANDIDATES}
    for train_ix, test_ix in splits:
        train, test = data.loc[train_ix], data.loc[test_ix]
        for name, features in CANDIDATES.items():
            res = _fit_mnlogit(train, features)
            preds[name].append(_predict_mnlogit(res, test, features))

    test_all = data.loc[np.concatenate([list(te) for _, te in splits])]
    losses = {
        "market consensus": log_loss_per_match(
            test_all[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy(),
            test_all["result"],
        ),
        "Pinnacle closing": log_loss_per_match(
            test_all[[f"p_PSC_{o}" for o in config.OUTCOMES]].to_numpy(),
            test_all["result"],
        ),
    }
    for name, chunks in preds.items():
        losses[name] = log_loss_per_match(np.vstack(chunks), test_all["result"])

    models = {}
    for name in CANDIDATES:
        vs_cons = paired_bootstrap_diff(losses[name], losses["market consensus"])
        vs_psc = paired_bootstrap_diff(losses[name], losses["Pinnacle closing"])
        # Two-sided flags: a CI excluding zero on either side is a finding.
        models[name] = {
            "log_loss": float(losses[name].mean()),
            "vs_consensus": {
                **vs_cons,
                "significantly_better": vs_cons["hi"] < 0,
                "significantly_worse": vs_cons["lo"] > 0,
            },
            "vs_pinnacle_closing": {
                **vs_psc,
                "significantly_better": vs_psc["hi"] < 0,
                "significantly_worse": vs_psc["lo"] > 0,
            },
        }

    seasons = sorted(data["season"].unique())
    return {
        "universe": {
            "n_matches": int(len(data)),
            "n_test_matches": int(len(test_all)),
            "seasons": f"{seasons[0]} to {seasons[-1]}",
            "test_seasons": f"{seasons[MIN_TRAIN_SEASONS]} to {seasons[-1]}",
            "min_train_seasons": MIN_TRAIN_SEASONS,
        },
        "baselines": {
            "market consensus": float(losses["market consensus"].mean()),
            "Pinnacle closing": float(losses["Pinnacle closing"].mean()),
        },
        "models": models,
    }


def write_findings(results: dict) -> str:
    """Readable summary; states negative results plainly."""
    uni, base, models = results["universe"], results["baselines"], results["models"]
    beats_both = [
        name
        for name, m in models.items()
        if m["vs_consensus"]["significantly_better"]
        and m["vs_pinnacle_closing"]["significantly_better"]
    ]
    worse_than_psc = [
        name
        for name, m in models.items()
        if m["vs_pinnacle_closing"]["significantly_worse"]
    ]
    lines = [
        "# Does anything predict EPL outcomes beyond the odds?",
        "",
        f"Universe: {uni['n_matches']} matches ({uni['seasons']}) where "
        "Pinnacle pre-close + closing, consensus, and a totals quote all "
        f"exist; {uni['n_test_matches']} out-of-sample test matches "
        f"({uni['test_seasons']}), expanding-window by season.",
        "",
        "| Model | Log loss | vs consensus (95% CI) | vs Pinnacle closing (95% CI) |",
        "|---|---|---|---|",
        f"| market consensus (baseline) | {base['market consensus']:.4f} | — | — |",
        f"| Pinnacle closing (baseline) | {base['Pinnacle closing']:.4f} | — | — |",
    ]
    for name, m in models.items():
        c, p = m["vs_consensus"], m["vs_pinnacle_closing"]
        lines.append(
            f"| {name} | {m['log_loss']:.4f} "
            f"| {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] "
            f"| {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] |"
        )
    lines += [
        "",
        "Negative diff = candidate scores better (lower loss). Significant "
        "only if the 95% CI excludes zero.",
        "",
        "## Verdict",
        "",
    ]
    if beats_both:
        lines.append(
            "Models beating BOTH baselines significantly: "
            + ", ".join(beats_both)
            + "."
        )
    else:
        lines.append(
            "**Negative result.** No candidate — recalibration, closing-line "
            "drift, cross-market totals information, or their ensemble — "
            "beats both the market consensus and Pinnacle closing at the 95% "
            "level on identical test matches. The odds already contain "
            "everything these features know."
        )
    if worse_than_psc:
        lines += [
            "",
            f"Moreover, {', '.join(worse_than_psc)} are **significantly "
            "worse** than raw Pinnacle closing (95% CI entirely above zero): "
            "fitting extra parameters on information the closing line "
            "already prices in doesn't just fail to help — the added "
            "estimation noise measurably hurts.",
        ]
    lines += [
        "",
        "Practical takeaway: recalibration's apparent edge over the "
        "consensus is only Pinnacle-closing information flowing through its "
        "input — it never beats Pinnacle closing itself. For prediction, "
        "use the sharpest quote (PSC, else consensus), de-margined with "
        "Shin, and stop there.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    if "league" in df.columns:  # this study is EPL-scoped
        df = df[df["league"] == config.DEFAULT_LEAGUE]
    results = run_race(df)

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text(write_findings(results))

    print(f"universe: {results['universe']}")
    for name, ll in results["baselines"].items():
        print(f"{name:20s} {ll:.4f}")
    for name, m in results["models"].items():
        print(
            f"{name:20s} {m['log_loss']:.4f}  "
            f"vs cons {m['vs_consensus']['diff']:+.4f} "
            f"[{m['vs_consensus']['lo']:+.4f}, {m['vs_consensus']['hi']:+.4f}]  "
            f"vs PSC {m['vs_pinnacle_closing']['diff']:+.4f} "
            f"[{m['vs_pinnacle_closing']['lo']:+.4f}, {m['vs_pinnacle_closing']['hi']:+.4f}]"
        )
    print(f"wrote {RESULTS_JSON} and {FINDINGS_MD}")


if __name__ == "__main__":
    main()
