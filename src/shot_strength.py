"""Poor man's xG: is shot volume a better strength signal than goals?

The modern industry standard is to rate teams on expected goals rather
than goals, because finishing is noisy — a team that creates ten good
chances and scores one was unlucky, not weak, and a goals-based model
mistakes that luck for weakness. We have no xG feed, but the source has
carried **shots on target** since 2000, which is the same idea at lower
resolution.

The test is head-to-head and honest: fit the identical Dixon-Coles
machinery twice — once on goals, once on shots on target converted to
expected goals at the league conversion rate — and ask which better
predicts the *actual goals* that follow, out of sample.
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
from src.models import DixonColes

RESULTS_JSON = config.PROCESSED_DIR / "shot_strength.json"

SHOT_COLUMNS = ("home_shots_target", "away_shots_target")
GOAL_COLUMNS = ("home_goals", "away_goals")


def conversion_rates(df: pd.DataFrame) -> dict[str, float]:
    """League goals per shot on target, home and away separately."""
    usable = df.dropna(subset=list(SHOT_COLUMNS) + list(GOAL_COLUMNS))
    rates = {}
    for side, goal_col, shot_col in (
        ("home", "home_goals", "home_shots_target"),
        ("away", "away_goals", "away_shots_target"),
    ):
        shots = usable[shot_col].sum()
        rates[side] = float(usable[goal_col].sum() / shots) if shots else 0.0
    return rates


def shots_as_goals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace goals with shot-implied expected goals.

    Dixon-Coles wants integer goal counts, so the converted values are
    rounded — coarse, but it keeps the two models structurally identical
    so the comparison isolates the *signal*, not the machinery.
    """
    usable = df.dropna(subset=list(SHOT_COLUMNS) + list(GOAL_COLUMNS)).copy()
    rates = conversion_rates(usable)
    usable["home_goals"] = (
        (usable["home_shots_target"] * rates["home"]).round().astype(int)
    )
    usable["away_goals"] = (
        (usable["away_shots_target"] * rates["away"]).round().astype(int)
    )
    usable["result"] = np.select(
        [usable["home_goals"] > usable["away_goals"],
         usable["home_goals"] < usable["away_goals"]],
        ["H", "A"], default="D",
    )
    return usable


def fit_shot_model(matches: pd.DataFrame) -> DixonColes:
    """Dixon-Coles fitted on shot-implied goals rather than actual goals."""
    return DixonColes().fit(shots_as_goals(matches))


def shot_based_goal_probs(model: DixonColes, home: str, away: str) -> tuple[float, float]:
    """Expected shot-implied goals for each side (diagnostic helper)."""
    grid = model.score_grid(home, away)
    goals = np.arange(grid.shape[0])
    return float((grid.sum(axis=1) * goals).sum()), float((grid.sum(axis=0) * goals).sum())


def race_shots_vs_goals(
    df: pd.DataFrame,
    min_train_seasons: int = 5,
    max_train_seasons: int | None = 6,
) -> dict:
    """Expanding-window head-to-head, scored on the goals that happened.

    Both models are refit before each test season on the same training
    matches and judged on identical test matches, so the only difference
    is which signal the team strengths were learned from.

    ``max_train_seasons`` caps training to a trailing window: team
    strength from a decade ago describes a squad that no longer exists,
    and the cap keeps two full Dixon-Coles fits per season tractable.
    """
    usable = df.dropna(subset=list(SHOT_COLUMNS) + list(GOAL_COLUMNS))
    usable = usable.sort_values("date").reset_index(drop=True)
    splits = expanding_season_splits(usable, min_train_seasons)
    if not splits:
        raise ValueError("not enough seasons with shot data")

    shot_preds, goal_preds, tests = [], [], []
    for train_ix, test_ix in splits:
        train, test = usable.loc[train_ix], usable.loc[test_ix]
        if max_train_seasons is not None:
            window = sorted(train["season"].unique())[-max_train_seasons:]
            train = train[train["season"].isin(window)]
        if len(train) < 200 or test.empty:
            continue
        shot_model = fit_shot_model(train)
        goal_model = DixonColes().fit(train)
        shot_preds.append(
            np.vstack([shot_model.outcome_probs(r.home_team, r.away_team)
                       for r in test.itertuples()])
        )
        goal_preds.append(
            np.vstack([goal_model.outcome_probs(r.home_team, r.away_team)
                       for r in test.itertuples()])
        )
        tests.append(test)

    test_all = pd.concat(tests)
    truth = test_all["result"]
    shot_loss = log_loss_per_match(np.vstack(shot_preds), truth)
    goal_loss = log_loss_per_match(np.vstack(goal_preds), truth)
    boot = paired_bootstrap_diff(shot_loss, goal_loss)

    out = {
        "n_test": int(len(test_all)),
        "shots_log_loss": float(shot_loss.mean()),
        "goals_log_loss": float(goal_loss.mean()),
        "diff": boot["diff"],
        "lo": boot["lo"],
        "hi": boot["hi"],
        "shots_better": bool(boot["hi"] < 0),
        "conversion": conversion_rates(usable),
    }
    if all(f"p_cons_{o}" in test_all.columns for o in config.OUTCOMES):
        market = test_all[[f"p_cons_{o}" for o in config.OUTCOMES]]
        valid = market.notna().all(axis=1).to_numpy()
        if valid.any():
            market_loss = log_loss_per_match(
                market[valid].to_numpy(), truth[valid]
            )
            out["market_log_loss"] = float(market_loss.mean())
            out["shots_vs_market"] = paired_bootstrap_diff(
                shot_loss[valid], market_loss
            )["diff"]
    return out


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    results = {"leagues": {}}
    # Per league only: Dixon-Coles rates teams against the opponents they
    # actually face, so pooling divisions would fit strengths across teams
    # that never meet.
    for league in sorted(df["league"].unique()):
        sub = df[df["league"] == league]
        try:
            results["leagues"][league] = race_shots_vs_goals(sub)
        except (ValueError, KeyError):
            continue
        r = results["leagues"][league]
        print(
            f"  {league}: shots {r['shots_log_loss']:.4f} vs goals "
            f"{r['goals_log_loss']:.4f} (n={r['n_test']:,})", flush=True
        )
        # Checkpoint after every league: these fits are slow enough that
        # losing them to an interrupted run is worse than a re-write.
        results["computed_at"] = pd.Timestamp.now().isoformat()
        RESULTS_JSON.write_text(json.dumps(results, indent=2))

    for name, r in results["leagues"].items():
        verdict = "shots WIN" if r["shots_better"] else "goals hold"
        print(
            f"{name:7s} n={r['n_test']:6,} shots {r['shots_log_loss']:.4f} vs "
            f"goals {r['goals_log_loss']:.4f}  diff {r['diff']:+.4f} "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {verdict}"
        )
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
