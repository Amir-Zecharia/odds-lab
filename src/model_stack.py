"""The model of models: a stacking meta-model over every predictor we have.

Level 0 — independent knowledge sources, each producing a per-match signal:
Dixon-Coles (goal model), Elo, Pi-ratings, Massey, Colley (the rating zoo),
and optionally the market consensus itself. Level 1 — a multinomial logit
that learns, on strictly earlier seasons, how much to trust each source.

Two questions, answered separately:
- WITHOUT the market: how far can the best combination of odds-free,
  public-data models get? (The strongest odds-free predictor we can build.)
- WITH the market included as a source: does the meta-model keep any
  weight on the others, or collapse onto the market — the information
  hierarchy made visible.
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
from src.models import DixonColes, _fit_mnlogit, _predict_mnlogit
from src.ratings_zoo import add_zoo_features

RESULTS_JSON = config.PROCESSED_DIR / "model_stack.json"

MIN_TRAIN_SEASONS = 6
DC_TRAIN_WINDOW = 6

ZOO_SOURCES = ["dc_logit_H", "elo_diff", "pi_diff", "massey_diff", "colley_diff"]
MARKET_SOURCES = ["mkt_logit_H", "mkt_logit_D"]


def _standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """Scale features on TRAIN statistics only (no test leakage)."""
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1.0)
    return (train[cols] - mu) / sd, (test[cols] - mu) / sd


def add_dc_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-league, per-season Dixon-Coles home log-odds, trained on the
    trailing window of strictly earlier seasons (mirrors pricing_engine)."""
    df = df.copy()
    df["dc_logit_H"] = np.nan
    for league, group in df.groupby("league"):
        data = group.sort_values("date")
        seasons = sorted(data["season"].unique())
        for season in seasons[MIN_TRAIN_SEASONS:]:
            pool = data[data["season"] < season]
            window = sorted(pool["season"].unique())[-DC_TRAIN_WINDOW:]
            train = pool[pool["season"].isin(window)]
            test = data[data["season"] == season]
            if len(train) < 500 or test.empty:
                continue
            model = DixonColes().fit(train)
            preds = np.vstack(
                [model.outcome_probs(r.home_team, r.away_team) for r in test.itertuples()]
            )
            p = np.clip(preds, 1e-6, 1)
            df.loc[test.index, "dc_logit_H"] = np.log(p[:, 0] / p[:, 2])
    return df


def build_stack_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All level-0 signals on one frame; rows where every source exists."""
    df = add_zoo_features(df)
    df = add_dc_predictions(df)
    pH, pD, pA = (df[f"p_cons_{o}"] for o in config.OUTCOMES)
    df["mkt_logit_H"] = np.log(pH / pA)
    df["mkt_logit_D"] = np.log(pD / pA)
    needed = ZOO_SOURCES + MARKET_SOURCES + [
        f"p_cons_{o}" for o in config.OUTCOMES
    ] + ["result", "season", "date"]
    return df.dropna(subset=needed).sort_values("date").reset_index(drop=True)


def race(data: pd.DataFrame) -> dict:
    """Expanding-window race: each source alone, the odds-free stack, and
    the stack with the market inside — all against the raw consensus."""
    splits = expanding_season_splits(data, MIN_TRAIN_SEASONS)
    if not splits:
        raise ValueError("not enough seasons in the stack universe")

    contenders: dict[str, list[str]] = {
        "Dixon-Coles alone": ["dc_logit_H"],
        "Elo alone": ["elo_diff"],
        "Pi-ratings alone": ["pi_diff"],
        "Massey alone": ["massey_diff"],
        "Colley alone": ["colley_diff"],
        "STACK (odds-free zoo)": ZOO_SOURCES,
        "STACK + market": ZOO_SOURCES + MARKET_SOURCES,
    }

    preds: dict[str, list[np.ndarray]] = {name: [] for name in contenders}
    weight_history: list[dict] = []
    for train_ix, test_ix in splits:
        train, test = data.loc[train_ix], data.loc[test_ix]
        for name, cols in contenders.items():
            tr, te = _standardize(train, test, cols)
            tr, te = tr.assign(result=train["result"]), te
            res = _fit_mnlogit(tr, cols)
            preds[name].append(_predict_mnlogit(res, te, cols))
            if name == "STACK + market":
                # |coef| summed over the two logit equations per source.
                coefs = np.abs(np.asarray(res.params))[1:, :].sum(axis=1)
                total = coefs.sum() or 1.0
                weight_history.append(
                    {
                        "test_season": test["season"].iloc[0],
                        **{c: float(w / total) for c, w in zip(cols, coefs)},
                    }
                )

    test_all = data.loc[np.concatenate([list(te) for _, te in splits])]
    consensus = test_all[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy()
    losses = {"market consensus (no fit)": log_loss_per_match(consensus, test_all["result"])}
    for name, chunks in preds.items():
        losses[name] = log_loss_per_match(np.vstack(chunks), test_all["result"])

    baseline = losses["market consensus (no fit)"]
    table = []
    for name, loss in losses.items():
        boot = (
            paired_bootstrap_diff(loss, baseline)
            if name != "market consensus (no fit)"
            else {"diff": 0.0, "lo": 0.0, "hi": 0.0}
        )
        table.append(
            {
                "model": name,
                "log_loss": float(loss.mean()),
                "diff_vs_market": boot["diff"],
                "lo": boot["lo"],
                "hi": boot["hi"],
            }
        )
    table.sort(key=lambda r: r["log_loss"])

    seasons = sorted(data["season"].unique())
    return {
        "n_universe": int(len(data)),
        "n_test": int(len(test_all)),
        "test_seasons": f"{seasons[MIN_TRAIN_SEASONS]} to {seasons[-1]}",
        "race": table,
        "weights_with_market": weight_history,
    }


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    print("building level-0 signals (zoo + per-season Dixon-Coles) ...", flush=True)
    data = build_stack_frame(df)
    print(f"stack universe: {len(data):,} matches; racing ...", flush=True)
    results = {**race(data), "computed_at": pd.Timestamp.now().isoformat()}
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    for row in results["race"]:
        print(
            f"{row['model']:26s} ll={row['log_loss']:.4f} "
            f"diff={row['diff_vs_market']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"
        )
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
