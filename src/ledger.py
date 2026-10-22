"""The prediction ledger: forward testing that hindsight cannot touch.

Every other study on this platform is retrospective — refit on history,
evaluated on history. However carefully that is done (and it is done with
time-ordered splits and bootstrap CIs), a backtest can always be quietly
contaminated by choices made after seeing the data.

This module removes that objection. Predictions are written to disk
**before kickoff**, frozen the moment the match starts, and scored later
against the result. No refit can reach back; no fixture can be dropped
after the fact. What accumulates is the only evidence about this market
that is immune to researcher degrees of freedom.

Two verbs:

    ODDS_API_KEY=<key> python -m src.ledger record   # before matchday
    python -m src.ledger settle                      # after results land

The recipe recorded is the one the platform validated: the sharpest
available quote (Pinnacle, else the cross-book average), Shin de-margined.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src import config
from src.evaluate import bootstrap_ci, brier_per_match, log_loss_per_match

LEDGER_PARQUET = config.PROCESSED_DIR / "ledger.parquet"

# Bump when the prediction recipe changes, so the scoreboard can be split
# by model version rather than silently mixing methodologies.
MODEL_VERSION = "sharpest-quote-shin-v1"

# The backtest log loss of the market consensus over 32,780 matches
# (src/model_stack.py). The live ledger's honest reference point.
BACKTEST_BASELINE = 0.9832

OPEN_COLUMNS = [
    "prediction_id", "recorded_at", "kickoff", "league", "home_team",
    "away_team", "p_home", "p_draw", "p_away", "source", "books",
    "model_version",
]
SETTLED_COLUMNS = ["result", "log_loss", "brier", "settled_at"]


def _prediction_id(kickoff: pd.Timestamp, home: str, away: str) -> str:
    """Stable identity for a fixture: one prediction per match, ever."""
    return f"{pd.Timestamp(kickoff).date()}|{home}|{away}"


def empty_ledger() -> pd.DataFrame:
    frame = pd.DataFrame(columns=OPEN_COLUMNS + SETTLED_COLUMNS)
    for col in ("kickoff", "recorded_at", "settled_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def load_ledger(path=LEDGER_PARQUET) -> pd.DataFrame:
    if not path.exists():
        return empty_ledger()
    return pd.read_parquet(path)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def append_predictions(
    ledger: pd.DataFrame,
    predictions: pd.DataFrame,
    league: str = "E0",
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge fresh predictions into the ledger under the freeze rule.

    A fixture whose kickoff has passed is **immutable** — its recorded
    probabilities are the forecast we are being judged on, so a later
    re-run can never revise them. Fixtures still in the future may be
    updated (odds move; the latest pre-kickoff price is the honest
    forecast). New fixtures are appended.
    """
    now = now or _now()
    if predictions.empty:
        return ledger, {"added": 0, "updated": 0, "frozen": 0}

    fresh = predictions.copy()
    fresh["kickoff"] = pd.to_datetime(fresh["kickoff"], utc=True)
    if fresh["kickoff"].isna().any():
        raise ValueError("Prediction kickoff must be a valid timestamp")
    fresh["prediction_id"] = [
        _prediction_id(k, h, a)
        for k, h, a in zip(fresh["kickoff"], fresh["home_team"], fresh["away_team"])
    ]
    fresh["recorded_at"] = now
    fresh["league"] = fresh.get("league", league)
    fresh["model_version"] = MODEL_VERSION
    for col in SETTLED_COLUMNS:
        fresh[col] = pd.NA
    fresh = fresh[[c for c in OPEN_COLUMNS + SETTLED_COLUMNS if c in fresh.columns]]

    known = set(ledger["prediction_id"]) if not ledger.empty else set()
    frozen_ids = set()
    if not ledger.empty:
        past = pd.to_datetime(ledger["kickoff"], utc=True) <= now
        frozen_ids = set(ledger.loc[past, "prediction_id"])

    is_frozen = fresh["prediction_id"].isin(frozen_ids) | (fresh["kickoff"] <= now)
    is_update = fresh["prediction_id"].isin(known - frozen_ids) & ~is_frozen
    updatable = fresh[~is_frozen]
    probability_values = updatable[["p_home", "p_draw", "p_away"]]
    try:
        values = probability_values.to_numpy(dtype=float)
        has_boolean = any(isinstance(v, (bool, np.bool_))
                          for v in probability_values.to_numpy().flat)
        valid = (not has_boolean and np.isfinite(values).all()
                 and (values >= 0).all() and (values <= 1).all()
                 and np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Prediction probabilities must be finite, nonnegative and sum to 1")

    kept = ledger[~ledger["prediction_id"].isin(updatable["prediction_id"])] if not ledger.empty else ledger
    merged = pd.concat([kept, updatable], ignore_index=True)
    merged["kickoff"] = pd.to_datetime(merged["kickoff"], utc=True)
    merged = merged.sort_values("kickoff").reset_index(drop=True)
    return merged, {
        "added": int((~is_frozen & ~is_update).sum()),
        "updated": int(is_update.sum()),
        "frozen": int(is_frozen.sum()),
    }


# Explicit identity aliases only: never truncate or guess similar names.
_TEAM_ALIASES = {
    "manchestercity": "mancity", "manchesterunited": "manunited",
    "tottenhamhotspur": "tottenham", "wolverhamptonwanderers": "wolves",
    "newcastleunited": "newcastle", "leedsunited": "leeds",
    "nottinghamforest": "nottmforest", "sheffieldunited": "sheffieldunited",
    "sheffunited": "sheffieldunited", "sheffieldwednesday": "sheffieldweds",
    "westhamunited": "westham", "westbromwichalbion": "westbrom",
    "brightonandhovealbion": "brighton", "brightonhovealbion": "brighton",
    "leicestercity": "leicester", "norwichcity": "norwich",
    "cardiffcity": "cardiff", "swanseacity": "swansea", "stokecity": "stoke",
    "hullcity": "hull", "queensparkrangers": "qpr",
    "parissaintgermain": "parissg", "parissaintgermainfc": "parissg",
    "bayernmunich": "bayernmunich", "bayernmünchen": "bayernmunich",
    "borussiamonchengladbach": "mgladbach", "borussiamönchengladbach": "mgladbach",
    "borussiadortmund": "dortmund", "bayerleverkusen": "leverkusen",
    "internazionalemilano": "inter", "intermilan": "inter",
    "acmilan": "milan", "asroma": "roma",
    "atleticomadrid": "athmadrid", "athleticbilbao": "athbilbao",
    "realbetis": "betis", "realsociedad": "sociedad",
}


def _match_key(home: str, away: str) -> str:
    """Full normalized team identity with conservative explicit aliases."""
    def norm(name: str) -> str:
        key = "".join(ch for ch in str(name).lower() if ch.isalnum())
        return _TEAM_ALIASES.get(key, key)

    return f"{norm(home)}|{norm(away)}"


def settle(
    ledger: pd.DataFrame,
    matches: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Score every open prediction whose result has arrived.

    Unmatched fixtures stay open (the results feed lags by days), so
    ``settle`` is safe to re-run and idempotent on already-settled rows.
    """
    now = now or _now()
    if ledger.empty:
        return ledger, {"settled": 0, "still_open": 0}

    ledger = ledger.copy()
    ledger["kickoff"] = pd.to_datetime(ledger["kickoff"], utc=True)
    open_mask = ledger["result"].isna() & (ledger["kickoff"] <= now)
    if not open_mask.any():
        return ledger, {"settled": 0, "still_open": 0}

    results = matches.copy()
    results["key"] = [
        _match_key(h, a) for h, a in zip(results["home_team"], results["away_team"])
    ]
    results["date"] = pd.to_datetime(results["date"]).dt.date
    # Missing competition identity cannot be settled safely.
    if "league" not in results:
        return ledger, {"settled": 0, "still_open": int(open_mask.sum())}
    lookup = {}
    for row in results.itertuples():
        if row.result in config.OUTCOMES:
            lookup.setdefault((row.league, row.date, row.key), []).append(row.result)

    settled_count = 0
    for idx in ledger.index[open_mask]:
        row = ledger.loc[idx]
        key = _match_key(row["home_team"], row["away_team"])
        kickoff_date = row["kickoff"].date()
        # Kickoff timezone vs the source's match date can differ by a day.
        candidates = [
            outcome
            for d in (kickoff_date,
                      kickoff_date - pd.Timedelta(days=1).to_pytimedelta(),
                      kickoff_date + pd.Timedelta(days=1).to_pytimedelta())
            for outcome in lookup.get((row["league"], d, key), [])
        ]
        # Multiple candidates are ambiguous even if their outcomes agree.
        if len(candidates) != 1:
            continue
        outcome = candidates[0]
        probs = np.array(
            [[row["p_home"], row["p_draw"], row["p_away"]]], dtype=float
        )
        truth = pd.Series([outcome])
        ledger.loc[idx, "result"] = outcome
        ledger.loc[idx, "log_loss"] = float(log_loss_per_match(probs, truth)[0])
        ledger.loc[idx, "brier"] = float(brier_per_match(probs, truth)[0])
        ledger.loc[idx, "settled_at"] = now
        settled_count += 1

    still_open = int((ledger["result"].isna() & (ledger["kickoff"] <= now)).sum())
    return ledger, {"settled": settled_count, "still_open": still_open}


def scoreboard(ledger: pd.DataFrame) -> dict:
    """Live performance so far, with an honest small-sample caveat."""
    if ledger.empty:
        return {"n_settled": 0, "n_open": 0}

    settled = ledger.dropna(subset=["result"])
    open_rows = ledger[ledger["result"].isna()]
    out = {
        "n_settled": int(len(settled)),
        "n_open": int(len(open_rows)),
        "model_versions": sorted(ledger["model_version"].dropna().unique().tolist()),
        "backtest_baseline": BACKTEST_BASELINE,
    }
    if settled.empty:
        return out

    losses = settled["log_loss"].to_numpy(float)
    mean, lo, hi = bootstrap_ci(losses)
    probs = settled[["p_home", "p_draw", "p_away"]].to_numpy(float)
    picked = probs.argmax(axis=1)
    picks = np.array(config.OUTCOMES)[picked]
    out.update(
        {
            "log_loss": mean,
            "log_loss_lo": lo,
            "log_loss_hi": hi,
            "brier": float(settled["brier"].mean()),
            "favorite_hit_rate": float((picks == settled["result"].to_numpy()).mean()),
            "mean_favorite_prob": float(probs.max(axis=1).mean()),
            "first_kickoff": str(settled["kickoff"].min()),
            "last_kickoff": str(settled["kickoff"].max()),
            # Below ~200 settled matches the CI is too wide to conclude much;
            # the page says so rather than implying precision we lack.
            "sample_is_meaningful": bool(len(settled) >= 200),
        }
    )
    return out


def record(api_key: str | None = None, path=LEDGER_PARQUET, league: str = config.DEFAULT_LEAGUE) -> dict:
    """Fetch upcoming fixtures and append their predictions to the ledger."""
    from src.live import SPORT_KEYS, fetch_events, parse_events

    if league not in SPORT_KEYS:
        raise ValueError("Unsupported live league")
    api_key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "No ODDS_API_KEY. Get a free key at the-odds-api.com and run "
            "ODDS_API_KEY=<key> python -m src.ledger record"
        )
    events, quota = fetch_events(api_key, SPORT_KEYS[league])
    predictions = parse_events(events)
    ledger, stats = append_predictions(load_ledger(path), predictions, league=league)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(path, index=False)
    return {**stats, "quota_remaining": quota.get("remaining"), "total_rows": len(ledger)}


def settle_from_disk(path=LEDGER_PARQUET) -> dict:
    matches = pd.read_parquet(config.MATCHES_PARQUET)
    ledger, stats = settle(load_ledger(path), matches)
    ledger.to_parquet(path, index=False)
    return {**stats, "total_rows": len(ledger)}


def main() -> None:
    verb = sys.argv[1] if len(sys.argv) > 1 else "scoreboard"
    if verb == "record":
        print(record())
    elif verb == "settle":
        print(settle_from_disk())
    elif verb == "scoreboard":
        board = scoreboard(load_ledger())
        for key, value in board.items():
            print(f"{key}: {value}")
    else:
        print("usage: python -m src.ledger [record|settle|scoreboard]")
        sys.exit(1)


if __name__ == "__main__":
    main()
