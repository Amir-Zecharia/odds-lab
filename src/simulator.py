"""Betting strategy backtests — the educational 'why you lose' engine.

Every strategy returns one row per bet: the pick, the decimal odds taken,
the stake, and the settled profit. The bankroll path is just the cumsum.
No strategy here is advice; the point is watching the margin grind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

KELLY_CAP = 0.10  # never stake more than 10% of bankroll on one bet


def _odds_matrix(df: pd.DataFrame, book: str) -> np.ndarray:
    return df[[f"{book}{o}" for o in config.OUTCOMES]].to_numpy(float)


def _prob_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"p_{prefix}_{o}" for o in config.OUTCOMES]].to_numpy(float)


def settle(picks: pd.DataFrame) -> pd.DataFrame:
    """Add won/profit columns. Expects pick, odds, stake, result columns."""
    picks = picks.copy()
    picks["won"] = picks["pick"] == picks["result"]
    picks["profit"] = np.where(
        picks["won"], picks["stake"] * (picks["odds"] - 1), -picks["stake"]
    )
    return picks


def _base_frame(df: pd.DataFrame, book: str) -> pd.DataFrame:
    odds = _odds_matrix(df, book)
    ok = ~np.isnan(odds).any(axis=1)
    sub = df.loc[ok, ["date", "season", "home_team", "away_team", "result"]].copy()
    return sub, odds[ok]


def bet_favorite(df: pd.DataFrame, book: str) -> pd.DataFrame:
    """Flat 1-unit stake on the shortest price every match."""
    sub, odds = _base_frame(df, book)
    idx = odds.argmin(axis=1)
    sub["pick"] = [config.OUTCOMES[i] for i in idx]
    sub["odds"] = odds[np.arange(len(odds)), idx]
    sub["stake"] = 1.0
    return settle(sub)


def bet_longshot(df: pd.DataFrame, book: str) -> pd.DataFrame:
    """Flat 1-unit stake on the longest price every match."""
    sub, odds = _base_frame(df, book)
    idx = odds.argmax(axis=1)
    sub["pick"] = [config.OUTCOMES[i] for i in idx]
    sub["odds"] = odds[np.arange(len(odds)), idx]
    sub["stake"] = 1.0
    return settle(sub)


def bet_home(df: pd.DataFrame, book: str) -> pd.DataFrame:
    """Flat 1-unit stake on the home team every match."""
    sub, odds = _base_frame(df, book)
    sub["pick"] = "H"
    sub["odds"] = odds[:, 0]
    sub["stake"] = 1.0
    return settle(sub)


def bet_value(
    df: pd.DataFrame,
    book: str,
    prob_prefix: str = "cons",
    edge_threshold: float = 0.02,
    kelly: bool = False,
) -> pd.DataFrame:
    """Bet whenever model probability x offered odds exceeds 1 + threshold.

    The "model" is any probability source in the frame (consensus,
    Pinnacle closing, ...). Stake is flat 1 unit, or a capped Kelly
    fraction of a nominal 100-unit bankroll when kelly=True.
    """
    sub, odds = _base_frame(df, book)
    probs = _prob_matrix(df.loc[sub.index], prob_prefix)
    ok = ~np.isnan(probs).any(axis=1)
    sub, odds, probs = sub.loc[ok], odds[ok], probs[ok]

    ev = probs * odds - 1.0
    idx = ev.argmax(axis=1)
    rows = np.arange(len(sub))
    edge = ev[rows, idx]
    take = edge >= edge_threshold

    sub = sub.iloc[take].copy()
    pick_idx = idx[take]
    sub["pick"] = [config.OUTCOMES[i] for i in pick_idx]
    sub["odds"] = odds[take, pick_idx]
    sub["edge"] = edge[take]
    if kelly:
        b = sub["odds"] - 1.0
        p = probs[take, pick_idx]
        frac = np.clip((b * p - (1 - p)) / b, 0.0, KELLY_CAP)
        sub["stake"] = 100.0 * frac  # nominal 100-unit bankroll
    else:
        sub["stake"] = 1.0
    return settle(sub)


STRATEGIES = {
    "always the favorite": bet_favorite,
    "always the longshot": bet_longshot,
    "always the home team": bet_home,
    "value vs model": bet_value,
}


def summarize(bets: pd.DataFrame) -> dict:
    staked = bets["stake"].sum()
    return {
        "bets": int(len(bets)),
        "staked": float(staked),
        "profit": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / staked) if staked else 0.0,
        "hit_rate": float(bets["won"].mean()) if len(bets) else 0.0,
    }
