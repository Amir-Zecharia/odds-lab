"""From decimal odds to probabilities: de-margining and cross-book features.

Three de-margining methods, ordered by sophistication:

- ``proportional``: divide implied probabilities by their sum. Simple, but
  ignores that bookmakers load more margin onto longshots.
- ``power``: raise implied probabilities to the exponent k solving
  sum(p^k) = 1. Shrinks longshots more than favorites.
- ``shin``: Shin (1992) models the margin as protection against insider
  trading; solving for the insider fraction z yields probabilities that
  correct the favorite-longshot bias. Preferred method here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from src import config


def implied(odds: np.ndarray) -> np.ndarray:
    """Raw implied probabilities 1/odds (rows sum to > 1 by the margin)."""
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds: np.ndarray) -> np.ndarray:
    """Bookmaker margin: sum of implied probabilities minus 1."""
    return implied(odds).sum(axis=-1) - 1.0


def demargin_proportional(odds: np.ndarray) -> np.ndarray:
    p = implied(odds)
    return p / p.sum(axis=-1, keepdims=True)


def _power_row(p: np.ndarray) -> np.ndarray:
    f = lambda k: np.sum(p**k) - 1.0
    k = optimize.brentq(f, 1.0, 20.0)
    return p**k


def demargin_power(odds: np.ndarray) -> np.ndarray:
    p = implied(np.atleast_2d(odds))
    out = np.vstack([_power_row(row) for row in p])
    return out if np.ndim(odds) > 1 else out[0]


def _shin_row(p: np.ndarray) -> np.ndarray:
    """Solve for Shin's insider-trading fraction z, return true probs."""
    total = p.sum()
    # Shin assumes a positive margin; best-price aggregates can have
    # overround <= 0, where the model degenerates to proportional.
    if total <= 1.0 + 1e-9:
        return p / total

    def true_probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * p**2 / total) - z) / (2 * (1 - z))

    f = lambda z: true_probs(z).sum() - 1.0
    try:
        z = optimize.brentq(f, 0.0, 0.5)
    except ValueError:
        return p / total
    return true_probs(z)


def demargin_shin(odds: np.ndarray) -> np.ndarray:
    p = implied(np.atleast_2d(odds))
    out = np.vstack([_shin_row(row) for row in p])
    return out if np.ndim(odds) > 1 else out[0]


DEMARGIN = {
    "proportional": demargin_proportional,
    "power": demargin_power,
    "shin": demargin_shin,
}


def _odds_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}{o}" for o in config.OUTCOMES]].to_numpy(dtype=float)


def book_probabilities(
    df: pd.DataFrame, prefix: str, method: str = "shin"
) -> pd.DataFrame:
    """De-margined H/D/A probabilities for one book. NaN rows preserved."""
    odds = _odds_matrix(df, prefix)
    result = np.full_like(odds, np.nan)
    valid = ~np.isnan(odds).any(axis=1)
    if valid.any():
        result[valid] = DEMARGIN[method](odds[valid])
    return pd.DataFrame(
        result,
        index=df.index,
        columns=[f"p_{prefix}_{o}" for o in config.OUTCOMES],
    )


def _present(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    return [
        p for p in prefixes
        if all(f"{p}{o}" in df.columns for o in config.OUTCOMES)
    ]


def available_books(df: pd.DataFrame) -> list[str]:
    """Individual (pre-closing) bookmaker prefixes present in the frame."""
    return _present(df, config.BOOK_PREFIXES)


def available_closing_books(df: pd.DataFrame) -> list[str]:
    """Closing-odds prefixes (e.g. PSC, B365C, AvgC) present in the frame."""
    return _present(df, config.CLOSING_PREFIXES)


def add_market_features(df: pd.DataFrame, method: str = "shin") -> pd.DataFrame:
    """Attach per-match market features to the match frame.

    - ``p_cons_*``: margin-weighted consensus across individual books
      (weight = 1/overround, so sharper prices count more).
    - ``disp_std`` / ``disp_range``: cross-book dispersion of home-win
      probability (std and max-min).
    - ``sharp_soft_gap``: Pinnacle minus Bet365 home-win probability.
    - ``p_Avg_*`` / ``p_Max_*``: de-margined market average / best price;
      ``p_PSC_*`` etc. for closing lines where the source provides them.
    - ``margin_<book>``: overround per book.
    """
    books = available_books(df)
    new: dict[str, pd.Series] = {}

    home_probs, weights = {}, {}
    for prefix in books:
        probs = book_probabilities(df, prefix, method)
        margin = pd.Series(overround(_odds_matrix(df, prefix)), index=df.index)
        new[f"margin_{prefix}"] = margin
        for o in config.OUTCOMES:
            new[f"p_{prefix}_{o}"] = probs[f"p_{prefix}_{o}"]
        home_probs[prefix] = probs[f"p_{prefix}_H"]
        weights[prefix] = 1.0 / (1.0 + margin)

    # Margin-weighted consensus over all individual books.
    for o in config.OUTCOMES:
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for prefix in books:
            p = new[f"p_{prefix}_{o}"]
            w = weights[prefix].where(p.notna())
            num = num.add(p * w, fill_value=0.0)
            den = den.add(w, fill_value=0.0)
        new[f"p_cons_{o}"] = num / den

    # Renormalize consensus (weights vary slightly per outcome when a book
    # is missing one column).
    cons = pd.DataFrame({o: new[f"p_cons_{o}"] for o in config.OUTCOMES})
    cons = cons.div(cons.sum(axis=1), axis=0)
    for o in config.OUTCOMES:
        new[f"p_cons_{o}"] = cons[o]

    hp = pd.DataFrame(home_probs)
    new["disp_std"] = hp.std(axis=1)
    new["disp_range"] = hp.max(axis=1) - hp.min(axis=1)

    if config.SHARP_BOOK in books and config.SOFT_BOOK in books:
        new["sharp_soft_gap"] = (
            new[f"p_{config.SHARP_BOOK}_H"] - new[f"p_{config.SOFT_BOOK}_H"]
        )

    # Market aggregates and closing lines (closing available from 2019-20,
    # Pinnacle closing from 2012-13): the most informative prices.
    extra = [a for a in ("Avg", "Max") if f"{a}H" in df.columns]
    extra += available_closing_books(df)
    for prefix in extra:
        probs = book_probabilities(df, prefix, method)
        for o in config.OUTCOMES:
            new[f"p_{prefix}_{o}"] = probs[f"p_{prefix}_{o}"]
        new[f"margin_{prefix}"] = pd.Series(
            overround(_odds_matrix(df, prefix)), index=df.index
        )

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
