"""Half time as a second outcome, and the states in between.

The source has recorded the half-time score of every match since 2000 —
100% coverage, and until now completely unused. That is a whole extra
outcome per match, plus the transitions between them: how often a lead
survives, how often it is overturned, and how the second half differs
from the first.

None of this needs a market to be worth predicting; it is simply
structure in football that the platform had not looked at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

HT_FT_STATES = [f"{a}/{b}" for a in config.OUTCOMES for b in config.OUTCOMES]


def add_halftime_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive every half-time / full-time target from the raw columns."""
    out = df.copy()
    ht_home = out["ht_home_goals"]
    ht_away = out["ht_away_goals"]

    out["ht_result"] = np.select(
        [ht_home > ht_away, ht_home < ht_away], ["H", "A"], default="D"
    )
    out.loc[ht_home.isna() | ht_away.isna(), "ht_result"] = None
    out["ht_ft"] = out["ht_result"].astype("string") + "/" + out["result"].astype("string")
    out["ht_goals"] = ht_home + ht_away
    out["second_half_goals"] = (
        (out["home_goals"] + out["away_goals"]) - out["ht_goals"]
    )
    out["btts"] = (out["home_goals"] > 0) & (out["away_goals"] > 0)
    out["ht_btts"] = (ht_home > 0) & (ht_away > 0)
    return out


def halftime_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """P(full-time result | half-time result) — how leads hold up."""
    data = add_halftime_targets(df).dropna(subset=["ht_result"])
    matrix = (
        pd.crosstab(data["ht_result"], data["result"], normalize="index")
        .reindex(index=config.OUTCOMES, columns=config.OUTCOMES)
        .dropna(how="all")
    )
    return matrix.fillna(0.0)


def comeback_rates(df: pd.DataFrame) -> dict:
    """How often a half-time lead is thrown away, by side."""
    data = add_halftime_targets(df).dropna(subset=["ht_result"])
    home_led = data["ht_result"] == "H"
    away_led = data["ht_result"] == "A"
    home_lost = home_led & (data["result"] == "A")
    away_lost = away_led & (data["result"] == "H")
    led = int(home_led.sum() + away_led.sum())
    overturned = int(home_lost.sum() + away_lost.sum())
    return {
        "home_led_at_ht": int(home_led.sum()),
        "home_led_and_lost": int(home_lost.sum()),
        "home_led_and_won": int((home_led & (data["result"] == "H")).sum()),
        "away_led_at_ht": int(away_led.sum()),
        "away_led_and_lost": int(away_lost.sum()),
        "away_led_and_won": int((away_led & (data["result"] == "A")).sum()),
        "matches_with_a_lead": led,
        "leads_overturned": overturned,
        "comeback_rate": float(overturned / led) if led else float("nan"),
    }


def half_comparison(df: pd.DataFrame) -> dict:
    """Are the two halves the same game? Goals and home edge, compared."""
    data = add_halftime_targets(df).dropna(subset=["ht_result"])
    return {
        "first_half_goals": float(data["ht_goals"].mean()),
        "second_half_goals": float(data["second_half_goals"].mean()),
        "ht_home_win_rate": float((data["ht_result"] == "H").mean()),
        "ft_home_win_rate": float((data["result"] == "H").mean()),
        "ht_draw_rate": float((data["ht_result"] == "D").mean()),
        "ft_draw_rate": float((data["result"] == "D").mean()),
        "btts_rate": float(data["btts"].mean()),
        "n": int(len(data)),
    }
