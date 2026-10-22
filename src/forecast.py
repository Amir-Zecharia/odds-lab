"""One fixture, everything the platform knows about it.

Ten studies live in this repo, each pointed at history. This module points
them all at a single match instead: the outcome, the goals and the likely
scorelines, the corner and card counts (adjusted for whoever is
refereeing), and how the first half is likely to unfold.

Nothing here is new modelling — it is the existing, separately-validated
pieces assembled into the answer the whole project was actually built to
give: *what do we expect from this match?*
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.halftime import halftime_transition_matrix
from src.match_stats import CountModel
from src.models import DixonColes
from src.pricing_engine import price_over, top_scorelines

CORNER_LINE = 10.5
CARD_LINE = 3.5


@dataclass
class MatchForecast:
    """Every prediction the platform can make about one fixture."""

    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    expected_home_goals: float
    expected_away_goals: float
    p_over_25_goals: float
    p_btts: float
    top_scorelines: list[dict]
    expected_corners: float
    p_over_corners_line: float
    corner_line: float
    expected_cards: float
    p_over_cards_line: float
    card_line: float
    half_time: dict[str, float]
    ht_to_ft: dict[str, dict[str, float]] = field(default_factory=dict)
    referee: str | None = None
    market: dict[str, float] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _half_time_distribution(grid: np.ndarray, share: float = 0.42) -> dict[str, float]:
    """Half-time outcome probabilities from the full-time goal model.

    Roughly 42% of a match's goals arrive before the break (measured over
    all 61k matches), so the half is modelled as an independent Poisson
    pair scaled by that share — coarse, but honest about where the number
    comes from.
    """
    goals = np.arange(grid.shape[0])
    lam_home = float((grid.sum(axis=1) * goals).sum()) * share
    lam_away = float((grid.sum(axis=0) * goals).sum()) * share
    ht = np.outer(
        stats.poisson.pmf(goals, lam_home), stats.poisson.pmf(goals, lam_away)
    )
    ht = ht / ht.sum()
    return {
        "H": float(np.tril(ht, -1).sum()),
        "D": float(np.trace(ht)),
        "A": float(np.triu(ht, 1).sum()),
    }


def forecast_match(
    history: pd.DataFrame,
    home_team: str,
    away_team: str,
    referee: str | None = None,
    market: dict[str, float] | None = None,
    corner_line: float = CORNER_LINE,
    card_line: float = CARD_LINE,
) -> MatchForecast:
    """Assemble every model's view of one fixture.

    ``history`` must contain only matches that finished **before** the one
    being forecast — the caller owns that discipline, exactly as the
    rolling evaluations elsewhere do.
    """
    goal_model = DixonColes().fit(history)
    grid = goal_model.score_grid(home_team, away_team)
    goals = np.arange(grid.shape[0])
    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())

    hg, ag = np.meshgrid(goals, goals, indexing="ij")
    p_btts = float(grid[(hg > 0) & (ag > 0)].sum())

    counts = {}
    for target, line in (("corners", corner_line), ("cards", card_line)):
        try:
            model = CountModel(target=target, use_referee=referee is not None).fit(
                history
            )
        except (ValueError, KeyError):
            counts[target] = (float("nan"), float("nan"))
            continue
        lam_home, lam_away = model.expected(home_team, away_team, referee)
        counts[target] = (
            lam_home + lam_away,
            model.prob_over(home_team, away_team, line, referee),
        )

    transitions = {}
    try:
        matrix = halftime_transition_matrix(history)
        transitions = {
            str(idx): {str(c): float(v) for c, v in row.items()}
            for idx, row in matrix.iterrows()
        }
    except (KeyError, ValueError):
        transitions = {}

    return MatchForecast(
        home_team=home_team,
        away_team=away_team,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        expected_home_goals=float((grid.sum(axis=1) * goals).sum()),
        expected_away_goals=float((grid.sum(axis=0) * goals).sum()),
        p_over_25_goals=price_over(grid, 2.5),
        p_btts=p_btts,
        top_scorelines=top_scorelines(grid, k=6),
        expected_corners=counts["corners"][0],
        p_over_corners_line=counts["corners"][1],
        corner_line=corner_line,
        expected_cards=counts["cards"][0],
        p_over_cards_line=counts["cards"][1],
        card_line=card_line,
        half_time=_half_time_distribution(grid),
        ht_to_ft=transitions,
        referee=referee,
        market=market,
    )


def market_view(row: pd.Series) -> dict[str, float] | None:
    """The market's own probabilities for a match, when the feed has them.

    Uses the recipe every study here validated: the sharpest available
    quote (Pinnacle closing, else pre-close, else consensus).
    """
    for prefix in ("PSC", "PS", "cons"):
        cols = [f"p_{prefix}_{o}" for o in config.OUTCOMES]
        if all(c in row.index for c in cols) and row[cols].notna().all():
            return {
                "source": {"PSC": "Pinnacle closing", "PS": "Pinnacle",
                           "cons": "Consensus"}[prefix],
                "p_home": float(row[cols[0]]),
                "p_draw": float(row[cols[1]]),
                "p_away": float(row[cols[2]]),
            }
    return None
