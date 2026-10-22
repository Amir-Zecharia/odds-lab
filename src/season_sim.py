"""Season simulation: turning match probabilities into the questions fans
actually ask.

"Who wins the league?" and "are we going down?" are prediction problems no
single match forecast answers. Monte Carlo bridges the gap: play out every
remaining fixture thousands of times from the per-match probabilities the
platform already produces, and count how often each story happens.

This is pure forecasting — no market, no margin, nothing to beat. The only
thing that matters is that the inputs are honest and the accounting is
right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

WIN, DRAW, LOSS = 3, 1, 0


def league_table(played: pd.DataFrame) -> pd.DataFrame:
    """Standings from completed matches: points, goal difference, rank."""
    teams = sorted(set(played["home_team"]) | set(played["away_team"]))
    stats = {
        t: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
            "goals_for": 0, "goals_against": 0, "points": 0}
        for t in teams
    }
    for row in played.itertuples():
        hg, ag = int(row.home_goals), int(row.away_goals)
        h, a = stats[row.home_team], stats[row.away_team]
        h["played"] += 1
        a["played"] += 1
        h["goals_for"] += hg
        h["goals_against"] += ag
        a["goals_for"] += ag
        a["goals_against"] += hg
        if hg > ag:
            h["won"] += 1
            h["points"] += WIN
            a["lost"] += 1
        elif hg < ag:
            a["won"] += 1
            a["points"] += WIN
            h["lost"] += 1
        else:
            h["drawn"] += 1
            a["drawn"] += 1
            h["points"] += DRAW
            a["points"] += DRAW

    table = pd.DataFrame(stats).T
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def simulate_season(
    played: pd.DataFrame,
    remaining: pd.DataFrame,
    probabilities: np.ndarray,
    n_sims: int = 1000,
    seed: int = 0,
    goals_for_winner: int = 2,
    goals_for_loser: int = 0,
) -> list[pd.DataFrame]:
    """Play the rest of the season ``n_sims`` times.

    Only the outcome distribution matters for points; scorelines are filled
    with a nominal margin so goal difference stays a sensible tiebreaker
    without pretending to model exact scores.
    """
    if len(remaining) != len(probabilities):
        raise ValueError("one probability row required per remaining fixture")
    rng = np.random.default_rng(seed)
    outcomes = np.array(config.OUTCOMES)
    fixtures = list(remaining[["home_team", "away_team"]].itertuples(index=False))

    tables = []
    for _ in range(n_sims):
        draws = [
            outcomes[rng.choice(3, p=row / row.sum())] for row in probabilities
        ]
        simulated = pd.DataFrame(
            [
                {
                    "home_team": home, "away_team": away,
                    "home_goals": goals_for_winner if o == "H" else (
                        goals_for_loser if o == "A" else 1),
                    "away_goals": goals_for_winner if o == "A" else (
                        goals_for_loser if o == "H" else 1),
                }
                for (home, away), o in zip(fixtures, draws)
            ]
        )
        full = pd.concat([played[["home_team", "away_team", "home_goals",
                                  "away_goals"]], simulated], ignore_index=True)
        tables.append(league_table(full))
    return tables


def summarize_simulations(
    tables: list[pd.DataFrame],
    european_places: int = 4,
    relegation_places: int = 3,
) -> pd.DataFrame:
    """Per-team outcome probabilities across the simulated seasons."""
    n = len(tables)
    teams = tables[0].index
    n_teams = len(teams)
    rows = {}
    for team in teams:
        ranks = np.array([t.loc[team, "rank"] for t in tables])
        points = np.array([t.loc[team, "points"] for t in tables])
        rows[team] = {
            "expected_points": float(points.mean()),
            "points_lo": float(np.quantile(points, 0.05)),
            "points_hi": float(np.quantile(points, 0.95)),
            "expected_rank": float(ranks.mean()),
            "title": float((ranks == 1).mean()),
            "european": float((ranks <= european_places).mean()),
            "relegation": float((ranks > n_teams - relegation_places).mean()),
        }
    summary = pd.DataFrame(rows).T
    return summary.sort_values("expected_points", ascending=False)


def position_distribution(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Full P(team finishes in position k) matrix — the fun chart."""
    teams = tables[0].index
    n_teams = len(teams)
    counts = pd.DataFrame(0.0, index=teams, columns=range(1, n_teams + 1))
    for table in tables:
        for team in teams:
            counts.loc[team, int(table.loc[team, "rank"])] += 1
    return counts / len(tables)
