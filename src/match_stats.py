"""Predicting the other things a football match produces.

The platform has proved it cannot out-forecast the market on match
outcomes. That verdict says nothing about corners, cards, shots or fouls —
counts the source has recorded since 2000, that no liquid closing line
disciplines, and that nobody in the open-source ecosystem models honestly.

The model is deliberately simple: counts are close to Poisson, so a
ratio-based strength model (how many does this team generate, how many
does that team concede) plus a referee multiplier captures most of the
signal without an optimizer. Every rate is **shrunk toward the league
average** by sample size — with 578 referees, most of whom officiate a
handful of matches, an unshrunk rate is mostly noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# target -> (home column, away column). Cards are derived (yellow + red).
STAT_TARGETS = {
    "corners": ("home_corners", "away_corners"),
    "cards": ("home_cards", "away_cards"),
    "shots": ("home_shots", "away_shots"),
    "shots_target": ("home_shots_target", "away_shots_target"),
    "fouls": ("home_fouls", "away_fouls"),
}

TEAM_PRIOR_MATCHES = 5    # shrinkage strength for team rates
REFEREE_PRIOR_MATCHES = 20  # referees need far more evidence to move


def with_derived_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns the source implies but doesn't ship (cards = Y + R)."""
    df = df.copy()
    if "home_yellows" in df.columns and "home_cards" not in df.columns:
        df["home_cards"] = df["home_yellows"].fillna(0) + df["home_reds"].fillna(0)
        df["away_cards"] = df["away_yellows"].fillna(0) + df["away_reds"].fillna(0)
    return df


def _columns(target: str) -> tuple[str, str]:
    if target not in STAT_TARGETS:
        raise ValueError(f"unknown target {target!r}; try {list(STAT_TARGETS)}")
    return STAT_TARGETS[target]


def _shrink(rate: pd.Series, counts: pd.Series, prior: int) -> pd.Series:
    """Pull a rate toward 1.0 in proportion to how little evidence backs it."""
    weight = counts / (counts + prior)
    return 1.0 + (rate - 1.0) * weight


def referee_multipliers(
    df: pd.DataFrame,
    target: str = "cards",
    prior: int = REFEREE_PRIOR_MATCHES,
    by_league: bool = True,
    by_season: bool = True,
) -> dict[str, float]:
    """Per-referee multiplier on the match total, shrunk toward neutral.

    Baselined **within league and season** by default. Italian officials
    card far more than English ones, and cards per match have risen for
    two decades — pool either dimension and the multiplier measures the
    competition or the era rather than the person holding the cards.
    """
    home_col, away_col = _columns(target)
    data = with_derived_stats(df).dropna(subset=["referee", home_col, away_col])
    if data.empty:
        return {}
    totals = data[home_col] + data[away_col]

    keys = []
    if by_league and "league" in data.columns:
        keys.append(data["league"])
    if by_season and "season" in data.columns:
        keys.append(data["season"])
    if keys:
        baseline = totals.groupby(keys).transform("mean")
    else:
        baseline = pd.Series(totals.mean(), index=totals.index)
    usable = baseline > 0
    if not usable.any():
        return {}

    relative = (totals[usable] / baseline[usable]).groupby(data.loc[usable, "referee"])
    return _shrink(relative.mean(), relative.size(), prior).to_dict()


def evaluate_lines(
    forecast: pd.DataFrame, lines: tuple[float, ...] = (8.5, 9.5, 10.5)
) -> pd.DataFrame:
    """Score count forecasts the way the platform scores everything else.

    A count prediction only becomes falsifiable once it is a probability,
    so each expected total is turned into P(total > line) under Poisson and
    graded with binary log loss plus a predicted-vs-realized comparison.
    """
    rows = []
    lam = forecast["pred_total"].to_numpy(float)
    actual = forecast["actual_total"].to_numpy(float)
    for line in lines:
        p_over = stats.poisson.sf(np.floor(line), lam)
        went_over = actual > line
        p = np.clip(p_over, 1e-12, 1 - 1e-12)
        losses = -(went_over * np.log(p) + (~went_over) * np.log(1 - p))
        rows.append(
            {
                "line": line,
                "n": int(len(forecast)),
                "predicted_over_rate": float(p_over.mean()),
                "actual_over_rate": float(went_over.mean()),
                "gap": float(went_over.mean() - p_over.mean()),
                "log_loss": float(losses.mean()),
            }
        )
    return pd.DataFrame(rows)


class CountModel:
    """Ratio-based Poisson strength model for a per-match count.

    ``expected(home, away)`` multiplies the league baseline by how much the
    home side generates and how much the visitor concedes — the classic
    attack/defence decomposition, applied to counts rather than goals.
    """

    def __init__(
        self,
        target: str = "corners",
        use_referee: bool = False,
        half_life_days: float | None = 730,
    ):
        self.target = target
        self.use_referee = use_referee
        self.half_life_days = half_life_days
        self.home_base_ = 0.0
        self.away_base_ = 0.0
        self.home_attack_: dict[str, float] = {}
        self.away_attack_: dict[str, float] = {}
        self.home_concede_: dict[str, float] = {}
        self.away_concede_: dict[str, float] = {}
        self.referee_: dict[str, float] = {}

    def _weights(self, data: pd.DataFrame) -> np.ndarray:
        """Exponential recency decay relative to the most recent match."""
        if self.half_life_days is None or "date" not in data.columns:
            return np.ones(len(data))
        age = (data["date"].max() - data["date"]).dt.days.to_numpy(float)
        return np.exp(-np.log(2) * age / self.half_life_days)

    @staticmethod
    def _wmean(values: pd.Series, weights: pd.Series) -> float:
        w = np.asarray(weights, float)
        total = w.sum()
        if total <= 0:
            return float(values.mean())
        return float(np.average(np.asarray(values, float), weights=w))

    def fit(self, matches: pd.DataFrame) -> "CountModel":
        home_col, away_col = _columns(self.target)
        data = with_derived_stats(matches).dropna(subset=[home_col, away_col])
        if data.empty:
            raise ValueError(f"no rows with {self.target} data to fit")

        # Corner and card rates drift across eras (corners down, cards up),
        # so old matches describe a game that is no longer being played.
        data = data.assign(_w=self._weights(data))
        self.home_base_ = self._wmean(data[home_col], data["_w"])
        self.away_base_ = self._wmean(data[away_col], data["_w"])

        by_home = data.groupby("home_team")
        by_away = data.groupby("away_team")
        self.home_attack_ = _shrink(
            by_home.apply(lambda g: self._wmean(g[home_col], g["_w"]))
            / self.home_base_,
            by_home["_w"].sum(), TEAM_PRIOR_MATCHES,
        ).to_dict()
        self.home_concede_ = _shrink(
            by_home.apply(lambda g: self._wmean(g[away_col], g["_w"]))
            / self.away_base_,
            by_home["_w"].sum(), TEAM_PRIOR_MATCHES,
        ).to_dict()
        self.away_attack_ = _shrink(
            by_away.apply(lambda g: self._wmean(g[away_col], g["_w"]))
            / self.away_base_,
            by_away["_w"].sum(), TEAM_PRIOR_MATCHES,
        ).to_dict()
        self.away_concede_ = _shrink(
            by_away.apply(lambda g: self._wmean(g[home_col], g["_w"]))
            / self.home_base_,
            by_away["_w"].sum(), TEAM_PRIOR_MATCHES,
        ).to_dict()

        if self.use_referee:
            self.referee_ = referee_multipliers(data, self.target)
        return self

    def expected(
        self, home: str, away: str, referee: str | None = None
    ) -> tuple[float, float]:
        """Expected counts (home, away). Unknown teams fall back to neutral."""
        lam_home = (
            self.home_base_
            * self.home_attack_.get(home, 1.0)
            * self.away_concede_.get(away, 1.0)
        )
        lam_away = (
            self.away_base_
            * self.away_attack_.get(away, 1.0)
            * self.home_concede_.get(home, 1.0)
        )
        if referee is not None:
            mult = self.referee_.get(referee, 1.0)
            lam_home, lam_away = lam_home * mult, lam_away * mult
        return float(lam_home), float(lam_away)

    def prob_over(
        self, home: str, away: str, line: float, referee: str | None = None
    ) -> float:
        """P(match total > line), treating the total as Poisson."""
        lam_home, lam_away = self.expected(home, away, referee)
        total = lam_home + lam_away
        return float(stats.poisson.sf(np.floor(line), total))


def rolling_forecast(
    df: pd.DataFrame,
    target: str = "corners",
    min_train: int = 200,
    use_referee: bool = False,
    refit_every: int = 1,
    half_life_days: float | None = 730,
) -> pd.DataFrame:
    """Expanding-window forecasts: every match predicted from earlier ones.

    Refitting per matchday (not per match) keeps this fast while preserving
    the guarantee that a match never informs its own prediction.
    """
    home_col, away_col = _columns(target)
    data = with_derived_stats(df).sort_values("date")
    data = data.dropna(subset=[home_col, away_col])

    rows = []
    positions = np.arange(len(data))
    model: CountModel | None = None
    last_fit = -1
    for day, day_positions in pd.Series(
        positions, index=data["date"].to_numpy()
    ).groupby(level=0):
        start = day_positions.iloc[0]
        if start < min_train:
            continue
        if model is None or (start - last_fit) >= refit_every:
            model = CountModel(target, use_referee, half_life_days).fit(
                data.iloc[:start]
            )
            last_fit = start
        for pos in day_positions:
            row = data.iloc[pos]
            ref = row.get("referee") if use_referee else None
            lam_home, lam_away = model.expected(
                row["home_team"], row["away_team"],
                ref if isinstance(ref, str) else None,
            )
            rows.append(
                {
                    "index": data.index[pos],
                    "date": row["date"],
                    "league": row.get("league"),
                    "season": row.get("season"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "pred_home": lam_home,
                    "pred_away": lam_away,
                    "pred_total": lam_home + lam_away,
                    "actual_home": row[home_col],
                    "actual_away": row[away_col],
                    "actual_total": row[home_col] + row[away_col],
                }
            )
    return pd.DataFrame(rows).set_index("index")
