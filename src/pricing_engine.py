"""A bookmaker-style origination engine: one scoreline distribution prices
every market consistently.

This is the architecture real trading desks use: a goal model produces the
full P(home goals, away goals) grid, and every market — 1X2, over/under,
Asian handicap, correct score, BTTS — is a sum over that same grid. Prices
across markets can then never contradict each other, and the model can be
raced against the actual books market-by-market: is a stats-only engine
relatively closer to the market in goals markets than in 1X2?

Asian handicap fair pricing is financially exact: with win/half-win/push/
half-loss/loss splits from quarter lines, the fair implied probability of
the home side is (P_w + 0.5*P_hw) / (P_w + 0.5*P_hw + P_l + 0.5*P_hl) —
the zero-EV price with pushes refunded — directly comparable to the
de-margined two-way book price.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.asian_handicap import settle_home_ah
from src.evaluate import paired_bootstrap_diff
from src.models import MAX_GOALS, DixonColes
from src.totals import binary_log_loss

RESULTS_JSON = config.PROCESSED_DIR / "pricing_engine.json"

TEST_FIRST_SEASON = "2019-20"  # era with all three markets quoted
TRAIN_WINDOW_SEASONS = 6

_GOALS = np.arange(MAX_GOALS + 1)
_HG, _AG = np.meshgrid(_GOALS, _GOALS, indexing="ij")


def price_1x2(grid: np.ndarray) -> np.ndarray:
    """P(H), P(D), P(A) from the scoreline grid."""
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def price_over(grid: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line)."""
    return float(grid[(_HG + _AG) > line].sum())


def price_btts(grid: np.ndarray) -> float:
    """P(both teams score)."""
    return float(grid[(_HG > 0) & (_AG > 0)].sum())


def price_ah_cover(grid: np.ndarray, line: float) -> float:
    """Fair implied probability that the HOME Asian-handicap bet wins.

    Uses the settlement fraction of every scoreline at the quoted line and
    the zero-EV pricing identity for two-way markets with pushes/half-stakes.
    """
    diff = _HG - _AG
    frac = np.vectorize(settle_home_ah)(diff, line)
    p_w = grid[frac == 1.0].sum()
    p_hw = grid[frac == 0.5].sum()
    p_l = grid[frac == -1.0].sum()
    p_hl = grid[frac == -0.5].sum()
    num = p_w + 0.5 * p_hw
    den = num + p_l + 0.5 * p_hl
    return float(num / den) if den > 0 else float("nan")


def top_scorelines(grid: np.ndarray, k: int = 8) -> list[dict]:
    """The k most likely exact scores — the correct-score market's head."""
    flat = [
        {"score": f"{i}-{j}", "prob": float(grid[i, j])}
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
    ]
    return sorted(flat, key=lambda r: -r["prob"])[:k]


def price_match(model: DixonColes, home: str, away: str, ah_line: float | None) -> dict:
    """Every market for one fixture from one consistent grid."""
    grid = model.score_grid(home, away)
    h, d, a = price_1x2(grid)
    out = {
        "p_home": float(h),
        "p_draw": float(d),
        "p_away": float(a),
        "p_over25": price_over(grid),
        "p_btts": price_btts(grid),
        "top_scores": top_scorelines(grid),
    }
    if ah_line is not None and np.isfinite(ah_line):
        out["p_ah_cover"] = price_ah_cover(grid, float(ah_line))
    return out


# ---------------------------------------------------------------------------
# The race: engine vs the real books, per market, expanding by season.
# ---------------------------------------------------------------------------


def _market_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Book probability columns each market is judged against."""
    return {
        "1x2": [f"p_cons_{o}" for o in config.OUTCOMES],
        "totals": ["p_over_Avg"],
        "ah": ["p_ahcover_Avg", "ah_line", "ah_result"],
    }


def engine_vs_market(df: pd.DataFrame) -> dict:
    """Refit Dixon-Coles before each test season per league; price all
    markets per match; per-market paired log-loss gap engine − market."""
    from src.evaluate import log_loss_per_match

    needed = _market_columns(df)
    rows_1x2, rows_tot, rows_ah = [], [], []

    for league, group in df.groupby("league"):
        data = group.sort_values("date")
        seasons = sorted(data["season"].unique())
        test_seasons = [s for s in seasons if s >= TEST_FIRST_SEASON]
        for season in test_seasons:
            train_pool = data[data["season"] < season]
            window = sorted(train_pool["season"].unique())[-TRAIN_WINDOW_SEASONS:]
            train = train_pool[train_pool["season"].isin(window)]
            test = data[data["season"] == season]
            if len(train) < 500 or test.empty:
                continue
            model = DixonColes().fit(train)
            for r in test.itertuples():
                ah_line = getattr(r, "ah_line", float("nan"))
                priced = price_match(model, r.home_team, r.away_team, ah_line)
                base = {"league": league, "season": season}
                rows_1x2.append(
                    {
                        **base,
                        "model": [priced["p_home"], priced["p_draw"], priced["p_away"]],
                        "market": [getattr(r, c) for c in needed["1x2"]],
                        "result": r.result,
                    }
                )
                over = getattr(r, "p_over_Avg", float("nan"))
                if np.isfinite(over) and np.isfinite(getattr(r, "went_over", np.nan)):
                    rows_tot.append(
                        {
                            **base,
                            "model": priced["p_over25"],
                            "market": float(over),
                            "went_over": bool(r.went_over),
                        }
                    )
                cover_mkt = getattr(r, "p_ahcover_Avg", float("nan"))
                ah_res = getattr(r, "ah_result", float("nan"))
                if (
                    "p_ah_cover" in priced
                    and np.isfinite(cover_mkt)
                    and np.isfinite(ah_res)
                    and ah_res != 0  # drop exact pushes, mirroring the AH study
                ):
                    rows_ah.append(
                        {
                            **base,
                            "model": priced["p_ah_cover"],
                            "market": float(cover_mkt),
                            "covered": bool(ah_res > 0),
                        }
                    )

    out: dict = {"markets": {}}

    frame = pd.DataFrame(rows_1x2).dropna(subset=["market"])
    valid = frame["market"].apply(lambda v: np.isfinite(v).all())
    frame = frame[valid]
    model_p = np.vstack(frame["model"])
    market_p = np.vstack(frame["market"])
    results = frame["result"].reset_index(drop=True)
    l_model = log_loss_per_match(model_p, results)
    l_market = log_loss_per_match(market_p, results)
    boot = paired_bootstrap_diff(l_model, l_market)
    out["markets"]["1x2"] = {
        "n": int(len(frame)),
        "model_log_loss": float(l_model.mean()),
        "market_log_loss": float(l_market.mean()),
        "gap": boot["diff"],
        "lo": boot["lo"],
        "hi": boot["hi"],
    }

    for name, rows, truth_col in (
        ("totals", rows_tot, "went_over"),
        ("ah", rows_ah, "covered"),
    ):
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        y = frame[truth_col].to_numpy(bool)
        l_model = binary_log_loss(frame["model"].to_numpy(float), y)
        l_market = binary_log_loss(frame["market"].to_numpy(float), y)
        boot = paired_bootstrap_diff(l_model, l_market)
        out["markets"][name] = {
            "n": int(len(frame)),
            "model_log_loss": float(l_model.mean()),
            "market_log_loss": float(l_market.mean()),
            "gap": boot["diff"],
            "lo": boot["lo"],
            "hi": boot["hi"],
        }

    # Relative gap: which market does a stats-only engine trail least in?
    for m in out["markets"].values():
        m["gap_pct_of_market"] = m["gap"] / m["market_log_loss"]
    return out


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    results = {
        "test_first_season": TEST_FIRST_SEASON,
        "train_window_seasons": TRAIN_WINDOW_SEASONS,
        **engine_vs_market(df),
        "computed_at": pd.Timestamp.now().isoformat(),
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    for name, m in results["markets"].items():
        print(
            f"{name:7s} n={m['n']:6,}  engine {m['model_log_loss']:.4f} vs "
            f"market {m['market_log_loss']:.4f}  gap {m['gap']:+.4f} "
            f"[{m['lo']:+.4f}, {m['hi']:+.4f}]  ({m['gap_pct_of_market']:+.1%})"
        )
    print(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
