import numpy as np
import pandas as pd
import pytest

from src.models import MAX_GOALS, DixonColes
from src.pricing_engine import (
    price_1x2,
    price_ah_cover,
    price_btts,
    price_match,
    price_over,
    top_scorelines,
)


def _grid(cells: dict[tuple[int, int], float]) -> np.ndarray:
    """Sparse scoreline grid from {(home, away): prob}."""
    g = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for (i, j), p in cells.items():
        g[i, j] = p
    return g


def test_price_1x2_partitions_the_grid():
    g = _grid({(2, 0): 0.5, (1, 1): 0.3, (0, 3): 0.2})
    h, d, a = price_1x2(g)
    assert (h, d, a) == pytest.approx((0.5, 0.3, 0.2))
    assert h + d + a == pytest.approx(1.0)


def test_price_over_and_btts_on_known_scores():
    g = _grid({(1, 0): 0.4, (2, 2): 0.6})
    assert price_over(g) == pytest.approx(0.6)   # only 2-2 clears 2.5
    assert price_btts(g) == pytest.approx(0.6)   # only 2-2 has both scoring


def test_ah_full_line_push_is_refunded():
    # Line -1.0: a 1-0 win is a push (excluded); 2-0 covers.
    g = _grid({(2, 0): 0.6, (1, 0): 0.4})
    assert price_ah_cover(g, -1.0) == pytest.approx(1.0)


def test_ah_quarter_line_half_stakes():
    # Line -0.75 with a certain 1-0: half-win -> fair prob 1.
    assert price_ah_cover(_grid({(1, 0): 1.0}), -0.75) == pytest.approx(1.0)
    # Line -0.25 with a certain 0-0 draw: half-loss -> fair prob 0.
    assert price_ah_cover(_grid({(0, 0): 1.0}), -0.25) == pytest.approx(0.0)
    # Mixed: line -0.25, half the mass draws (half-loss), half wins by one.
    g = _grid({(0, 0): 0.5, (1, 0): 0.5})
    # num = 0.5(win), den = 0.5 + 0.5*0.5(half-loss) = 0.75
    assert price_ah_cover(g, -0.25) == pytest.approx(0.5 / 0.75)


def test_ah_all_push_returns_nan():
    assert np.isnan(price_ah_cover(_grid({(1, 0): 1.0}), -1.0))


def test_top_scorelines_sorted():
    g = _grid({(1, 0): 0.5, (1, 1): 0.3, (0, 0): 0.2})
    top = top_scorelines(g, k=2)
    assert top[0]["score"] == "1-0" and top[1]["score"] == "1-1"


def _mini_matches(n_rounds=30, seed=3):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    rows = []
    date = pd.Timestamp("2019-08-01")
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg, ag = rng.poisson(1.4), rng.poisson(1.1)
                rows.append(
                    {"date": date, "season": "2019-20", "home_team": h,
                     "away_team": a, "home_goals": hg, "away_goals": ag,
                     "result": "H" if hg > ag else ("A" if ag > hg else "D")}
                )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_price_match_is_internally_consistent():
    model = DixonColes().fit(_mini_matches())
    priced = price_match(model, "A", "B", ah_line=-0.5)
    probs = [priced["p_home"], priced["p_draw"], priced["p_away"]]
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert 0 < priced["p_over25"] < 1
    assert 0 < priced["p_btts"] < 1
    # At line -0.5 the AH home side wins iff home wins outright.
    assert priced["p_ah_cover"] == pytest.approx(priced["p_home"], abs=1e-9)
    assert len(priced["top_scores"]) == 8
