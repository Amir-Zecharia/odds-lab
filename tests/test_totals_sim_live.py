import numpy as np
import pandas as pd
import pytest

from src.ingest import normalize_season
from src.live import parse_events
from src.simulator import bet_favorite, bet_value, summarize
from src.totals import add_totals_features

# ---- totals ----------------------------------------------------------------

TOTALS_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,BbAv>2.5,BbAv<2.5,BbMx>2.5,BbMx<2.5
E0,14/08/10,Arsenal,Wigan,3,1,H,1.4,4.5,8.0,1.85,1.95,1.95,2.05
E0,14/08/10,Fulham,Bolton,0,0,D,2.1,3.3,3.6,2.10,1.72,2.20,1.80
"""


def test_ingest_keeps_and_renames_totals_columns():
    out = normalize_season(TOTALS_CSV, "2010-11")
    assert "Avg>2.5" in out.columns and "Max<2.5" in out.columns
    assert out.loc[0, "Avg>2.5"] == 1.85


def test_totals_features_probabilities_and_truth():
    df = add_totals_features(normalize_season(TOTALS_CSV, "2010-11"))
    assert df.loc[0, "went_over"]  # 3-1 = 4 goals
    assert not df.loc[1, "went_over"]  # 0-0
    p = df["p_over_Avg"]
    assert ((p > 0) & (p < 1)).all()
    # Over priced shorter than under in match 1 -> p_over > 0.5.
    assert p.iloc[0] > 0.5 > p.iloc[1]


# ---- simulator ---------------------------------------------------------------


def _sim_frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "season": "2019-20",
            "home_team": list("abc"),
            "away_team": list("xyz"),
            "result": ["H", "A", "D"],
            "B365H": [1.5, 1.5, 2.0],
            "B365D": [4.0, 4.0, 3.4],
            "B365A": [6.0, 6.0, 3.8],
            "p_cons_H": [0.70, 0.20, 0.40],
            "p_cons_D": [0.20, 0.20, 0.35],
            "p_cons_A": [0.10, 0.60, 0.25],
        }
    )


def test_favorite_strategy_settles_correctly():
    bets = bet_favorite(_sim_frame(), "B365")
    # Favorite (H at 1.5) wins match 1: +0.5; loses matches 2 and 3: -1 each.
    assert bets["profit"].tolist() == [0.5, -1.0, -1.0]
    s = summarize(bets)
    assert s["bets"] == 3 and s["profit"] == pytest.approx(-1.5)


def test_value_strategy_only_bets_positive_edge():
    bets = bet_value(_sim_frame(), "B365", "cons", edge_threshold=0.04)
    # Match 1: EV H = .7*1.5-1 ~ .05 -> bet. Match 2: EV A = .6*6-1 = 2.6 -> bet.
    # Match 3: best EV = D: .35*3.4-1 = .19 -> bet.
    assert len(bets) == 3
    assert bets.iloc[1]["pick"] == "A" and bets.iloc[1]["won"]


def test_kelly_stakes_scale_with_edge():
    bets = bet_value(_sim_frame(), "B365", "cons", edge_threshold=0.0, kelly=True)
    assert (bets["stake"] >= 0).all()
    # The huge-edge bet (match 2) must be capped at 10 units (10% of 100).
    assert bets["stake"].max() == pytest.approx(10.0)


# ---- live odds parsing ---------------------------------------------------------

EVENTS = [
    {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2026-08-15T14:00:00Z",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Draw", "price": 3.5},
                            {"name": "Chelsea", "price": 3.9},
                        ],
                    }
                ],
            },
            {
                "key": "unibet",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 1.95},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Chelsea", "price": 3.8},
                        ],
                    }
                ],
            },
        ],
    },
    {
        "home_team": "Leeds",
        "away_team": "Everton",
        "commence_time": "2026-08-15T16:30:00Z",
        "bookmakers": [
            {
                "key": "unibet",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Leeds", "price": 2.6},
                            {"name": "Draw", "price": 3.2},
                            {"name": "Everton", "price": 2.9},
                        ],
                    }
                ],
            }
        ],
    },
]


def test_parse_events_prefers_pinnacle():
    out = parse_events(EVENTS)
    assert len(out) == 2
    arsenal = out.iloc[0]
    assert arsenal["source"] == "Pinnacle"
    assert arsenal["books"] == 2
    np.testing.assert_allclose(
        arsenal[["p_home", "p_draw", "p_away"]].astype(float).sum(), 1.0, atol=1e-9
    )
    assert out.iloc[1]["source"] == "average of 1 books"


def test_parse_events_empty_input():
    out = parse_events([])
    assert out.empty and "p_home" in out.columns
