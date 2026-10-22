import numpy as np
import pandas as pd
import pytest

from src.asian_handicap import (
    add_ah_features,
    efficiency_ah,
    p_home_cover,
    settle_home_ah,
)
from src.ingest import normalize_season

# ---- settlement math --------------------------------------------------------


@pytest.mark.parametrize(
    "goal_diff, line, expected",
    [
        (0, 0.0, 0.0),    # level line, draw -> stake back (push)
        (0, -0.5, -1.0),  # home -0.5, draw -> full loss
        (0, -0.25, -0.5), # home -0.25, draw -> half loss (0.0 push + -0.5)
        (1, -0.75, 0.5),  # home -0.75, win by 1 -> half win (-0.5 win + -1.0 push)
        (0, 0.5, 1.0),    # home +0.5, draw -> full win
        (1, -0.5, 1.0),   # home -0.5, win by 1 -> full win
        (-1, 0.0, -1.0),  # level line, home loses -> full loss
        (2, -0.75, 1.0),  # home -0.75, win by 2 -> full win on both halves
    ],
)
def test_settle_home_ah_hand_cases(goal_diff, line, expected):
    assert settle_home_ah(goal_diff, line) == expected


def test_settle_quarter_line_symmetry():
    # +0.25 draw -> half win mirrors -0.25 draw -> half loss.
    assert settle_home_ah(0, 0.25) == 0.5
    assert settle_home_ah(0, -0.25) == -0.5


# ---- de-margining -----------------------------------------------------------


def test_p_home_cover_sums_to_one_two_way():
    df = pd.DataFrame({"B365AHH": [1.90, 2.10], "B365AHA": [1.98, 1.78]})
    p_home = p_home_cover(df, "B365")
    p_away = p_home_cover(df.rename(columns={"B365AHH": "B365AHA", "B365AHA": "B365AHH"}), "B365")
    np.testing.assert_allclose((p_home + p_away).to_numpy(), 1.0, atol=1e-9)
    # Shorter home price -> higher home-cover probability.
    assert p_home.iloc[0] > 0.5 > p_home.iloc[1]


# ---- ingest keeps AH columns ------------------------------------------------

AH_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AHh,B365AHH,B365AHA,AHCh,B365CAHH,B365CAHA
E0,14/08/21,Arsenal,Wigan,2,0,H,1.4,4.5,8.0,-0.75,1.90,2.00,-1.0,1.95,1.95
E0,14/08/21,Fulham,Bolton,1,1,D,2.1,3.3,3.6,0.25,2.05,1.85,0.0,1.98,1.92
"""


def test_ingest_keeps_ah_columns():
    out = normalize_season(AH_CSV, "2021-22")
    for col in ("AHh", "AHCh", "B365AHH", "B365AHA", "B365CAHH", "B365CAHA"):
        assert col in out.columns, col
    # Negative and quarter lines survive untouched (no 1.01 odds floor).
    assert out.loc[0, "AHh"] == -0.75
    assert out.loc[1, "AHh"] == 0.25
    assert out.loc[0, "AHCh"] == -1.0


def test_add_ah_features_settles_and_scores():
    df = add_ah_features(normalize_season(AH_CSV, "2021-22"))
    # Arsenal win 2-0 at -0.75 -> full win; draw 1-1 at +0.25 -> half win.
    assert df.loc[0, "ah_result"] == 1.0
    assert df.loc[1, "ah_result"] == 0.5
    assert df.loc[0, "ah_covered"] == 1.0 and df.loc[1, "ah_covered"] == 1.0
    p = df["p_ahcover_B365"]
    assert ((p > 0) & (p < 1)).all()


def test_add_ah_features_no_ops_without_raw_columns():
    # A frame lacking every raw AH column still returns cleanly (NaN truth,
    # no P(cover) columns) — mirrors enriched.parquet before ingest re-runs.
    df = pd.DataFrame(
        {"home_goals": [1, 2], "away_goals": [0, 2], "season": "2020-21"}
    )
    out = add_ah_features(df)
    assert out["ah_line"].isna().all()
    assert out["ah_result"].isna().all()
    assert not [c for c in out.columns if c.startswith("p_ahcover_")]


# ---- efficiency recovers a calibrated slope ---------------------------------


def test_efficiency_recovers_slope_one_on_calibrated_data():
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.uniform(0.15, 0.85, n)
    covered = (rng.uniform(size=n) < p).astype(float)  # perfectly calibrated
    # Two-way even-ish prices whose de-margined home prob equals p.
    df = pd.DataFrame(
        {
            "home_goals": np.zeros(n, int),
            "away_goals": np.zeros(n, int),
            "p_ahcover_Avg": p,
            "ah_covered": covered,
        }
    )
    eff = efficiency_ah(df, "Avg")
    assert abs(eff["slope"] - 1.0) < 0.1
    assert abs(eff["intercept"]) < 0.1
