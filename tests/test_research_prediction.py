import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import expanding_season_splits
from src.models import _fit_mnlogit
from src.research_prediction import (
    BASE_FEATURES,
    CANDIDATES,
    MIN_TRAIN_SEASONS,
    build_features,
)


def _synthetic_market(n=8000, seed=11):
    """Matches whose outcomes are drawn from the market's own probabilities,
    i.e. an already-calibrated market."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([5.0, 3.0, 4.0], size=n)
    outcomes = np.array(config.OUTCOMES)
    result = [outcomes[rng.choice(3, p=p)] for p in raw]
    dates = pd.Timestamp("2015-08-01") + pd.to_timedelta(np.arange(n), "h")
    df = pd.DataFrame(
        {
            "date": dates,
            "season": [f"s{i * 8 // n}" for i in range(n)],
            "result": result,
        }
    )
    for i, o in enumerate(config.OUTCOMES):
        df[f"p_PS_{o}"] = raw[:, i]
        df[f"p_PSC_{o}"] = raw[:, i]
        df[f"p_cons_{o}"] = raw[:, i]
    df["p_over_PC"] = rng.uniform(0.3, 0.7, size=n)
    df["p_over_Avg"] = df["p_over_PC"]
    return df


def test_build_features_drops_nan_rows():
    df = _synthetic_market(n=200)
    df.loc[df.index[:20], "p_PSC_H"] = np.nan  # kills drift + best price
    df.loc[df.index[20:40], "p_over_PC"] = np.nan
    df.loc[df.index[20:40], "p_over_Avg"] = np.nan  # kills totals fallback
    out = build_features(df)
    assert len(out) == 160
    for features in CANDIDATES.values():
        assert not out[features].isna().any().any()


def test_totals_fallback_keeps_rows_with_avg_only():
    df = _synthetic_market(n=100)
    df.loc[df.index[:50], "p_over_PC"] = np.nan  # Avg still present
    out = build_features(df)
    assert len(out) == 100
    assert not out["over_logit"].isna().any()


def test_recalibration_recovers_identity_on_calibrated_data():
    data = build_features(_synthetic_market())
    res = _fit_mnlogit(data, BASE_FEATURES)
    # params: rows (const, best_logit_H, best_logit_D), columns are the
    # D-vs-A and H-vs-A equations (class order A/D/H).
    params = np.asarray(res.params)
    assert params[2, 0] == pytest.approx(1.0, abs=0.2)  # logit_D in D eq
    assert params[1, 1] == pytest.approx(1.0, abs=0.2)  # logit_H in H eq
    assert params[1, 0] == pytest.approx(0.0, abs=0.2)
    assert params[2, 1] == pytest.approx(0.0, abs=0.2)
    assert params[0, :] == pytest.approx(0.0, abs=0.2)  # intercepts


def test_splits_have_no_date_leakage():
    path = config.PROCESSED_DIR / "enriched.parquet"
    if not path.exists():
        pytest.skip("enriched dataset not built")
    data = build_features(pd.read_parquet(path))
    splits = expanding_season_splits(data, MIN_TRAIN_SEASONS)
    assert splits
    for train_ix, test_ix in splits:
        assert data.loc[train_ix, "date"].max() < data.loc[test_ix, "date"].min()
