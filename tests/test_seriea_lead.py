import numpy as np
import pytest

from src.seriea_lead import era_verdict


def _losses(n, seed, edge=0.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(1.0, 0.15, n)
    model = base + edge + rng.normal(0, 0.02, n)
    return model, base


def test_era_verdict_confirms_persistent_edge():
    n = 3000
    model, base = _losses(n, seed=1, edge=-0.02)
    seasons = np.array(["2015-16"] * (n // 2) + ["2020-21"] * (n // 2))
    out = era_verdict(model, base, seasons, cutoff="2017-18")
    assert out["discovery"]["significantly_better"]
    assert out["confirmation"]["significantly_better"]
    assert out["confirmed"]


def test_era_verdict_rejects_one_era_wonder():
    n = 3000
    seasons = np.array(["2015-16"] * (n // 2) + ["2020-21"] * (n // 2))
    model_d, base_d = _losses(n // 2, seed=2, edge=-0.02)
    model_c, base_c = _losses(n // 2, seed=3, edge=0.0)
    out = era_verdict(
        np.concatenate([model_d, model_c]),
        np.concatenate([base_d, base_c]),
        seasons,
        cutoff="2017-18",
    )
    assert out["discovery"]["significantly_better"]
    assert not out["confirmation"]["significantly_better"]
    assert not out["confirmed"]


def test_era_verdict_reports_both_ns():
    n = 2000
    model, base = _losses(n, seed=4, edge=0.0)
    seasons = np.array(["2015-16", "2020-21"] * (n // 2))
    out = era_verdict(model, base, seasons, cutoff="2017-18")
    assert not out["confirmed"]
    assert out["discovery"]["n"] + out["confirmation"]["n"] == n
