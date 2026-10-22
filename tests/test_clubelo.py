import numpy as np
import pandas as pd
import pytest

from src import config
from src import clubelo
from src.clubelo import (
    CANDIDATE_FEATURES,
    _resolve_history,
    _safe_club_filename,
    attach_club_elo,
    build_clubelo_features,
    fetch_club_history,
    run_clubelo,
)


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    """Point the cache at tmp so no test ever touches the network."""
    monkeypatch.setattr(clubelo, "CACHE_DIR", tmp_path / "clubelo")
    (tmp_path / "clubelo").mkdir()
    return tmp_path / "clubelo"


def _write_history(cache_dir, clubelo_name, rows):
    """rows: list of (elo, from_iso, to_iso). Writes a Club Elo-shaped CSV."""
    lines = ["Rank,Club,Country,Level,Elo,From,To"]
    for elo, frm, to in rows:
        lines.append(f"None,{clubelo_name},ENG,1,{elo},{frm},{to}")
    path = cache_dir / f"{_safe_club_filename(clubelo_name)}.csv"
    path.write_text("\n".join(lines) + "\n")


def test_name_mapping_prefers_dict_then_falls_back_to_raw(_cache_in_tmp):
    # "Ath Madrid" is only cached under its mapped Club Elo name "Atletico".
    _write_history(_cache_in_tmp, "Atletico", [(1900.0, "2020-01-01", "2020-12-31")])
    hist, resolved = _resolve_history("Ath Madrid", live=False)
    assert resolved == "Atletico"
    assert not hist.empty

    # A team with no map entry resolves via its raw name.
    _write_history(_cache_in_tmp, "Arsenal", [(1800.0, "2020-01-01", "2020-12-31")])
    hist, resolved = _resolve_history("Arsenal", live=False)
    assert resolved == "Arsenal"
    assert not hist.empty

    # Absent everywhere -> unmapped (None), empty history, no crash.
    hist, resolved = _resolve_history("Nowhere United", live=False)
    assert resolved is None
    assert hist.empty


def test_attach_picks_temporally_correct_rating_and_nan_when_uncovered(_cache_in_tmp):
    # Two contiguous rating periods for one club.
    _write_history(
        _cache_in_tmp,
        "Arsenal",
        [(1700.0, "2020-01-01", "2020-06-30"), (1750.0, "2020-07-01", "2020-12-31")],
    )
    _write_history(_cache_in_tmp, "Chelsea", [(1720.0, "2020-01-01", "2020-12-31")])
    df = pd.DataFrame(
        {
            # as-of date is match date - 1 day: 2020-07-02 -> lookup 2020-07-01
            # falls in the SECOND period (1750); 2019 match is uncovered.
            "date": pd.to_datetime(["2020-07-02", "2019-05-05"]),
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Chelsea"],
        }
    )
    out, unmapped = attach_club_elo(df, live=False)
    assert out.loc[0, "elo_home_clubelo"] == pytest.approx(1750.0)
    assert out.loc[0, "elo_away_clubelo"] == pytest.approx(1720.0)
    assert out.loc[0, "elo_diff_clubelo"] == pytest.approx(30.0)
    # 2019 predates every rating period -> NaN, not a crash.
    assert np.isnan(out.loc[1, "elo_home_clubelo"])
    assert unmapped == []


def test_leakage_guard_chosen_rating_predates_the_match(_cache_in_tmp):
    _write_history(
        _cache_in_tmp,
        "Arsenal",
        [(1700.0, "2020-01-01", "2020-06-30"), (1750.0, "2020-07-01", "2020-12-31")],
    )
    _write_history(_cache_in_tmp, "Chelsea", [(1720.0, "2020-01-01", "2020-12-31")])
    match_date = pd.Timestamp("2020-07-01")  # boundary: rating starts today
    df = pd.DataFrame(
        {"date": [match_date], "home_team": ["Arsenal"], "away_team": ["Chelsea"]}
    )
    out, _ = attach_club_elo(df, live=False)
    # As-of is 2020-06-30, so the FIRST (older) rating is used, not the one
    # stamped on the match day itself.
    assert out.loc[0, "elo_home_clubelo"] == pytest.approx(1700.0)
    hist, _ = _resolve_history("Arsenal", live=False)
    used = hist[hist["Elo"] == 1700.0].iloc[0]
    assert used["From"] < match_date


def _synthetic_universe(n=6000, seed=3):
    """Calibrated market + an injected Club Elo diff correlated with outcome."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([5.0, 3.0, 4.0], size=n)
    outcomes = np.array(config.OUTCOMES)
    result = np.array([outcomes[rng.choice(3, p=p)] for p in raw])
    df = pd.DataFrame(
        {
            "date": pd.Timestamp("2015-08-01") + pd.to_timedelta(np.arange(n), "h"),
            "season": [f"s{i * 8 // n}" for i in range(n)],
            "result": result,
            "home_team": "H",
            "away_team": "A",
        }
    )
    for i, o in enumerate(config.OUTCOMES):
        df[f"p_cons_{o}"] = raw[:, i]
    df["elo_diff_clubelo"] = (raw[:, 0] - raw[:, 2]) * 400 + rng.normal(0, 50, n)
    return df


def test_build_features_standardizes_and_keeps_joint_universe():
    data = build_clubelo_features(_synthetic_universe(n=500))
    assert len(data) == 500
    assert not data[CANDIDATE_FEATURES].isna().any().any()
    assert data["z_elo_clubelo"].mean() == pytest.approx(0.0, abs=1e-9)
    assert data["z_elo_clubelo"].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_run_clubelo_returns_expected_keys():
    res = run_clubelo(_synthetic_universe())
    assert res is not None
    for key in (
        "n_test",
        "consensus_log_loss",
        "baseline_log_loss",
        "candidate_log_loss",
        "candidate_vs_consensus",
        "candidate_vs_baseline",
        "lr_test",
        "naive",
    ):
        assert key in res
    assert {"diff", "lo", "hi", "significantly_better"} <= set(
        res["candidate_vs_consensus"]
    )
    assert {"lr_stat", "df", "p_value"} <= set(res["lr_test"])
    assert {"clubelo_sign_accuracy", "always_home_accuracy", "n"} <= set(res["naive"])
