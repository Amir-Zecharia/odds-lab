import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.models import DixonColes, _dc_tau


def _synthetic_league(n_rounds=60, seed=5):
    """Simulate Poisson scores from known team strengths."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    attack = {"A": 0.4, "B": 0.1, "C": -0.1, "D": -0.4}
    defense = {"A": -0.2, "B": 0.0, "C": 0.0, "D": 0.2}
    mu, home_adv = 0.1, 0.25

    rows = []
    date = pd.Timestamp("2018-08-01")
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(mu + home_adv + attack[h] + defense[a])
                m = np.exp(mu + attack[a] + defense[h])
                hg, ag = rng.poisson(lam), rng.poisson(m)
                rows.append(
                    {
                        "date": date,
                        "home_team": h,
                        "away_team": a,
                        "home_goals": hg,
                        "away_goals": ag,
                        "result": "H" if hg > ag else ("A" if ag > hg else "D"),
                    }
                )
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_dixon_coles_recovers_team_ordering():
    df = _synthetic_league()
    model = DixonColes(xi=0.0).fit(df)
    attack, defense, mu, home_adv, rho = model._unpack(model.params_)
    order = [model.teams_.index(t) for t in ["A", "B", "C", "D"]]
    fitted_attack = attack[order]
    # Strongest attacker A, weakest D — ordering must be recovered.
    assert list(np.argsort(-fitted_attack)) == [0, 1, 2, 3]
    assert home_adv == pytest.approx(0.25, abs=0.1)


def test_outcome_probs_are_a_distribution():
    df = _synthetic_league(n_rounds=20)
    model = DixonColes().fit(df)
    probs = model.outcome_probs("A", "D")
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert (probs > 0).all()
    # Best team at home vs worst team: home win must be the favorite.
    assert probs[0] > probs[2]


def test_unseen_team_gets_average_strength():
    df = _synthetic_league(n_rounds=10)
    model = DixonColes().fit(df)
    probs = model.outcome_probs("Newcastle", "Sunderland")
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_tau_correction_only_touches_low_scores():
    lam, m, rho = 1.5, 1.2, -0.08
    hg, ag = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    tau = _dc_tau(hg, ag, lam, m, rho)
    assert (tau[2:, :] == 1).all() and (tau[:, 2:] == 1).all()
    assert tau[0, 0] != 1 and tau[1, 1] != 1


def test_tau_preserves_total_probability():
    """The Dixon-Coles correction reshuffles mass among low scores but the
    full grid must still (approximately) sum to 1."""
    lam, m, rho = 1.4, 1.1, -0.06
    goals = np.arange(15)
    grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, m))
    hg, ag = np.meshgrid(goals, goals, indexing="ij")
    assert (grid * _dc_tau(hg, ag, lam, m, rho)).sum() == pytest.approx(1.0, abs=1e-3)
