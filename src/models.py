"""Analysis models: nested market-beating comparison, XGBoost, Dixon-Coles.

Analysis 1 (nested comparison) protocol:
- Baseline: multinomial logit of outcome on the two consensus log-odds
  ratios log(pH/pA), log(pD/pA). If the market is efficient, no added
  feature should improve out-of-sample log loss.
- Each candidate model adds features on top of the baseline; in-sample
  likelihood-ratio tests plus out-of-sample expanding-window log loss.
- XGBoost on the same features as a nonlinear robustness check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from scipy.stats import poisson

from src import config
from src.evaluate import expanding_season_splits, log_loss_per_match

# Outcome integer coding (alphabetical, matching statsmodels/sklearn column
# order for sorted classes): A=0, D=1, H=2.
_CLASS_ORDER = ["A", "D", "H"]

BASELINE_FEATURES = ["mkt_logit_H", "mkt_logit_D"]
FEATURE_SETS = {
    "baseline (consensus only)": [],
    "+ dispersion": ["disp_std"],
    "+ sharp-soft gap": ["sharp_soft_gap"],
    "+ Elo": ["elo_diff"],
    "+ form": ["form_diff"],
    "+ rest days": ["rest_diff"],
    "+ all features": [
        "disp_std", "sharp_soft_gap", "elo_diff", "form_diff", "rest_diff",
    ],
}


def model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for modeling: consensus + all engineered features present."""
    df = df.copy()
    pH, pD, pA = (df[f"p_cons_{o}"] for o in config.OUTCOMES)
    df["mkt_logit_H"] = np.log(pH / pA)
    df["mkt_logit_D"] = np.log(pD / pA)
    needed = BASELINE_FEATURES + FEATURE_SETS["+ all features"] + [
        f"p_cons_{o}" for o in config.OUTCOMES
    ]
    return df.dropna(subset=needed)


def _to_hda(probs_adh: np.ndarray) -> np.ndarray:
    """Reorder columns from A/D/H (class-sorted) to H/D/A (config order)."""
    return probs_adh[:, [2, 1, 0]]


def _fit_mnlogit(train: pd.DataFrame, features: list[str]):
    y = pd.Categorical(train["result"], categories=_CLASS_ORDER).codes
    X = sm.add_constant(train[features].to_numpy(float), has_constant="add")
    return sm.MNLogit(y, X).fit(method="newton", maxiter=200, disp=False)


def _predict_mnlogit(res, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = sm.add_constant(test[features].to_numpy(float), has_constant="add")
    return _to_hda(np.asarray(res.predict(X)))


def likelihood_ratio_tests(df: pd.DataFrame) -> pd.DataFrame:
    """In-sample LR test of each feature set against the baseline."""
    base = _fit_mnlogit(df, BASELINE_FEATURES)
    rows = []
    for name, extra in FEATURE_SETS.items():
        if not extra:
            continue
        full = _fit_mnlogit(df, BASELINE_FEATURES + extra)
        lr = 2 * (full.llf - base.llf)
        dof = 2 * len(extra)  # (K-1) logit equations per added feature
        rows.append(
            {
                "model": name,
                "lr_stat": float(lr),
                "df": dof,
                "p_value": float(stats.chi2.sf(lr, dof)),
                "llf_baseline": float(base.llf),
                "llf_full": float(full.llf),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def _fit_predict_xgb(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> np.ndarray:
    from xgboost import XGBClassifier

    y = pd.Categorical(train["result"], categories=_CLASS_ORDER).codes
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=0,
        verbosity=0,
    )
    model.fit(train[features].to_numpy(float), y)
    return _to_hda(model.predict_proba(test[features].to_numpy(float)))


def out_of_sample_comparison(
    df: pd.DataFrame, min_train_seasons: int = 5, include_xgb: bool = True
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Expanding-window out-of-sample log loss for every model.

    Returns (summary frame, per-match losses keyed by model name); the
    per-match losses feed paired bootstrap CIs against the raw consensus.
    """
    data = model_dataset(df)
    splits = expanding_season_splits(data, min_train_seasons)
    test_idx = np.concatenate([list(te) for _, te in splits])
    test_all = data.loc[test_idx]

    preds: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_SETS}
    if include_xgb:
        preds["XGBoost (all features)"] = []

    for train_ix, test_ix in splits:
        train, test = data.loc[train_ix], data.loc[test_ix]
        for name, extra in FEATURE_SETS.items():
            res = _fit_mnlogit(train, BASELINE_FEATURES + extra)
            preds[name].append(_predict_mnlogit(res, test, BASELINE_FEATURES + extra))
        if include_xgb:
            feats = BASELINE_FEATURES + FEATURE_SETS["+ all features"]
            preds["XGBoost (all features)"].append(
                _fit_predict_xgb(train, test, feats)
            )

    losses: dict[str, np.ndarray] = {}
    # The market itself, unfitted: de-margined consensus probabilities.
    consensus_probs = test_all[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy()
    losses["market consensus (no fit)"] = log_loss_per_match(
        consensus_probs, test_all["result"]
    )
    for name, chunks in preds.items():
        losses[name] = log_loss_per_match(np.vstack(chunks), test_all["result"])

    summary = pd.DataFrame(
        {
            "log_loss": {k: v.mean() for k, v in losses.items()},
            "n_matches": {k: len(v) for k, v in losses.items()},
        }
    ).sort_values("log_loss")
    return summary, losses


def dispersion_error_regression(df: pd.DataFrame) -> dict:
    """Analysis 3: does cross-book disagreement predict market error?

    OLS of per-match consensus Brier score on dispersion, controlling for
    match lopsidedness (max consensus prob), HC1 robust errors.
    """
    from src.evaluate import brier_per_match

    cols = [f"p_cons_{o}" for o in config.OUTCOMES]
    sub = df.dropna(subset=cols + ["disp_std"])
    err = brier_per_match(sub[cols].to_numpy(), sub["result"])
    fav = sub[cols].max(axis=1).to_numpy()
    X = sm.add_constant(np.column_stack([sub["disp_std"].to_numpy(), fav]))
    res = sm.OLS(err, X).fit(cov_type="HC1")
    return {
        "coef_dispersion": float(res.params[1]),
        "se_dispersion": float(res.bse[1]),
        "t_dispersion": float(res.tvalues[1]),
        "p_dispersion": float(res.pvalues[1]),
        "coef_favprob": float(res.params[2]),
        "r_squared": float(res.rsquared),
        "n": int(res.nobs),
    }


# ---------------------------------------------------------------------------
# Dixon-Coles (1997): bivariate Poisson-with-correction goal model, fitted
# without any odds information. Quantifies how far a stats-only model
# trails the market.
# ---------------------------------------------------------------------------

DC_XI = 0.0019  # per-day time-decay: weight halves in ~1 year
MAX_GOALS = 10


def _dc_tau(hg, ag, lam, mu, rho):
    """Dixon-Coles low-score dependence correction."""
    tau = np.ones_like(lam)
    tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
    tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
    tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
    tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
    return tau


class DixonColes:
    def __init__(self, xi: float = DC_XI):
        self.xi = xi
        self.teams_: list[str] = []
        self.params_: np.ndarray | None = None

    def _unpack(self, params):
        n = len(self.teams_)
        attack = params[:n] - params[:n].mean()
        defense = params[n : 2 * n] - params[n : 2 * n].mean()
        mu, home_adv, rho = params[2 * n : 2 * n + 3]
        return attack, defense, mu, home_adv, rho

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None):
        as_of = as_of or matches["date"].max()
        self.teams_ = sorted(
            set(matches["home_team"]) | set(matches["away_team"])
        )
        idx = {t: i for i, t in enumerate(self.teams_)}
        h = matches["home_team"].map(idx).to_numpy()
        a = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(float)
        ag = matches["away_goals"].to_numpy(float)
        days = (as_of - matches["date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)

        def nll(params):
            attack, defense, mu, home_adv, rho = self._unpack(params)
            lam = np.exp(mu + home_adv + attack[h] + defense[a])
            m = np.exp(mu + attack[a] + defense[h])
            tau = np.clip(_dc_tau(hg, ag, lam, m, rho), 1e-10, None)
            ll = (
                np.log(tau)
                + poisson.logpmf(hg, lam)
                + poisson.logpmf(ag, m)
            )
            return -(w * ll).sum()

        n = len(self.teams_)
        x0 = np.concatenate([np.zeros(2 * n), [0.1, 0.25, -0.05]])
        res = optimize.minimize(nll, x0, method="L-BFGS-B")
        self.params_ = res.x
        return self

    def score_grid(self, home: str, away: str) -> np.ndarray:
        """Full scoreline distribution P(home goals=i, away goals=j).

        The origination primitive: every market's fair price is a sum over
        this grid, so all markets derived from it are mutually consistent.
        Unseen teams get league-average strength.
        """
        attack, defense, mu, home_adv, rho = self._unpack(self.params_)
        idx = {t: i for i, t in enumerate(self.teams_)}

        def strength(team):
            i = idx.get(team)
            return (attack[i], defense[i]) if i is not None else (0.0, 0.0)

        att_h, def_h = strength(home)
        att_a, def_a = strength(away)
        lam = np.exp(mu + home_adv + att_h + def_a)
        m = np.exp(mu + att_a + def_h)

        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, m))
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        grid = grid * _dc_tau(hg, ag, lam, m, rho)
        return grid / grid.sum()

    def outcome_probs(self, home: str, away: str) -> np.ndarray:
        """P(H), P(D), P(A). Unseen teams get league-average strength."""
        grid = self.score_grid(home, away)
        return np.array(
            [
                np.tril(grid, -1).sum(),  # home goals > away goals
                np.trace(grid),
                np.triu(grid, 1).sum(),
            ]
        )


def dixon_coles_benchmark(
    df: pd.DataFrame,
    test_seasons: list[str] | None = None,
    max_train_seasons: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Refit Dixon-Coles before each test season; compare vs consensus.

    ``max_train_seasons`` caps training to a trailing window — with the
    default one-year half-life decay, seasons beyond ~6 years back carry
    <2% weight anyway, and the cap keeps pooled multi-league fits tractable.
    """
    cols = [f"p_cons_{o}" for o in config.OUTCOMES]
    data = df.dropna(subset=cols).sort_values("date")
    seasons = sorted(data["season"].unique())
    test_seasons = test_seasons or seasons[5:]

    dc_chunks, mkt_chunks, results = [], [], []
    for season in test_seasons:
        train = data[data["season"] < season]
        if max_train_seasons is not None:
            window = sorted(train["season"].unique())[-max_train_seasons:]
            train = train[train["season"].isin(window)]
        test = data[data["season"] == season]
        if train.empty or test.empty:
            continue
        model = DixonColes().fit(train)
        dc_chunks.append(
            np.vstack(
                [
                    model.outcome_probs(r.home_team, r.away_team)
                    for r in test.itertuples()
                ]
            )
        )
        mkt_chunks.append(test[cols].to_numpy())
        results.append(test["result"])

    results = pd.concat(results)
    losses = {
        "Dixon-Coles (odds-free)": log_loss_per_match(
            np.vstack(dc_chunks), results
        ),
        "market consensus": log_loss_per_match(np.vstack(mkt_chunks), results),
    }
    summary = pd.DataFrame(
        {
            "log_loss": {k: v.mean() for k, v in losses.items()},
            "n_matches": {k: len(v) for k, v in losses.items()},
        }
    ).sort_values("log_loss")
    return summary, losses
