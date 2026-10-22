"""Bookmaker weighting schemes, raced out-of-sample.

Five ways to turn several books' de-margined probabilities into one
prediction, all evaluated on identical test matches with expanding-window
splits. Hyperparameters (ridge strength, softmax temperature) are chosen
automatically per split by validating on the last training season — no
manual tuning, no test-set leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from src import config
from src.evaluate import expanding_season_splits, log_loss_per_match, outcome_onehot

RIDGE_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 10.0]
TEMP_GRID = [0.002, 0.005, 0.01, 0.02, 0.05]

_CLASS_ORDER = ["A", "D", "H"]  # sklearn sorts labels alphabetically


def prob_cols(book: str) -> list[str]:
    return [f"p_{book}_{o}" for o in config.OUTCOMES]


def panel_universe(df: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    """Matches where every panel book quotes all three outcomes."""
    needed = [c for b in panel for c in prob_cols(b)]
    return df.dropna(subset=needed).sort_values("date")


def _book_probs(frame: pd.DataFrame, book: str) -> np.ndarray:
    return frame[prob_cols(book)].to_numpy(float)


def _renorm(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def _weighted(frame: pd.DataFrame, panel: list[str], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    out = np.zeros((len(frame), 3))
    for book in panel:
        out += (weights[book] / total) * _book_probs(frame, book)
    return _renorm(out)


# ---- schemes -------------------------------------------------------------


def predict_equal(test: pd.DataFrame, panel: list[str]) -> np.ndarray:
    return _weighted(test, panel, {b: 1.0 for b in panel})


def margin_weights(frame: pd.DataFrame, panel: list[str]) -> dict[str, float]:
    return {
        b: float(1.0 / (1.0 + frame[f"margin_{b}"].mean())) for b in panel
    }


def predict_margin(test: pd.DataFrame, panel: list[str]) -> np.ndarray:
    # Row-wise margin weighting, matching add_market_features' logic.
    num = np.zeros((len(test), 3))
    den = np.zeros((len(test), 1))
    for book in panel:
        w = (1.0 / (1.0 + test[f"margin_{book}"])).to_numpy()[:, None]
        num += w * _book_probs(test, book)
        den += w
    return _renorm(num / den)


def performance_weights(
    train: pd.DataFrame, panel: list[str], temperature: float
) -> dict[str, float]:
    """softmax(-log_loss / T) over the books, from training matches only."""
    losses = {
        b: log_loss_per_match(_book_probs(train, b), train["result"]).mean()
        for b in panel
    }
    best = min(losses.values())
    raw = {b: np.exp(-(ll - best) / temperature) for b, ll in losses.items()}
    total = sum(raw.values())
    return {b: float(v / total) for b, v in raw.items()}


def predict_performance(
    train: pd.DataFrame, test: pd.DataFrame, panel: list[str]
) -> tuple[np.ndarray, dict[str, float], float]:
    """Auto-tunes the temperature on the last training season."""
    seasons = sorted(train["season"].unique())
    fit, val = train[train["season"].isin(seasons[:-1])], train[train["season"] == seasons[-1]]
    best_t, best_ll = TEMP_GRID[0], np.inf
    for t in TEMP_GRID:
        w = performance_weights(fit, panel, t)
        ll = log_loss_per_match(_weighted(val, panel, w), val["result"]).mean()
        if ll < best_ll:
            best_t, best_ll = t, ll
    weights = performance_weights(train, panel, best_t)
    return _weighted(test, panel, weights), weights, best_t


def _logodds_features(frame: pd.DataFrame, panel: list[str]) -> np.ndarray:
    cols = []
    for book in panel:
        p = _book_probs(frame, book)
        cols.append(np.log(p[:, 0] / p[:, 2]))  # log(pH/pA)
        cols.append(np.log(p[:, 1] / p[:, 2]))  # log(pD/pA)
    return np.column_stack(cols)


def predict_stacked(
    train: pd.DataFrame, test: pd.DataFrame, panel: list[str]
) -> tuple[np.ndarray, dict[str, float], float]:
    """Ridge multinomial stacking; C auto-tuned on the last training season.

    Returns (probabilities in H/D/A order, per-book influence, chosen C).
    Influence = normalized sum of |coefficients| on a book's two features —
    how much the fit leans on that book, not a probability weight.
    """
    from sklearn.linear_model import LogisticRegression

    y_train = pd.Categorical(train["result"], categories=_CLASS_ORDER).codes
    seasons = sorted(train["season"].unique())
    val_mask = train["season"] == seasons[-1]
    X_all = _logodds_features(train, panel)

    best_c, best_ll = RIDGE_C_GRID[0], np.inf
    for c in RIDGE_C_GRID:
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit(X_all[~val_mask.to_numpy()], y_train[~val_mask.to_numpy()])
        probs = model.predict_proba(X_all[val_mask.to_numpy()])[:, [2, 1, 0]]
        ll = log_loss_per_match(probs, train.loc[val_mask, "result"]).mean()
        if ll < best_ll:
            best_c, best_ll = c, ll

    model = LogisticRegression(C=best_c, max_iter=2000)
    model.fit(X_all, y_train)
    probs = model.predict_proba(_logodds_features(test, panel))[:, [2, 1, 0]]

    coef = np.abs(model.coef_).sum(axis=0)
    per_book = {
        b: float(coef[2 * i] + coef[2 * i + 1]) for i, b in enumerate(panel)
    }
    total = sum(per_book.values()) or 1.0
    influence = {b: v / total for b, v in per_book.items()}
    return probs, influence, best_c


def optimal_weights(
    train: pd.DataFrame, panel: list[str], shrink: float = 0.0
) -> dict[str, float]:
    """Directly search the weight simplex for the blend minimizing log loss.

    The objective -log(sum_b w_b p_b(realized outcome)) is convex in w, so
    SLSQP from the uniform start finds the global optimum — the search is
    exact, not heuristic. ``shrink`` optionally penalizes distance from
    equal weights (ridge toward uniform).
    """
    realized = np.column_stack(
        [
            (_book_probs(train, b) * outcome_onehot(train["result"])).sum(axis=1)
            for b in panel
        ]
    )  # (n_matches, n_books): each book's probability of what happened
    uniform = np.full(len(panel), 1.0 / len(panel))

    def nll(w):
        mix = np.clip(realized @ w, 1e-12, None)
        return -np.log(mix).mean() + shrink * np.sum((w - uniform) ** 2)

    res = optimize.minimize(
        nll,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(panel),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return dict(zip(panel, map(float, w)))


SHARP_PREFERENCE = ["PSC", "PS", "AvgC", "B365C", "Avg", "B365", "Max"]


def predict_sharpest(test: pd.DataFrame, panel: list[str]) -> np.ndarray:
    """Row-wise: the single sharpest quote available for each match."""
    order = [b for b in SHARP_PREFERENCE if all(c in test.columns for c in prob_cols(b))]
    order += [b for b in panel if b not in order]
    out = np.full((len(test), 3), np.nan)
    for book in order:
        probs = _book_probs(test, book)
        missing = np.isnan(out).any(axis=1)
        usable = missing & ~np.isnan(probs).any(axis=1)
        out[usable] = probs[usable]
    return out


# ---- the race -------------------------------------------------------------


def race(
    df: pd.DataFrame, panel: list[str], min_train_seasons: int = 3
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict]:
    """Run every scheme on identical expanding-window test matches.

    Returns (summary, per-match losses, per-scheme weights from the final
    refit, tuning info).
    """
    data = panel_universe(df, panel)
    splits = expanding_season_splits(data, min_train_seasons)
    if not splits:
        raise ValueError("not enough seasons in the panel universe")

    preds: dict[str, list[np.ndarray]] = {
        "equal weights": [],
        "margin weights": [],
        "performance weights (auto T)": [],
        "optimized weights (auto-search)": [],
        "ridge stacking (auto C)": [],
        "sharpest single quote": [],
    }
    weights_final: dict[str, dict[str, float]] = {}
    tuning: dict = {}
    weight_history = []

    for train_ix, test_ix in splits:
        train, test = data.loc[train_ix], data.loc[test_ix]
        preds["equal weights"].append(predict_equal(test, panel))
        preds["margin weights"].append(predict_margin(test, panel))
        perf_p, perf_w, temp = predict_performance(train, test, panel)
        preds["performance weights (auto T)"].append(perf_p)
        opt_w = optimal_weights(train, panel)
        preds["optimized weights (auto-search)"].append(
            _weighted(test, panel, opt_w)
        )
        weight_history.append({"test_season": test["season"].iloc[0], **opt_w})
        stack_p, influence, ridge_c = predict_stacked(train, test, panel)
        preds["ridge stacking (auto C)"].append(stack_p)
        preds["sharpest single quote"].append(predict_sharpest(test, panel))

    weights_final = {
        "equal weights": {b: 1 / len(panel) for b in panel},
        "margin weights": {
            b: w / sum(margin_weights(data, panel).values())
            for b, w in margin_weights(data, panel).items()
        },
        "performance weights (auto T)": perf_w,
        "optimized weights (auto-search)": opt_w,
        "ridge stacking (auto C)": influence,
    }
    tuning = {
        "temperature": temp,
        "ridge_C": ridge_c,
        "weight_history": weight_history,
    }

    test_all = data.loc[np.concatenate([list(te) for _, te in splits])]
    losses = {
        name: log_loss_per_match(np.vstack(chunks), test_all["result"])
        for name, chunks in preds.items()
    }
    if all(c in data.columns for c in prob_cols("cons")):
        losses["global consensus (all books)"] = log_loss_per_match(
            test_all[prob_cols("cons")].to_numpy(), test_all["result"]
        )

    summary = pd.DataFrame(
        {
            "log_loss": {k: v.mean() for k, v in losses.items()},
            "n_matches": {k: len(v) for k, v in losses.items()},
        }
    ).sort_values("log_loss")
    weights_df = pd.DataFrame(weights_final).T
    return summary, losses, weights_df, tuning
