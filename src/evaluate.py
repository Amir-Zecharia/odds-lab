"""Scoring rules, time-ordered splits, and bootstrap uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

RNG_SEED = 20260704


def outcome_onehot(results: pd.Series) -> np.ndarray:
    """(n, 3) one-hot matrix in H/D/A order."""
    return np.column_stack([(results == o).to_numpy(int) for o in config.OUTCOMES])


def log_loss_per_match(probs: np.ndarray, results: pd.Series) -> np.ndarray:
    """Negative log likelihood of the realized outcome, per match."""
    y = outcome_onehot(results)
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.log((p * y).sum(axis=1))


def brier_per_match(probs: np.ndarray, results: pd.Series) -> np.ndarray:
    """Multiclass Brier score (sum of squared errors over H/D/A), per match."""
    y = outcome_onehot(results)
    return ((np.asarray(probs, dtype=float) - y) ** 2).sum(axis=1)


def log_loss(probs: np.ndarray, results: pd.Series) -> float:
    return float(log_loss_per_match(probs, results).mean())


def brier(probs: np.ndarray, results: pd.Series) -> float:
    return float(brier_per_match(probs, results).mean())


def time_ordered_split(
    df: pd.DataFrame, test_fraction: float = 0.25
) -> tuple[pd.Index, pd.Index]:
    """Single chronological split: earliest matches train, latest test."""
    order = df.sort_values("date").index
    cut = int(round(len(order) * (1 - test_fraction)))
    return order[:cut], order[cut:]


def expanding_season_splits(
    df: pd.DataFrame, min_train_seasons: int = 5
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window splits: train on seasons < s, test on season s."""
    seasons = sorted(df["season"].unique())
    splits = []
    for i in range(min_train_seasons, len(seasons)):
        train = df.index[df["season"].isin(seasons[:i])]
        test = df.index[df["season"] == seasons[i]]
        if len(test):
            splits.append((train, test))
    return splits


def bootstrap_ci(
    per_match_values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Mean with percentile bootstrap CI -> (mean, lo, hi)."""
    values = np.asarray(per_match_values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_bootstrap_diff(
    per_match_a: np.ndarray,
    per_match_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict:
    """CI on mean(a) - mean(b) resampling matches jointly (paired).

    Negative diff means model a scores better (lower loss) than b.
    """
    a = np.asarray(per_match_a, dtype=float)
    b = np.asarray(per_match_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal-length inputs")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    diffs = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_better": float((diffs < 0).mean()),
    }


def unpaired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict:
    """CI on mean(a) - mean(b) for two independent samples (may differ in n)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    means_b = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    diffs = means_a - means_b
    return {
        "diff": float(a.mean() - b.mean()),
        "lo": float(np.quantile(diffs, alpha / 2)),
        "hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_better": float((diffs < 0).mean()),
    }
