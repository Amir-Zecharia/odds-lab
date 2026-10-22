"""Club Elo cross-validation: does an EXTERNAL results-only rating add any
out-of-sample signal beyond the market consensus?

Club Elo (clubelo.com) publishes daily Elo ratings for European clubs built
purely from match results — no odds, no market information. It is therefore
the cleanest possible test of the platform's recurring finding: the market
already prices everything a stats model knows. This is the first EXTERNAL
information source tested here; every prior candidate (our own Elo, drift,
totals) was derived from the same football-data.co.uk feed as the odds.

Protocol mirrors src.research_prediction and src.models:
- Baseline: multinomial logit on the two consensus log-odds ratios.
- Candidate: baseline + the standardized Club Elo rating difference.
- Expanding-window by season, identical test matches, paired-bootstrap CI of
  the candidate against the RAW market consensus; plus an in-sample LR test.

As-of discipline: each club's pre-match rating is the Elo valid on
(match date − 1 day). We never read a rating stamped on or after kickoff, so
there is no leakage from the result being rated into its own prediction.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.evaluate import (
    expanding_season_splits,
    log_loss_per_match,
    paired_bootstrap_diff,
)
from src.models import BASELINE_FEATURES, _fit_mnlogit, _predict_mnlogit

MIN_TRAIN_SEASONS = 5
ELO_FEATURE = "z_elo_clubelo"
CANDIDATE_FEATURES = BASELINE_FEATURES + [ELO_FEATURE]

# Club Elo covers the top divisions well; E1 (Championship) only partially.
COVERED_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
PARTIAL_LEAGUES = ["E1"]

CACHE_DIR = config.RAW_DIR / "clubelo"
RESULTS_JSON = config.PROCESSED_DIR / "clubelo.json"

_API_URL = "http://api.clubelo.com/{name}"
_FETCH_SLEEP_S = 1.0  # be polite: one live request per second
_HTTP_TIMEOUT_S = 30

# Our team names -> Club Elo names where the two feeds disagree. The Club Elo
# API strips spaces from the URL (so "Man City" resolves without a mapping),
# which leaves only genuine spelling/abbreviation differences to fix here.
# Names verified live against api.clubelo.com. Umlauts are transliterated
# (Koeln, Nuernberg, Duesseldorf, Fuerth) and a few clubs use a short form.
CLUBELO_NAME_MAP: dict[str, str] = {
    # England
    "Nott'm Forest": "Forest",
    "Milton Keynes Dons": "MK Dons",
    # Spain
    "Ath Madrid": "Atletico",
    "Ath Bilbao": "Bilbao",
    "Espanol": "Espanyol",
    "Vallecano": "Rayo Vallecano",
    "La Coruna": "Depor",
    "Sp Gijon": "Gijon",
    # Germany
    "FC Koln": "Koeln",
    "M'gladbach": "Gladbach",
    "Fortuna Dusseldorf": "Duesseldorf",
    "Ein Frankfurt": "Frankfurt",
    "Greuther Furth": "Fuerth",
    "Holstein Kiel": "Holstein",
    "Nurnberg": "Nuernberg",
    # France
    "St Etienne": "Saint-Etienne",
}


def _safe_club_filename(clubelo_name: str) -> str:
    """Filesystem-safe cache key for a Club Elo name (one file per club)."""
    return re.sub(r"[^0-9A-Za-z-]+", "_", clubelo_name).strip("_")


def _url_name(clubelo_name: str) -> str:
    """The API path expects the club name with spaces removed (hyphens kept)."""
    return clubelo_name.replace(" ", "")


def _parse_history(csv_text: str) -> pd.DataFrame:
    """Parse a Club Elo CSV into a From/To/Elo history, sorted ascending.

    Header-only responses (unknown club) parse to an empty frame; that is a
    legitimate stable answer, not an error, so callers cache it.
    """
    frame = pd.read_csv(io.StringIO(csv_text))
    if frame.empty or "Elo" not in frame.columns:
        return pd.DataFrame(columns=["Club", "Elo", "From", "To"])
    frame = frame.dropna(subset=["Elo", "From", "To"]).copy()
    frame["Elo"] = pd.to_numeric(frame["Elo"], errors="coerce")
    frame["From"] = pd.to_datetime(frame["From"], errors="coerce")
    frame["To"] = pd.to_datetime(frame["To"], errors="coerce")
    frame = frame.dropna(subset=["Elo", "From", "To"])
    return frame.sort_values("From").reset_index(drop=True)


def fetch_club_history(clubelo_name: str, live: bool = True) -> pd.DataFrame:
    """Club Elo history for one club, cached under data/raw/clubelo/.

    Reads the cached CSV when present (no network). Otherwise, when ``live``,
    fetches once, sleeps to stay polite, and caches the raw response. Resilient
    to 404s, empty bodies and transient network errors — returns an empty frame
    rather than raising, so one absent club never aborts a whole run.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_safe_club_filename(clubelo_name)}.csv"
    if path.exists():
        return _parse_history(path.read_text())
    if not live:
        return pd.DataFrame(columns=["Club", "Elo", "From", "To"])

    url = _API_URL.format(name=urllib.parse.quote(_url_name(clubelo_name)))
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as resp:
            csv_text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return pd.DataFrame(columns=["Club", "Elo", "From", "To"])
    finally:
        time.sleep(_FETCH_SLEEP_S)
    path.write_text(csv_text)  # cache every successful response, incl. "unknown"
    return _parse_history(csv_text)


def _resolve_history(
    team: str, live: bool
) -> tuple[pd.DataFrame, str | None]:
    """Try the mapped Club Elo name first, then the raw team name.

    Returns (history, resolved_name); resolved_name is None when neither the
    mapping nor the raw name yields any Club Elo data (an unmapped team).
    """
    candidates = []
    if team in CLUBELO_NAME_MAP:
        candidates.append(CLUBELO_NAME_MAP[team])
    candidates.append(team)  # raw fallback (Club Elo strips spaces itself)
    for name in candidates:
        hist = fetch_club_history(name, live)
        if not hist.empty:
            return hist, name
    return pd.DataFrame(columns=["Club", "Elo", "From", "To"]), None


def _asof_elos(hist: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
    """Vectorized strict as-of lookup: the Elo whose [From, To] covers each
    target date. NaN when no interval covers it. Ranges are contiguous and
    non-overlapping, so a right-searchsorted on From + a To bound suffices."""
    out = np.full(len(targets), np.nan)
    if hist.empty:
        return out
    froms = hist["From"].to_numpy("datetime64[ns]")
    tos = hist["To"].to_numpy("datetime64[ns]")
    elos = hist["Elo"].to_numpy(float)
    idx = np.searchsorted(froms, targets, side="right") - 1
    valid = idx >= 0
    ii = idx[valid]
    covered = targets[valid] <= tos[ii]
    picked = np.where(covered, elos[ii], np.nan)
    out[valid] = picked
    return out


def attach_club_elo(
    df: pd.DataFrame, live: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Add pre-match Club Elo columns to every match.

    ``elo_home_clubelo`` / ``elo_away_clubelo`` are each side's rating valid on
    (match date − 1 day) — strictly before kickoff, so no result ever informs
    its own rating. ``elo_diff_clubelo`` is home − away. Missing ratings stay
    NaN. Returns (frame, unmapped_teams).
    """
    df = df.copy()
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    histories: dict[str, pd.DataFrame] = {}
    unmapped: list[str] = []
    for team in teams:
        hist, resolved = _resolve_history(team, live)
        histories[team] = hist
        if resolved is None:
            unmapped.append(team)

    # As-of target is the day before the match; lookups are done per team so
    # each club's sorted history is scanned once.
    target = (df["date"] - pd.Timedelta(days=1)).to_numpy("datetime64[ns]")
    home_elo = np.full(len(df), np.nan)
    away_elo = np.full(len(df), np.nan)
    for team, hist in histories.items():
        if hist.empty:
            continue
        h_mask = (df["home_team"] == team).to_numpy()
        if h_mask.any():
            home_elo[h_mask] = _asof_elos(hist, target[h_mask])
        a_mask = (df["away_team"] == team).to_numpy()
        if a_mask.any():
            away_elo[a_mask] = _asof_elos(hist, target[a_mask])

    df["elo_home_clubelo"] = home_elo
    df["elo_away_clubelo"] = away_elo
    df["elo_diff_clubelo"] = home_elo - away_elo
    return df, unmapped


def build_clubelo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Joint universe for one league (or pool): rows with the consensus AND
    both Club Elo ratings, so baseline and candidate score identical matches.

    The Club Elo difference is standardized within the frame. Multinomial logit
    is invariant to an affine feature rescaling, so this is purely for
    interpretable coefficients — it does not leak or move the log loss.
    """
    data = df.copy()
    pH, pD, pA = (data[f"p_cons_{o}"] for o in config.OUTCOMES)
    data["mkt_logit_H"] = np.log(pH / pA)
    data["mkt_logit_D"] = np.log(pD / pA)
    needed = BASELINE_FEATURES + [f"p_cons_{o}" for o in config.OUTCOMES] + [
        "elo_diff_clubelo", "result", "season", "date",
    ]
    data = data.dropna(subset=needed).sort_values("date").copy()
    e = data["elo_diff_clubelo"].to_numpy(float)
    sd = e.std()
    data[ELO_FEATURE] = (e - e.mean()) / sd if sd > 0 else 0.0
    return data


def _naive_sign_check(data: pd.DataFrame) -> dict:
    """Can the sign of the Club Elo difference (higher-rated side) out-pick a
    naive always-home rule? A results-only rating should beat always-home on
    accuracy yet still trail the market on probability — that gap is the point.
    """
    higher = np.where(data["elo_diff_clubelo"] > 0, "H", "A")
    return {
        "clubelo_sign_accuracy": float((higher == data["result"]).mean()),
        "always_home_accuracy": float((data["result"] == "H").mean()),
        "n": int(len(data)),
    }


def run_clubelo(df: pd.DataFrame) -> dict | None:
    """Nested expanding-window comparison on one prepared feature frame.

    ``df`` must already carry the Club Elo columns (see ``attach_club_elo``).
    Returns None when the joint universe lacks enough seasons to test.
    """
    data = build_clubelo_features(df)
    splits = expanding_season_splits(data, MIN_TRAIN_SEASONS)
    if not splits:
        return None

    base_pred, cand_pred = [], []
    for train_ix, test_ix in splits:
        train, test = data.loc[train_ix], data.loc[test_ix]
        base_pred.append(
            _predict_mnlogit(_fit_mnlogit(train, BASELINE_FEATURES), test, BASELINE_FEATURES)
        )
        cand_pred.append(
            _predict_mnlogit(_fit_mnlogit(train, CANDIDATE_FEATURES), test, CANDIDATE_FEATURES)
        )

    test_all = data.loc[np.concatenate([list(te) for _, te in splits])]
    consensus = test_all[[f"p_cons_{o}" for o in config.OUTCOMES]].to_numpy()
    loss_cons = log_loss_per_match(consensus, test_all["result"])
    loss_base = log_loss_per_match(np.vstack(base_pred), test_all["result"])
    loss_cand = log_loss_per_match(np.vstack(cand_pred), test_all["result"])

    vs_cons = paired_bootstrap_diff(loss_cand, loss_cons)
    vs_base = paired_bootstrap_diff(loss_cand, loss_base)

    # In-sample LR test of the added Club Elo feature (dof = K-1 = 2).
    llf_base = _fit_mnlogit(data, BASELINE_FEATURES).llf
    llf_full = _fit_mnlogit(data, CANDIDATE_FEATURES).llf
    lr_stat = 2 * (llf_full - llf_base)
    from scipy import stats

    seasons = sorted(data["season"].unique())
    return {
        "n_universe": int(len(data)),
        "n_test": int(len(test_all)),
        "test_seasons": f"{seasons[MIN_TRAIN_SEASONS]} to {seasons[-1]}",
        "consensus_log_loss": float(loss_cons.mean()),
        "baseline_log_loss": float(loss_base.mean()),
        "candidate_log_loss": float(loss_cand.mean()),
        "candidate_vs_consensus": {
            **vs_cons,
            "significantly_better": vs_cons["hi"] < 0,
            "significantly_worse": vs_cons["lo"] > 0,
        },
        "candidate_vs_baseline": {
            **vs_base,
            "significantly_better": vs_base["hi"] < 0,
            "significantly_worse": vs_base["lo"] > 0,
        },
        "lr_test": {
            "lr_stat": float(lr_stat),
            "df": 2,
            "p_value": float(stats.chi2.sf(lr_stat, 2)),
        },
        "naive": _naive_sign_check(data),
    }


def _coverage(df: pd.DataFrame) -> float:
    """Fraction of matches with BOTH Club Elo ratings present."""
    if df.empty:
        return 0.0
    both = df["elo_home_clubelo"].notna() & df["elo_away_clubelo"].notna()
    return float(both.mean())


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "enriched.parquet")
    leagues = COVERED_LEAGUES + PARTIAL_LEAGUES
    df = df[df["league"].isin(leagues)].copy()

    network_error: str | None = None
    try:
        attached, unmapped = attach_club_elo(df, live=True)
    except Exception as exc:  # never hard-fail on the external feed
        network_error = f"{type(exc).__name__}: {exc}"
        attached, unmapped = df.assign(
            elo_home_clubelo=np.nan,
            elo_away_clubelo=np.nan,
            elo_diff_clubelo=np.nan,
        ), sorted(set(df["home_team"]) | set(df["away_team"]))

    coverage = {lg: _coverage(attached[attached["league"] == lg]) for lg in leagues}
    coverage["ALL"] = _coverage(attached[attached["league"].isin(COVERED_LEAGUES)])

    per_league = {}
    for lg in leagues:
        res = run_clubelo(attached[attached["league"] == lg])
        if res is not None:
            per_league[lg] = res
    pooled = run_clubelo(attached[attached["league"].isin(COVERED_LEAGUES)])

    results = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "covered_leagues": COVERED_LEAGUES,
        "partial_leagues": PARTIAL_LEAGUES,
        "coverage": coverage,
        "n_matches": int(len(attached)),
        "unmapped_teams": sorted(unmapped),
        "leagues": per_league,
        "pooled": pooled,
        "network_error": network_error,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    print(f"wrote {RESULTS_JSON}")
    print(f"network_error: {network_error}")
    for lg in leagues:
        print(f"  coverage {lg}: {coverage[lg]:.1%}")
    print(f"  coverage ALL(covered): {coverage['ALL']:.1%}")
    print(f"  unmapped teams ({len(unmapped)}): {sorted(unmapped)}")
    if pooled:
        c = pooled["candidate_vs_consensus"]
        print(
            f"  pooled n_test={pooled['n_test']} "
            f"consensus={pooled['consensus_log_loss']:.4f} "
            f"candidate={pooled['candidate_log_loss']:.4f} "
            f"diff={c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] "
            f"LR p={pooled['lr_test']['p_value']:.3g}"
        )


if __name__ == "__main__":
    main()
