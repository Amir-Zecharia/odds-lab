"""Download and normalize football-data.co.uk CSVs into one clean Parquet.

Handles the known quirks of the source data:
- ``BbAv*``/``BbMx*`` market aggregates renamed to ``Avg*``/``Max*`` (the
  source switched naming around 2019-20).
- Aggregates computed from individual book columns where the source
  provides none (seasons before 2005-06).
- Mixed date formats (dd/mm/yy vs dd/mm/yyyy) and stray blank rows.
- Team-name inconsistencies across seasons.
"""

from __future__ import annotations

import csv
import io
import re
import sys

import pandas as pd
import requests

from src import config

TEAM_NAME_FIXES = {
    "Middlesboro": "Middlesbrough",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheff Utd": "Sheffield United",
    "Sheff United": "Sheffield United",
    "Sheff Weds": "Sheffield Weds",
    "QP Rangers": "QPR",
}

CORE_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
}

# Match statistics the source has carried since 2000 but the platform never
# used: a second outcome per match (half time), the shot counts that proxy
# chance quality, and the discipline/corner counts that drive the thinnest
# markets in football. Optional — missing in a few early files — so every
# one is coerced with NaN on absence rather than assumed present.
STAT_COLUMNS = {
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    "Referee": "referee",
}
_NUMERIC_STATS = {v for k, v in STAT_COLUMNS.items() if k not in ("HTR", "Referee")}

# Bb-era market aggregate columns -> modern names.
BB_RENAMES = {
    "BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA",
    "BbMxH": "MaxH", "BbMxD": "MaxD", "BbMxA": "MaxA",
    "BbAv>2.5": "Avg>2.5", "BbAv<2.5": "Avg<2.5",
    "BbMx>2.5": "Max>2.5", "BbMx<2.5": "Max<2.5",
}

# Over/under 2.5 goals odds (Pinnacle appears as plain "P" here).
_TOTALS_COL_RE = re.compile(r"^(B365|P|Avg|Max)C?[<>]2\.5$")

# Asian-handicap columns: two-way home/away prices (``B365AHH``/``B365AHA``,
# closing ``B365CAHH`` ...) plus the signed handicap line applied to the home
# team (``AHh`` opening, ``AHCh`` closing). Same book family as totals.
_AH_COL_RE = re.compile(r"^((B365|P|Avg|Max)C?AH[HA]|AHC?h)$")
# The line columns carry signed handicaps (negatives and quarter values are
# legitimate), so they must skip the decimal-odds sanity floor below.
_AH_LINE_COLS = frozenset({"AHh", "AHCh"})

AGGREGATE_PREFIXES = ["Avg", "Max", "AvgC", "MaxC"]

_ALL_PREFIXES = config.BOOK_PREFIXES + config.CLOSING_PREFIXES
_ODDS_COL_RE = re.compile(
    r"^(?P<prefix>{})(?P<suffix>[HDA])$".format(
        "|".join(sorted(_ALL_PREFIXES, key=len, reverse=True))
    )
)


def download_season(
    start_year: int,
    league: str = config.DEFAULT_LEAGUE,
    session: requests.Session | None = None,
) -> str:
    url = config.BASE_URL.format(
        code=config.season_url_code(start_year), league=league
    )
    getter = session or requests
    resp = getter.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content.decode("latin-1")


def _raw_path(start_year: int, league: str):
    return config.RAW_DIR / f"{league}_{config.season_label(start_year)}.csv"


def fetch_all(force: bool = False) -> None:
    """Download every league's season CSVs into data/raw/ (cached unless force)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for league in config.LEAGUES:
        for year in config.all_seasons():
            path = _raw_path(year, league)
            if path.exists() and not force:
                continue
            print(f"downloading {league} {config.season_label(year)} ...", flush=True)
            path.write_text(download_season(year, league, session), encoding="utf-8")


def book_columns(columns) -> dict[str, list[str]]:
    """Map bookmaker prefix -> [H, D, A] column names present (all three)."""
    found: dict[str, dict[str, str]] = {}
    for col in columns:
        m = _ODDS_COL_RE.match(col)
        if m:
            found.setdefault(m.group("prefix"), {})[m.group("suffix")] = col
    return {
        prefix: [cols["H"], cols["D"], cols["A"]]
        for prefix, cols in found.items()
        if set(cols) == {"H", "D", "A"}
    }


def _repair_ragged_csv(raw_csv: str) -> str:
    """Trim/pad rows to the header width (old files have stray commas)."""
    rows = list(csv.reader(io.StringIO(raw_csv)))
    if not rows:
        return raw_csv
    width = len(rows[0])
    out = io.StringIO()
    writer = csv.writer(out)
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue
        row = row[:width] + [""] * max(0, width - len(row))
        writer.writerow(row)
    return out.getvalue()


def normalize_season(raw_csv: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_repair_ragged_csv(raw_csv)), skip_blank_lines=True)
    df = df.dropna(axis=1, how="all")
    df = df.rename(columns=BB_RENAMES)

    # Drop trailing junk rows without a real match.
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"]).reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    out["season"] = season
    for src_col, dst in CORE_COLUMNS.items():
        out[dst] = df[src_col]
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, format="mixed")
    for side in ("home_team", "away_team"):
        out[side] = out[side].str.strip().replace(TEAM_NAME_FIXES)
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    for src_col, dst in STAT_COLUMNS.items():
        if src_col not in df.columns:
            out[dst] = pd.NA if dst not in _NUMERIC_STATS else float("nan")
            continue
        if dst in _NUMERIC_STATS:
            out[dst] = pd.to_numeric(df[src_col], errors="coerce")
        else:
            out[dst] = df[src_col].astype("string").str.strip()

    books = book_columns(df.columns)
    for prefix, cols in books.items():
        for outcome, col in zip(config.OUTCOMES, cols):
            odds = pd.to_numeric(df[col], errors="coerce")
            # Decimal odds below 1.01 are data errors, not prices.
            out[f"{prefix}{outcome}"] = odds.where(odds >= 1.01)

    # Market aggregates: keep source Avg/Max when present, else compute
    # from the individual (non-closing) books in the file.
    individual = [
        p for p in books
        if p not in ("Avg", "Max") and p not in config.CLOSING_PREFIXES
    ]
    for outcome in config.OUTCOMES:
        book_odds = out[[f"{p}{outcome}" for p in individual]]
        for agg, fn in (("Avg", "mean"), ("Max", "max")):
            src_col = f"{agg}{outcome}"
            if src_col in df.columns:
                out[src_col] = pd.to_numeric(df[src_col], errors="coerce")
            else:
                out[src_col] = getattr(book_odds, fn)(axis=1)

    for col in df.columns:
        if _TOTALS_COL_RE.match(col):
            odds = pd.to_numeric(df[col], errors="coerce")
            out[col] = odds.where(odds >= 1.01)

    for col in df.columns:
        if _AH_COL_RE.match(col):
            vals = pd.to_numeric(df[col], errors="coerce")
            # Handicap lines pass through untouched; prices get the same
            # sub-1.01 data-error filter applied to every other odds column.
            out[col] = vals if col in _AH_LINE_COLS else vals.where(vals >= 1.01)

    out["n_books"] = out[[f"{p}H" for p in individual]].notna().sum(axis=1)
    return out.sort_values("date").reset_index(drop=True)


# Each league owns a fixed match_id range so ids never shift when another
# league is added to the registry later.
_LEAGUE_ID_SPAN = 1_000_000


def build_dataset() -> pd.DataFrame:
    frames = []
    for offset_idx, league in enumerate(sorted(config.LEAGUES)):
        seasons = []
        for year in config.all_seasons():
            season = config.season_label(year)
            raw = _raw_path(year, league).read_text(encoding="utf-8")
            seasons.append(normalize_season(raw, season))
        lf = pd.concat(seasons, ignore_index=True, sort=False)
        lf = lf.sort_values(["date", "home_team"]).reset_index(drop=True)
        lf.insert(0, "league", league)
        lf["match_id"] = offset_idx * _LEAGUE_ID_SPAN + lf.index
        frames.append(lf)
    return pd.concat(frames, ignore_index=True, sort=False)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation failures (empty list = healthy)."""
    problems = []
    if not df["result"].isin(config.OUTCOMES).all():
        problems.append("unexpected values in result column")

    counts = df.groupby(["league", "season"]).size()
    for league, spec in config.LEAGUES.items():
        expected = spec["recent_matches"]
        for season in config.REQUIRED_RECENT_SEASONS:
            n = counts.get((league, season), 0)
            if n < expected:
                problems.append(
                    f"required recent season {league} {season} incomplete: "
                    f"{n}/{expected} matches"
                )
    for (league, season), n in counts.items():
        cap = config.LEAGUES[league]["max_matches"]
        if n > cap:
            problems.append(f"{league} {season} has {n} rows (> expected {cap})")

    odds_cols = [c for c in df.columns if _ODDS_COL_RE.match(c)]
    for col in odds_cols:
        bad = df[col].dropna()
        if ((bad < 1.01) | (bad > 1001)).any():
            problems.append(f"implausible odds values in {col}")
    return problems


def main(force: bool = False) -> None:
    fetch_all(force=force)
    df = build_dataset()
    problems = validate(df)
    if problems:
        for p in problems:
            print(f"VALIDATION FAILURE: {p}", file=sys.stderr)
        sys.exit(1)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MATCHES_PARQUET, index=False)
    per_league = df.groupby("league").size()
    print(
        f"wrote {len(df):,} matches "
        f"({', '.join(f'{lg}: {n:,}' for lg, n in per_league.items())}) "
        f"to {config.MATCHES_PARQUET}"
    )


if __name__ == "__main__":
    main(force="--force" in sys.argv)
