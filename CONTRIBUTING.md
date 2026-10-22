# Contributing to Odds Lab

Thanks for your interest! This project is deliberately extensible — most
contributions are one of three shapes.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.bundle verify
python -m pytest -q
python scripts/smoke_dashboard.py  # all 29 defaults + controls, isolated offline ZIP copy
python -m src.launch
```

Use Python 3.11 (the CI version). For data setup, optional studies, and
macOS troubleshooting, see [docs/SETUP.md](docs/SETUP.md).

## The three contribution shapes

1. **A new league** — one entry in `config.LEAGUES` (football-data.co.uk
   division code, display name, `max_matches` historical upper bound,
   `recent_matches` expected current size). Ingest, validation,
   features, the league selector and the pooled All-leagues view pick it
   up automatically. Check the source CSVs exist for the code first. Optional
   live odds also need a verified sport key in `src.live.SPORT_KEYS`.
2. **A new research study** — a module in `src/` that writes its numbers
   to `data/processed/<study>.json`, plus **either** a dedicated dashboard
   page **or** an entry in the `STUDIES` registry in
   `app/pages/17_Research_Log.py`. House rule: every study ships visible
   on the dashboard, and negative results are stated plainly. JSON + a
   Research Log entry is the standard; a long-form write-up under `docs/`
   (see `docs/research-prediction-findings.md`) is optional extra.
3. **A new market** (Asian handicap, correct score, ...) — extend the
   ingest column regexes, preserve the new columns in the enriched parquet,
   and follow the totals market (`src/totals.py`) as the template.

## Standards

- Strictly time-ordered evaluation everywhere; no future leakage. The
  test suite has leakage guards — add one for anything new that fits.
- Multiple-comparison honesty: anything that scans segments reports
  FDR-controlled significance (see `src/mismatch.py`).
- `python -m pytest -q` must pass; tests run on synthetic data so CI
  needs no data download.
- Match the existing code style: type hints, docstrings that explain the
  *why*, minimal comments.

## Not the goal

This is a research platform, not a betting product. PRs adding "winning
strategies", tipster feeds, or bookmaker affiliate hooks won't be merged.

The committed snapshot includes third-party data with separate terms; see
[DATA_SOURCES.md](DATA_SOURCES.md). Rebuilding studies is optional development
work, not first-run setup. Preserve local ledgers and never commit credentials.
