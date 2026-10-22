# Architecture

Odds Lab is a local research application. Streamlit presents results;
Python modules perform ingestion, feature generation, statistical analysis,
and forecasting. There is no deployed backend or database service.

## Data flow

1. `src.launch` calls `src.bundle.bootstrap` to verify manifest paths, hashes,
   schemas and required inventory, then atomically install missing immutable
   snapshot files into ignored working paths. Existing files and ledger survive.
   Direct Streamlit entry calls bootstrap in `common.page_setup` too.
2. `src/config.py` defines project-relative paths, leagues, seasons, and
   bookmaker column families.
3. `src/ingest.py` downloads and normalizes CSVs, repairs source quirks,
   validates coverage, and writes `data/processed/matches.parquet`.
4. `src/probabilities.py` removes bookmaker margins. `src/features.py`
   computes historical team features within each league.
5. `src/run_analyses.py` builds `enriched.parquet` and precomputes the
   per-league and pooled comparison results in `analysis.json`.
6. Dedicated study modules write their own JSON results. See
   [SETUP.md](SETUP.md) for the commands and order.
7. `app/common.py` caches files by modification time and supplies shared
   filtering helpers. `app/Home.py` and 29 pages display research and live-collection state.

## Module map

| Area | Main modules |
| --- | --- |
| Calibration and evaluation | `calibration`, `evaluate`, `models` |
| Market microstructure | `consensus`, `influence`, `drift_analysis`, `mismatch` |
| Goal and handicap markets | `totals`, `asian_handicap`, `pricing_engine` |
| Team strength and forecasts | `ratings_zoo`, `model_stack`, `forecast`, `season_sim`, `clubelo` |
| Match statistics | `halftime`, `match_stats`, `shot_strength` |
| Manual live odds and forward testing | `live`, `ledger` |
| Scheduled collection, persistence and learning | `collector`, `collection_schedule`, `live_store`, `live_learning` |

Tests use synthetic data and cover normalization, probability calculations,
time ordering, evaluation, forecasting, and ledger behavior. They run
without downloading the historical dataset or calling the live services.
The portable `scripts/smoke_dashboard.py` creates an isolated code/bundle copy,
bootstraps fresh working data, blocks HTTP requests and exercises all 30 page
defaults and important controls.

## Boundaries

- Historical season coverage is fixed in configuration, not automatically
  advanced each calendar year.
- Results in documentation are research snapshots; generating current
  results is separate from passing the tests.
- Keep temporal splits and leakage guards when adding models. Segment
  searches need multiple-testing correction and independent confirmation.
- Adding a study should include a dashboard page or Research Log entry,
  including when the study finds no improvement.
- `data/bundled/` contains 23 immutable, attributed third-party assets;
  all runtime data, credentials, live responses and the ledger remain local.
  Ten Club Elo cache files are partial inputs, not complete external coverage.
- Upcoming Matches resolves per-user credentials from environment, Streamlit
  secrets or password input and requires explicit enablement. Requests cache
  for one hour. Ledger recording is scoped to the selected league.
- Forecast and simulator caches include the enriched dataset modification token.
  Historical forecasts and simulations are retrospective.
- The legacy manual ledger settles against local results; configuration ends at 2025–26. Later
  seasons require deliberate configuration, ingestion and analysis refresh.
- Provider attribution and unresolved redistribution terms are documented in
  `DATA_SOURCES.md`; code licensing does not license third-party data.

## Persistent collector

`launchd → src.collector → src.live → src.live_store → src.live_learning`

The Mac LaunchAgent invokes this checkout's virtual-environment Python at
00:00/08:00/16:00 local time. UTC capture and observation times are measured
after requests. A nonblocking process lock prevents concurrent collection;
slot and per-league step uniqueness make completed slots idempotent and partial
slots retryable. Authentication/quota failures stop further affected requests;
successful steps remain saved.

The collector fetches completed scores first, fits a candidate from eligible
past data, then captures odds and freezes baseline/candidate/active predictions
before kickoff. Provider league plus event ID links new odds and finals exactly.
SQLite keeps raw odds, snapshot and prediction versions, final observations and
corrections, model JSON, sanitized run status and quota metadata. Rescheduled
matches exclude captures after the final actual kickoff from evaluation.

`data/live/collector.sqlite3`, its lock/WAL files and private `settings.json`
are ignored. Keys remain in private settings (mode 0600) or an environment
override; they are excluded from database records, plist and process arguments.
The dashboard reads local state on startup and makes requests only after an
explicit manual run. The scheduler works independently of Streamlit.

Learning uses historical pre-close Pinnacle probabilities (consensus fallback),
league and canonical team features, plus one last eligible snapshot per labeled
live match. Labels must have been observed by training time. Chronological
holdout dates are disjoint; at least 500 training and 200 validation examples
and all classes are required. Promotion requires lower log loss, paired 95% CI
upper bound below zero and no worse mean Brier loss. Safe JSON stores model
parameters and lineage. Failed gates leave the market baseline active. Later
fits never rewrite saved predictions or the historical bundle/studies.

The 30-page offline smoke includes the empty/awaiting-key collector page. Fake
provider tests cover persistence through final settlement and later prediction,
retry, validation, leakage, quota/secret handling and plist scheduling. They do
not establish authenticated provider availability. See [COLLECTION.md](COLLECTION.md)
for operational limits and command reference.
