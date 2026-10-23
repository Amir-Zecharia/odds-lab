# Data and study setup

Use Python 3.11 and run commands from the repository root with the virtual
environment activated. Install dependencies from `requirements.txt` first.
On macOS, a missing XGBoost OpenMP library can be resolved with
`brew install libomp`.

## Offline dashboard: clone or ZIP

```bash
python -m src.bundle verify
python -m src.launch
```

All 23 committed assets are already under `data/bundled/`: 61,061 matches
from six leagues, 2000–01 through 2025–26 (ending May 2026), enriched parquet,
study JSONs, and ten partial Club Elo cache inputs. Original retrieval dates
are unknown. See [DATA_SOURCES.md](../DATA_SOURCES.md) for attribution,
coverage and provider terms; no explicit redistribution permission is claimed.

The launcher verifies the immutable manifest and materializes missing files
into ignored `data/processed/` and `data/raw/clubelo/`, then opens Streamlit.
`streamlit run app/Home.py` also bootstraps through page setup. Installation
is resumable and never overwrites existing working files or ledger. No downloads
or recomputation are needed. Snapshot coverage is not a promise of current-season results.

For deliberate regeneration only, run `python -m src.ingest` then
`python -m src.run_analyses`. These contact the historical provider and may
require tens of minutes. Ingestion remains configured through 2025–26.
`--force` refreshes cached source CSVs, but does not advance configured seasons.

## Additional studies: optional regeneration

The bundled snapshot already includes these research results for the pages
and Research Log. No study command is required for first-run setup. Run these
individually only to regenerate results deliberately after refreshing working
data or changing a study. Some are costly; they are also unnecessary for the
synthetic test suite.

| Command | Study |
| --- | --- |
| `python -m src.research_prediction` | Alternative prediction models |
| `python -m src.drift_analysis` | Line drift and surprises |
| `python -m src.mismatch` | Mispricing discovery and confirmation |
| `python -m src.influence` | Bookmaker influence |
| `python -m src.seriea_lead` | Serie A anomaly replication |
| `python -m src.asian_handicap` | Asian handicap calibration |
| `python -m src.clubelo` | External Club Elo comparison; requires its API |
| `python -m src.pricing_engine` | Scoreline pricing engine |
| `python -m src.model_stack` | Rating-model stack |
| `python -m src.shot_strength` | Shots versus goals |

`python -m src.research_prediction` also regenerates the tracked file
`docs/research-prediction-findings.md`; review that diff after a rerun.

Historical findings in the README describe an earlier analysis run. A new
download, dependency version, or configuration can change the results;
inspect the generated outputs before quoting them as a replication.

## Optional live odds

Historical analysis and tests need no API key. Upcoming Matches accepts a
key in a password field, Streamlit secrets (`ODDS_API_KEY`), or the environment.
Select **Enable optional live odds** to fetch. All six leagues are supported;
responses cache for one hour, with cache age and a refresh control. There is
no API request at startup. For example:

```bash
export ODDS_API_KEY='your-key'
python -m src.launch
```

`.env.example` documents the variable; `.env` is not automatically loaded.
Do not paste real keys into commits, screenshots, or issue reports.

The ledger can record pre-kickoff predictions and later settle them:

```bash
python -m src.ledger record
python -m src.ingest  # only after deliberately updating configuration for a new season
python -m src.ledger settle
python -m src.ledger scoreboard
```

Current-season settlement requires a deliberate update of the configured season
range, refreshing result CSVs, and regenerating working analyses. The bundled
May 2026 snapshot cannot settle later results automatically. A real authenticated
API call remains unverified without an end-user key.

Keep a separate backup of `data/processed/ledger.parquet` if you use forward
testing: it is local evidence and cannot be reconstructed from historical
match data. Ingest reuses cached CSVs; use `--force` when refreshing results
that are not present in the cached season files.

## Automatic odds and final-score collection

Use the **Live Collection** page or `python -m src.collector status` to inspect
local collector state. No key is required to open the page or inspect status.
For the Mac scheduler, preview then install the LaunchAgent:

```bash
python -m src.collector schedule preview
python -m src.collector schedule install
```

Installation enables scheduled runs at 00:00, 08:00 and 16:00 in the Mac's local
timezone. Without a key, a run records
`needs_key` and makes no request. Future users configure their own key privately
using the Live Collection password field or `python -m src.collector configure
--set-key`. An environment override must be present in the collector process;
launchd does not inherit variables exported only in an interactive terminal.

See [COLLECTION.md](COLLECTION.md) for league selection, cost estimates, pause,
removal, backups, immutable predictions and validation gates. This separate
collector gets finals from the scores endpoint; the manual ledger's local-result
settlement limits above still apply to the manual ledger.

## Common first-run issues

- **No processed data:** run `python -m src.bundle verify`, then `python -m src.launch`.
  Check that the ZIP includes `data/bundled/` and that the bundle is intact.
- **Study results missing:** run the command shown on that page or in the
  study table above.
- **Source/API request fails:** check connectivity and retry. Club Elo and
  live odds are external services, separate from the historical CSV source.
- **Recent-season validation fails:** check that the configured required
  seasons are complete at the source; validation deliberately rejects
  incomplete required seasons.

## Optional Israel live coverage

**Ligat Ha'Al (IL1)** is an explicit opt-in on Upcoming Matches and Live
Collection. The six European defaults and historical research bundle are
unchanged; no Israeli historical snapshot is included.

Israel uses [direct API-Football v3](https://www.api-football.com/documentation-v3),
not RapidAPI. Supply its separate password field on Live Collection or set
`API_FOOTBALL_KEY`; the client sends `x-apisports-key` to the direct provider.
The private CLI options are `python -m src.collector configure
--set-api-football-key` and `--clear-api-football-key`. Blank fields preserve
stored keys; clearing either provider key leaves the other intact. Environment
values override private settings. `.env` files are not automatically loaded.
Select `IL1` explicitly with `python -m src.collector configure --leagues IL1`
(or include European codes for a combined collection).

The client discovers Israel's current top-flight season and requires its odds
coverage flag. Current authenticated coverage and plan access remain **unverified**
until an end user supplies a key. Coverage or plan failures are explicit and
retryable; no old-season fallback is used. The historical six-league validation
results do not establish Israel coverage or forecast quality.

Upcoming Matches makes requests only after the live checkbox is enabled with a
key, caches for one hour and offers an explicit refresh. Its Israel recording
action points to Live Collection: the historical Live Ledger cannot settle IL1.
The collector stores provider event identities, pre-kickoff frozen predictions
and regulation full-time finals. Only collected IL1 data trains the separate
Israel candidate, subject to its own chronological 500-training/200-validation
minimum and confidence-interval/Brier promotion gate. European promotion cannot
activate Israel learned forecasts. Until Israel passes, active forecasts use the
market baseline.

The Odds API estimate remains nine credits per European league/day at three
runs. API-Football has separate daily/per-minute quotas; discovery, fixtures and
paginated odds determine requests, so there is no fixed per-league cost promise.
Missing keys and quota/authentication failures are isolated by provider, allowing
healthy selected leagues to continue. Status and page startup make no HTTP calls.
