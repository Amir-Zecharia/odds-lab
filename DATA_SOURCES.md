# Data sources and historical bundle

The offline snapshot contains 61,061 completed matches from 2000-07-28 through
2026-05-24 (seasons 2000–01 through 2025–26). It is historical research data;
it does not establish coverage of the current season or upcoming fixtures.

| Division | Matches |
| --- | ---: |
| England Premier League (E0) | 9,880 |
| England Championship (E1) | 14,352 |
| Spain La Liga (SP1) | 9,880 |
| Germany Bundesliga (D1) | 7,956 |
| Italy Serie A (I1) | 9,584 |
| France Ligue 1 (F1) | 9,409 |

Results, match statistics and bookmaker prices originate from
[Football-Data.co.uk](https://www.football-data.co.uk/), using its historical
`mmz4281/{season}/{division}.csv` feeds. The local ingestion and research
pipeline produced `matches.parquet`, `enriched.parquet`, and the study JSONs.
The committed bundle preserves their existing bytes; it does not recompute the
research or download newer seasons at startup.

Cached external Elo inputs originate from [Club Elo](http://clubelo.com/)
([API](http://api.clubelo.com/)). The bundle contains the ten CSV cache files
available in this checkout. Coverage is partial; the header-only
`Ajaccio_GFCO.csv` preserves an unavailable-history response, not a rating.
The existing `processed/clubelo.json` records study coverage and network status.
Missing Club Elo histories are not fetched by bootstrap.

Original retrieval timestamps are unknown. `original_retrieved_at: null` in the
manifest records that fact. `assembled_at` is the bundle assembly time, not a
provider retrieval or freshness timestamp. Study `computed_at` fields describe
local computation only.

## Integrity and limits

`data/bundled/manifest.json` records each file's SHA-256, byte size and schema,
source attribution, match coverage and league counts, and validation checks.
`python -m src.bundle verify` checks bytes, safe allowlisted paths, schemas,
unique fixtures, score/result consistency, consensus probability bounds and
normalization, metadata, and alignment of enriched and base matches. These
checks establish integrity and internal consistency; they do not independently
verify the factual accuracy of provider records or research conclusions.

`python -m src.launch` verifies the bundle and atomically installs missing
working files under ignored `data/processed/` and `data/raw/clubelo/` before
launching Streamlit. Direct Streamlit pages also bootstrap on page setup.
Existing working files and personal ledgers are never automatically replaced.
Interrupted installation can be rerun. The bundle excludes credentials,
ledgers, live odds responses and logs. A maintainer can create a new bundle
explicitly with `python -m src.bundle build --bundle-dir NEW_DIRECTORY`;
this refuses to overwrite an existing bundle directory.

## Rights and attribution

The repository's MIT license covers its original code, not third-party data.
Football-Data.co.uk's published personal/research terms do **not** establish an
explicit redistribution license. No such license is asserted for the included
snapshot or Club Elo inputs. Attribution and local validation do not grant
redistribution rights. Consult the providers' current terms and obtain any
necessary permission before redistributing data or using it commercially.
This bundle records provenance; it is not a representation that redistribution
permission has been obtained.

Live odds are separate optional per-user API data and are not included in the
snapshot. Each end user supplies their own credentials.

## Optional per-user live collection

[The Odds API v4](https://the-odds-api.com/liveapi/guides/v4/) supplies current
1X2 odds and completed scores. Each user supplies a private API key; these live
responses are stored only in ignored local `data/live/collector.sqlite3` and are
excluded from the committed historical snapshot. The collector joins by provider
sport/league and event ID, records actual UTC observation times, and preserves
final-score corrections. Local validation rejects unfinished/malformed finals;
it does not independently verify provider correctness or promise uptime.

Scores request `daysFrom=3`; missing results older than that window remain
unresolved after long outages. Estimated all-six-league usage is 54 credits/day
or 1,620 per 30 days at three runs daily (one odds credit plus two score credits
per league/run), compared with the 500-credit free allowance. Empty responses
may cost less. Provider plans, terms and actual quota headers govern usage;
no plan is purchased automatically. See [collection setup](docs/COLLECTION.md).
Authenticated live access has not been verified without an end-user key.

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
