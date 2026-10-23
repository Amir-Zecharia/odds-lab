# ⚽ Odds Lab — the betting market under a microscope

[Tests run in GitHub Actions](../../actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/code_license-MIT-green)
![data](https://img.shields.io/badge/matches-61%2C061-orange)

An open-source, locally-run research platform asking one question with
proper statistics: **how well do bookmaker odds predict football — and
where, if anywhere, do they crack?**

Six leagues · 26 seasons · 61,061 matches · 16 bookmakers · 30 interactive
dashboard pages · 12 studies · every result reproducible from free public data.

> **This is not a betting tool.** Its central, repeatedly-confirmed finding
> is that the market is brutally hard to beat — which is exactly what makes
> it a beautiful statistics playground.

## Quick start

```bash
# Clone this repository from its GitHub page, then:
cd odds-lab
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.launch         # verify/install bundled snapshot and open dashboard offline
```

Python 3.11 is tested in CI. Run commands from the repository root.
On macOS, install `libomp` with `brew install libomp` if XGBoost reports
that its OpenMP library is missing.

Tests need no downloaded data or API key:

```bash
python -m pytest -q
```

A clone or extracted ZIP includes all 23 snapshot assets: 61,061 matches
across six leagues, 2000–01 through 2025–26, ending in May 2026, enriched
matches, study results and ten partial Club Elo cache files. No historical
downloads or model recomputation are required. Original retrieval dates are
unknown. Bootstrap installs missing working files without overwriting existing
data or your ledger. Verify the immutable snapshot with `python -m src.bundle verify`.
See [setup](docs/SETUP.md) and [source attribution](DATA_SOURCES.md).

Live odds are optional for the six research leagues and opt-in Israel. Supply your own `ODDS_API_KEY`
through the environment, Streamlit secrets, or the page's password input,
then select **Enable optional live odds**. Startup makes no live request.
Responses cache for one hour; the page shows cache age and a refresh button.
`.env` files are not loaded automatically.

## What it found

| Finding | Evidence |
| --- | --- |
| **The market is calibrated.** When odds say 60%, it happens ~60% of the time | slope 1.02 ± 0.02, 61k matches, 6 leagues |
| **Nothing beats the closing odds** — Elo, form, ML, recalibration, cross-market blends all fail out-of-sample | expanding-window log loss, paired bootstrap CIs |
| **Pinnacle closing is the sharpest single price**; optimal bookmaker weights beat a plain average by 0.0001 | convex weight optimization over the simplex |
| **The early→close line move is real information** (drift z = 4.7) — and fully absorbed by kickoff: adding it as a feature makes predictions *worse* | Line Movement + Research Log |
| **A deliberate mispricing dredge found nothing durable**: 7 FDR-significant pockets in 2000-2017, zero replicated on 2018-2026 | Mismatch Explorer (BH-FDR + era-split confirmation) |
| **Bookmakers copy each other, hierarchically**: Pinnacle prices independently (herding 0.10); Bet365/William Hill erase ~70% of their disagreement with the pack by kickoff; leadership is one-way (z ≈ +24) | Bookmaker Influence |
| **Every mechanical betting rule loses ≈ the margin** (ROI ≈ −3% over 26 years) | Betting Simulator |
| **Asian handicap — the cleanest test (no draw) — is calibrated too**: slope 1.019 ± 0.156 over 15,239 settled bets | Asian Handicap |
| **A stack of five rating systems beats each of its members but still trails the market by +0.0165**; let the market join and it ties exactly while taking 72.5% of the learned weight | Model Stack |
| **A desk-style pricing engine trails the market least in goals markets** (+2.2% totals, +3.1% 1X2, +4.3% Asian handicap) — where public data reaches furthest | Pricing Engine |
| **The one anomaly found (Serie A, slope 1.10) faded**: strong pre-2018, gone since — an inefficiency that closed | Research Log |
| **Forward testing is available**: predictions recorded before kickoff, frozen at kickoff, scored after — the one result hindsight cannot touch | Live Ledger |
| **Away half-time leads are twice as fragile as home ones** (13.9% vs 6.8% overturned) — home advantage is partly the ability to recover | Half Time |
| **Corner and card rates drift by era**, so pooled-history models are miscalibrated; even corrected, team tendencies barely beat a flat league average | Corners, Cards & Referees |
| **Referees have real personal tendencies** (0.64×–1.24×) — but only visible after removing league *and* era confounds | Corners, Cards & Referees |
| **Shots on target are a *worse* strength signal than goals** in all six leagues (+0.017 to +0.030 nats) — shot *quality*, not shot count, is what xG contributes | Research Log |

## The dashboard

30 pages (Home plus 29 dashboard pages), with interactive analyses,
league/season/segment filters and plain-language explainers: data exploration,
calibration, bookmaker leaderboard & deep-dives, market-beating attempts,
odds-gap and line-movement microstructure, the totals and Asian-handicap
markets, a betting simulator, live upcoming-match probabilities (The Odds
API), a mismatch scanner with false-discovery-rate control, bookmaker
influence networks, a desk-style pricing engine (one scoreline grid prices
every market), a stacking meta-model over five rating systems, and a
Research Log where every study — including the negative results — is
recorded. The Live Collection page adds persistent odds snapshots, final-score
comparisons and guarded learning for future fixtures.

## What twelve studies add up to

Read together, the results tell one coherent story about where football
information lives:

1. **The market is the best public forecast of a match outcome, and
   nothing closes the gap.** Not Elo, form, or rest days; not XGBoost; not
   recalibration or cross-market signals; not a stacking meta-model over
   five rating systems (which beats every one of its members and still
   trails by +0.0165); not an external results-only rating. The gap is
   *information*, not machinery.
2. **That efficiency is manufactured, and you can watch it happen.**
   Pinnacle prices independently while recreational books erase ~70% of
   their disagreement with the pack by kickoff, in a strictly one-way
   hierarchy. Prices start imperfect and converge: the early line is
   measurably worse than the close.
3. **It is near-perfect, not perfect.** Pooled across 61k matches the
   calibration slope is 1.038 ± 0.010 — a real favourite–longshot tilt,
   detectable but far smaller than the 2–7% margin, so measurable and
   unexploitable at once. One genuine anomaly (Serie A pre-2018) existed
   and has since closed.
4. **Away from the money, prediction gets harder, not easier.** Corners
   and cards have no closing line to defer to — and turn out to be close
   to noise, barely beating a league average. Where structure does exist
   it is often somewhere unglamorous: referee tendencies, the second-half
   goal surge, the fragility of away leads.
5. **Every negative result here was earned.** The dredge that found
   nothing used FDR control and era-split confirmation. The Serie A lead
   was killed by its own replication test. The shots-as-xG idea died in
   all six leagues. Publishing those is the point.

## Forward testing

Every study above is retrospective. They are done carefully — strictly
time-ordered splits, paired bootstrap CIs, FDR control — but a backtest can
always be quietly shaped by decisions made after seeing the data. The
**Live Ledger** removes that objection: predictions are written to disk
before kickoff, **frozen** the moment a match starts (enforced in code and
covered by a tamper test), and scored once results land.

```bash
ODDS_API_KEY=<key> python -m src.ledger record   # before matchday
python -m src.ledger settle                   # after deliberately refreshing results
python -m src.ledger scoreboard                  # where it stands
```

The Upcoming Matches page also records predictions for the selected league.
This legacy ledger settles against local historical results; it is separate
from Live Collection, which joins provider odds and scores by event ID. Ingestion is configured only through 2025–26;
current-season settlement requires a deliberate season configuration update,
result ingestion refresh (including stale cached CSVs), and analysis refresh.
It does not happen automatically. Keep a backup of your local ledger.
A real authenticated API call has not been verified without an end-user key.

## Automatic local collection

The separate **Live Collection** page stores live odds, final observations and
frozen forecasts in local SQLite. A macOS LaunchAgent can run at **00:00,
08:00 and 16:00 in the Mac's local timezone**, including with the dashboard
closed. It remains awaiting a private key until you configure one. Setup,
commands, learning safeguards and operating limits are in
[Live collection](docs/COLLECTION.md).

```bash
python -m src.collector status
python -m src.collector schedule preview
python -m src.collector schedule install  # macOS; enables this user's schedule
```

Each user supplies their own The Odds API key privately through the page or
`python -m src.collector configure --set-key`. These scheduled runs do not use
Streamlit secrets or the Upcoming Matches page's temporary password input.
All six leagues at three runs daily are estimated at 54 credits/day or 1,620
per 30 days, above the 500-credit free allowance; select fewer leagues as needed.
Scores cover the last three days, so long interruptions can leave unresolved
matches. Forecasts remain frozen; new results train later candidates, promoted
only after chronological validation against the market baseline. Improvement
is not guaranteed, and historical studies are not automatically recomputed.

## Methodology (one paragraph)

Implied probabilities (`1/odds`) are corrected for the bookmaker margin
with the **Shin method** (margins hide disproportionately in longshots).
A margin-weighted cross-book **consensus** is the baseline every model
must beat; **Pinnacle's closing line** is the harder second baseline.
Evaluation is **log loss** under strictly **time-ordered** expanding-window
splits with **paired bootstrap** confidence intervals; segment scans use
**Benjamini-Hochberg FDR** plus discover-then-confirm era splits. Negative
results are published as findings.

## Architecture

```
odds-lab/
├── data/            # committed bundled snapshot + ignored working data
├── src/             # ingest, probabilities, features, models, research modules
├── app/             # Streamlit dashboard (Home + pages/)
├── tests/           # synthetic-data tests — no download needed
└── .github/         # CI: pytest on every push
```

- **Adding a league** = one entry in `config.LEAGUES` (the registry drives
  ingest, validation, features, the UI selector and the pooled All-leagues
  view). Optional live odds also need a verified provider sport key in
  `src.live.SPORT_KEYS`.
- **Adding a study** = a module writing `data/processed/<study>.json` plus
  a dashboard page or a `STUDIES` entry in the Research Log. House rule:
  no study is finished until it is visible on the dashboard.
- **Storage**: parquet — `matches.parquet` (clean source rows) and
  `enriched.parquet` (+ de-margined probabilities, features, market
  columns), rebuilt by `src.ingest` and `src.run_analyses`.
- **Data**: [football-data.co.uk](https://www.football-data.co.uk/) (attributed bundled third-party CSV-derived snapshot). Premier League, Championship, La Liga, Bundesliga,
  Serie A, Ligue 1; 2000-01 → 2025-26.

## Standing on the shoulders of

- [penaltyblog](https://github.com/martineastwood/penaltyblog) — the
  reference Python library for football modelling (Shin, Dixon-Coles).
- [sports-betting](https://github.com/georgedouzas/sports-betting) — value-bet
  framing and backtesting ideas.
- [football_analytics](https://github.com/eddwebster/football_analytics) — a
  curated map of the football-analytics world.
- The market-efficiency literature: Shin (1992), Dixon & Coles (1997),
  Hvattum & Arntzen (2010), and the closing-line-value tradition.

For a module map, see [Architecture](docs/ARCHITECTURE.md). To publish a
copy of the repository, see [Publishing](PUBLISHING.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — new leagues, new research studies
(house rule: every study ships with a dashboard page, negative results
stated plainly), new markets.

## License & disclaimer

Project code is MIT-licensed. External data has separate provider terms;
no explicit redistribution permission is established by their published
personal/research terms. See [DATA_SOURCES.md](DATA_SOURCES.md). For education and research; not betting advice. The core result —
odds are near-efficient — implies profitable betting on public information
is unlikely. Gambling regulations vary by jurisdiction.

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
