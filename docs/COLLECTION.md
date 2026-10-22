# Local live collection

The **Live Collection** dashboard page adds a persistent forward test alongside
the offline research and the older manual Live Ledger. It saves observed odds,
frozen pre-match forecasts and completed scores, compares forecasts with finals,
and fits later candidates using results available at that time. It does not
rewrite previous predictions or regenerate historical studies.

## Start without a key

From the repository root with `.venv` activated:

```bash
python -m src.collector status
python -m src.collector schedule preview
python -m src.collector schedule install
```

The Mac-only installer validates a plist, writes the user-owned
`~/Library/LaunchAgents/com.oddslab.collector.plist`, registers it with launchd
and enables collection settings. It refuses to replace an unrelated agent. The
plist invokes this checkout's `.venv/bin/python -m src.collector run`, so keep
the checkout and virtual environment at their installed paths. Reinstall after
moving them. Other platforms can inspect/run the portable collector manually;
the included scheduling installer is for macOS.

Times are **00:00, 08:00 and 16:00 in the Mac's local timezone**.
Stored timestamps are UTC. The dashboard
can remain closed. The logged-in user's Mac must be available: this is not a
hosted service, and it does not promise collection while the Mac is shut down,
asleep or disconnected. Missed runs are never represented as historical captures.

Without a key, an attempted run returns `needs_key` without HTTP. Installation
is therefore useful before activation; it does not mean real live collection
has succeeded. The page shows **Awaiting key** and local status. This project
has not verified a real authenticated provider call without an end-user key.

## Private activation when ready

Each person downloading the repository supplies their own The Odds API key.
Use the Live Collection password field and **Save private settings**, or:

```bash
python -m src.collector configure --set-key
```

The CLI prompts privately instead of accepting the secret in arguments. An
`ODDS_API_KEY` environment variable overrides the stored key. A variable exported
only in a terminal is not automatically available to launchd; the private settings
file is the straightforward option for scheduled runs. The collector does not
read Streamlit secrets, `.env` files or the Upcoming Matches temporary input.
Saving a blank password preserves an existing key. **Clear stored key** or
`python -m src.collector configure --clear-key` removes it; an environment
override can still apply. Do not share keys in issues, screenshots or commits.

Private settings are in `data/live/settings.json` with mode 0600. SQLite data,
settings and lock/WAL files are ignored under `data/live/`. The key is excluded
from the database, error details, generated plist and command arguments.

## Coverage, cost and controls

Default coverage is all six supported leagues. Select fewer on the page, or use
league codes `E0 E1 SP1 D1 I1 F1` with the CLI:

```bash
python -m src.collector configure --leagues E0
python -m src.collector status
python -m src.collector run
```

Manual `run` attempts the current scheduled slot even if scheduled collection
is disabled. Completed slots return `already_completed`; partial runs retry
unfinished steps without duplicating successful ones. The page has the same
**Run current scheduled slot now** action. A concurrent run returns `busy`.
Page startup and status inspection do not make API requests.

At three daily runs, estimated cost is nine credits per league/day: one odds
credit plus two scores credits per league/run. All six leagues are approximately
**54 credits/day or 1,620 per 30 days**, above the 500-credit free allowance.
Even one league is approximately 270 per 30 days. Empty responses may cost less;
actual provider headers and plan terms govern usage. The page displays estimates
and recorded quota metadata. Authentication/quota failures stop affected requests
and remain retryable; no subscription is purchased automatically.

To pause an installed scheduler, uncheck **Enable scheduled collection** and
save, or run:

```bash
python -m src.collector configure --enabled no
```

To resume it use `--enabled yes`. This setting controls an already installed
agent; it does not install one. Remove this project's agent with:

```bash
python -m src.collector schedule uninstall
```

Uninstalling disables scheduling and preserves local evidence. To inspect native
registration on macOS, run `launchctl print gui/$(id -u)/com.oddslab.collector`.

## What is stored and compared

Every run fetches scores first with `daysFrom=3`, fits a candidate from eligible
past data, then fetches odds. League plus provider event ID links new snapshots
to finals. Actual capture time is measured after the response; a forecast can be
saved only before kickoff. Baseline, candidate and active forecast records retain
probabilities, model version, feature inputs and training cutoff. Later odds
create new snapshots; old forecasts remain unchanged.

Only completed results with valid nonnegative integer scores are accepted.
Final corrections retain observation history. Current truth prefers the newest known
provider update among observations available at the evaluation cutoff; stale
responses remain in history. Equal provider timestamps use observation time and
insertion order. Missing provider timestamps rank below known versions, with
observation time as the fallback when all versions lack an update time. Finals
arriving later score saved forecasts using log loss and Brier loss (lower is
better). The page provides full snapshot/forecast/result histories and summary
losses using one last eligible pre-kickoff snapshot per event and forecast kind.
Captures after a rescheduled match's final actual kickoff cannot be scored.

The scores endpoint's three-day lookback means an outage longer than that can
leave a match unresolved. Missing outcomes are never invented. Coverage gaps,
provider errors, delays and inaccuracies remain possible; local validation is
not an independent guarantee of source truth. Runs preserve successful league
steps while reporting failures elsewhere.

## Learning safeguards

The learner uses historical de-margined pre-close Pinnacle probabilities, with
consensus fallback, league/team features and collected pre-match odds paired
with observed finals. It takes one eligible row per match; repeated snapshots do
not multiply that match's training weight. Labels observed after training time
are excluded. Historical bundle data remains unchanged.

A regularized multinomial candidate is tested on a chronological holdout with
no date shared across the split. Promotion requires at least 500 training and
200 validation matches, all three outcome classes, lower log loss with the
paired 95% confidence interval's upper bound below zero, and no worse mean Brier
loss. Otherwise active forecasts retain the market baseline and show the reason.
The candidate is refitted on eligible past data for later fixtures and stored as
safe JSON parameters and lineage, without executable pickle. Improvement is not
guaranteed. A match's final result can inform future fits; it cannot change its
own frozen pre-match prediction or make that evaluation look better retroactively.

## Backup and separation from the manual ledger

The authoritative collector evidence is `data/live/collector.sqlite3`; back it
up using SQLite's backup facility or while the collector is stopped (do not copy
only the main file during active WAL writes). Keep the private settings/key out
of shared backups. Deleting the database discards evidence that historical data
cannot reconstruct.

The earlier `src.ledger` and Upcoming Matches record/settle workflow remains
separate in `data/processed/ledger.parquet`. That manual ledger settles from
local historical results and retains its documented season-refresh requirements.
The new collector settles its own event-ID records from live scores. It does not
silently merge or replace the older ledger, update the immutable bundle, or
advance historical ingestion's configured seasons.
