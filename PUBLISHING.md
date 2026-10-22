# Publishing to GitHub

## Local preflight

From the repository root, activate the virtual environment and run:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m compileall -q src app tests
python -m pytest -q
git diff --check
git status --short
```

Review the files you intend to commit. Runtime data, generated working results, `.env`
files, Streamlit secrets, and private keys are ignored. `.env.example`
contains no credential. Ignore rules do not remove previously tracked
files or secrets from history.

The bundle is intentionally tracked; runtime data, ledger, credentials and live
responses must remain local. Before sharing it, review [DATA_SOURCES.md](DATA_SOURCES.md):
published provider terms do not establish explicit redistribution permission.
Do not describe external data as MIT-licensed. Original retrieval dates are unknown.

Validate the distribution itself:

```bash
python -m src.bundle verify
python scripts/smoke_dashboard.py
```

The smoke script bootstraps an isolated ZIP-like copy from bundled assets,
blocks HTTP at the requests boundary, and checks all 30 page defaults plus
important controls without credentials. This does not verify authenticated
provider access or guarantee factual accuracy of upstream data.

The private collector directory `data/live/` must stay ignored, including
SQLite/WAL/lock files and `settings.json`. Each clone installs its own scheduler
and provides its own key; a LaunchAgent is not installed by cloning, CI or the
dashboard startup. CI checks offline behavior without credentials or paid calls.
Review tracked files and reachable Git history for credentials before any push.
The optional authenticated API path remains unverified until a user configures
and tests their own key. This does not block offline use.

## Publish when ready

First review and commit the preparation changes. Set your desired author
identity before committing if you want attribution to your GitHub account;
use the exact private commit email shown in your GitHub email settings.

```bash
git diff
git add README.md CONTRIBUTING.md PUBLISHING.md .gitignore .env.example .github/workflows/ci.yml docs/SETUP.md docs/ARCHITECTURE.md docs/COLLECTION.md DATA_SOURCES.md requirements.txt src app tests scripts data/bundled
git diff --cached
git commit -m "Prepare repository for GitHub publication"
```

With the GitHub CLI installed and authenticated, the following command
creates a **public** repository and publishes all history reachable from
the current branch. Choose `--private` instead if that is your intent.
Run it only when ready to publish:

```bash
gh auth status
gh repo create odds-lab --public --source=. --push
```

Alternatively, create an empty repository on GitHub, then use its SSH URL:

```bash
git remote add origin git@github.com:YOUR-USERNAME/odds-lab.git
git push -u origin HEAD
```

If a remote already exists, inspect `git remote -v` before changing it.
No branch rename or history rewrite is needed.

## After publishing

- Check the `tests` workflow on the repository's Actions tab.
- Replace the README's relative Actions link with a workflow badge using
  the actual owner and repository URL, if desired.
- Add a description and topics such as `football`, `sports-analytics`,
  `market-efficiency`, `streamlit`, and `python`.
- Follow [docs/SETUP.md](docs/SETUP.md) for offline bundled startup and optional regeneration.
  GitHub publishing does not host the Streamlit application.
