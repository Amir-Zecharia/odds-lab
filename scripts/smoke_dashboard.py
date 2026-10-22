"""Offline distribution smoke: isolated bundle bootstrap and real Streamlit pages.

Run from any directory with the project's installed Python dependencies.
Default: 30 pages and controls. --quick checks Home and core controls only.
Each page uses a fresh subprocess with a 120-second limit; no API key needed.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def child(root, page, controls):
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "app")]
    os.environ.pop("ODDS_API_KEY", None)
    import requests

    calls = []

    def blocked(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Network request attempted during offline smoke")

    requests.sessions.Session.request = blocked
    import streamlit as st
    st.secrets = {}  # Ignore user-global secrets in this isolated offline check.
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(root / page), default_timeout=110).run()

    def check():
        if at.exception:
            raise AssertionError("; ".join(str(e.message) for e in at.exception))
        if calls:
            raise AssertionError("Offline page attempted network access")

    check()
    if controls and page.endswith("1_Data_Explorer.py"):
        league = next(w for w in at.sidebar.radio if w.label == "League")
        # AppTest options are display names; indices select actual registry values.
        for index in range(len(league.options)):
            next(w for w in at.sidebar.radio if w.label == "League").set_value(
                __import__("src.config", fromlist=["league_options"]).league_options()[index]
            ).run()
            check()
        next(w for w in at.sidebar.radio if w.label == "Season range").set_value("Last 3 seasons").run()
        check()
        next(w for w in at.sidebar.radio if w.label == "Season range").set_value("Custom").run()
        check()
        slider = next(w for w in at.sidebar.select_slider if w.label == "Seasons")
        slider.set_value((slider.options[-2], slider.options[-1])).run()
        check()
    if controls and page.endswith("15_Upcoming_Matches.py"):
        from src.live import SPORT_KEYS
        for league in SPORT_KEYS:
            at.sidebar.radio[0].set_value(league).run()
            check()
        assert at.text_input and at.text_input[0].value == "", "Expected empty per-user key input"
    if controls and page.endswith(("26_Season_Simulator.py", "28_Match_Forecast.py")):
        from src import config
        # Exercise a second league and an earlier season where these controls exist.
        next(w for w in at.sidebar.radio if w.label == "League").set_value(list(config.LEAGUES)[1]).run()
        check()
        season = next((w for w in at.selectbox if w.label == "Season"), None)
        if season and len(season.options) > 1:
            season.set_value(season.options[1]).run()
            check()
    print(f"PASS {page}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--child", nargs=2, metavar=("ROOT", "PAGE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(Path(args.child[0]), args.child[1], True)
        return 0
    source = Path(__file__).resolve().parents[1]
    pages = ["app/Home.py"] + [str(p.relative_to(source)) for p in sorted((source / "app/pages").glob("*.py"))]
    if len(pages) != 30:
        raise RuntimeError(f"Expected 30 actual pages, found {len(pages)}")
    if args.quick:
        pages = [p for p in pages if p.endswith(("Home.py", "1_Data_Explorer.py", "15_Upcoming_Matches.py", "29_Live_Collection.py"))]
    failures = []
    with tempfile.TemporaryDirectory(prefix="odds-lab-zip-smoke-") as temporary:
        root = Path(temporary)
        # Only source code and immutable assets: no local processed/raw data,
        # ledger, .env, .streamlit secrets, or working caches are copied.
        for name in ("src", "app"):
            shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(source / "data/bundled", root / "data/bundled")
        env = dict(os.environ, ODDS_API_KEY="", PYTHONPATH=str(root))
        bootstrap_code = (
            "import requests; "
            "requests.sessions.Session.request = lambda *a, **k: "
            "(_ for _ in ()).throw(AssertionError('Bootstrap attempted network')); "
            "from src.bundle import bootstrap; bootstrap()"
        )
        bootstrap = subprocess.run([sys.executable, "-c", bootstrap_code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
        if bootstrap.returncode:
            print("FAIL fresh bootstrap:", (bootstrap.stderr or bootstrap.stdout)[-2000:])
            return 1
        if not (root / "data/processed/enriched.parquet").is_file():
            raise AssertionError("Fresh bootstrap did not install enriched data")
        for page in pages:
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(root), page], cwd=root, env=env, capture_output=True, text=True, timeout=120)
                if result.returncode:
                    failures.append(page)
                    print(f"FAIL {page}:\n{(result.stderr or result.stdout)[-2500:]}", flush=True)
                else:
                    print(f"PASS {page}", flush=True)
            except subprocess.TimeoutExpired:
                failures.append(page)
                print(f"FAIL {page}: exceeded 120 seconds", flush=True)
    print(f"Offline ZIP smoke: {len(pages) - len(failures)}/{len(pages)} pages passed (including controls).")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
