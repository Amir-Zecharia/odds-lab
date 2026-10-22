import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from common import page_setup
from src import config

page_setup("Research Log", icon="📚")
st.caption(
    "Every research study run on this platform, with its verdict and "
    "artifacts. New studies are appended here as they complete."
)

DOCS = config.PROJECT_ROOT / "docs"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def prediction_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "research_prediction.json")
    if results is None:
        st.caption("Run `python -m src.research_prediction` to generate.")
        return
    base = results["baselines"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test matches", f"{results['universe']['n_test_matches']:,}")
    c2.metric("Pinnacle closing", f"{base['Pinnacle closing']:.4f}",
              help="Out-of-sample log loss; the bar to clear.")
    best = min(results["models"].items(), key=lambda kv: kv[1]["log_loss"])
    c3.metric(f"Best candidate ({best[0]})", f"{best[1]['log_loss']:.4f}")
    findings = DOCS / "research-prediction-findings.md"
    if findings.exists():
        with st.expander("Full findings", expanded=False):
            st.markdown(findings.read_text())


def drift_surprise_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "drift_surprise.json")
    if results is None:
        st.caption("Run `python -m src.drift_analysis` to generate.")
        return
    ps = results["pairs"].get("PS")
    if ps:
        reg, cvc = ps["regression"], ps["consensus_vs_closing"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pinnacle pair matches", f"{ps['n_matches']:,}",
                  help=ps["seasons"])
        c2.metric("Drift → miss (p-value)", f"{reg['logit_drift_p']:.3f}",
                  help="Logistic regression of favorite-miss on drift, "
                  "controlling for favorite strength.")
        c3.metric(
            "Big-move miss: consensus vs closing",
            f"{cvc['consensus_fav_miss_rate']:.1%} / "
            f"{cvc['closing_fav_miss_rate']:.1%}",
        )
    with st.expander("Per-book regression results"):
        for book, r in results["pairs"].items():
            reg = r["regression"]
            st.markdown(
                f"- **{book} → {r['closing']}** ({r['n_matches']:,} matches, "
                f"{r['seasons']}): drift coefficient z = "
                f"{reg['logit_drift_z']:+.2f}, p = {reg['logit_drift_p']:.3f}"
            )


def mismatch_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "mismatch_pockets.json")
    if results is None:
        st.caption("Run `python -m src.mismatch` to generate.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Scan rows", f"{results['n_discovery_rows'] + results['n_confirmation_rows']:,}")
    c2.metric("FDR candidates (discovery)", results["candidates_after_fdr"])
    c3.metric("Confirmed pockets", len(results["confirmed_pockets"]))


def influence_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "influence.json")
    if results is None:
        st.caption("Run `python -m src.influence` to generate.")
        return
    conv = {c["book"]: c["herding_coef"] for c in results["convergence"]}
    lead = {
        (r["leader"], r["follower"]): r["z"] for r in results["leadership"]
    }
    c1, c2, c3 = st.columns(3)
    c1.metric("Pinnacle herding", f"{conv.get('PS', float('nan')):.2f}",
              help="~0 = prices independently of the pack")
    c2.metric("Bet365 / WH herding",
              f"{conv.get('B365', float('nan')):.2f} / {conv.get('WH', float('nan')):.2f}")
    c3.metric("Pinnacle→B365 leadership z", f"{lead.get(('PS', 'B365'), float('nan')):+.1f}")


def seriea_lead_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "seriea_lead.json")
    if results is None:
        st.caption("Run `python -m src.seriea_lead` to generate.")
        return
    s, e = results["slope"], results["elo_vs_consensus"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Slope: discovery era",
        f"{s['discovery']['slope']:.3f}",
        help=f"z vs perfect calibration: {s['discovery']['z_vs_1']:+.1f}",
    )
    c2.metric(
        "Slope: confirmation era",
        f"{s['confirmation']['slope']:.3f}",
        help=f"z vs perfect calibration: {s['confirmation']['z_vs_1']:+.1f} — "
        "no longer significant",
    )
    c3.metric(
        "Elo edge confirmed?",
        "No",
        help=f"Discovery era has only {e['discovery']['n']} test matches "
        "(Pinnacle-era Serie A starts 2012-13), so the post-2018 result has "
        "no independent era to replicate on.",
    )


def asian_handicap_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "asian_handicap.json")
    if results is None:
        st.caption("Run `python -m src.asian_handicap` to generate.")
        return
    primary = results.get("primary_book")
    book = results["books"].get(primary) if primary else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches with AH odds", f"{results['coverage']:,}",
              help=f"of {results['n_matches']:,} total")
    if book:
        c2.metric(
            f"Efficiency slope ({book['name']})",
            f"{book['slope']:.3f} ± {book['slope_se']:.3f}",
            help="1.0 = the de-margined price is the true cover probability.",
        )
        c3.metric("Log loss / mean margin",
                  f"{book['log_loss']:.4f} / {book['mean_margin']:.1%}")
    slopes = results.get("efficiency_by_league", {})
    if slopes:
        with st.expander("Per-league efficiency slope"):
            for lg, slope in sorted(slopes.items()):
                st.markdown(f"- **{config.league_display_name(lg)}**: slope {slope:.3f}")


def club_elo_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "clubelo.json")
    if results is None:
        st.caption("Run `python -m src.clubelo` to generate.")
        return
    pooled = results.get("pooled")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Coverage (top-5 leagues)",
        f"{results['coverage'].get('ALL', 0):.1%}",
        help=f"{len(results.get('unmapped_teams', []))} team names unmapped.",
    )
    if pooled:
        vs = pooled["candidate_vs_consensus"]
        c2.metric(
            "Consensus + Club Elo vs consensus",
            f"{vs['diff']:+.4f}",
            help=f"95% CI [{vs['lo']:+.4f}, {vs['hi']:+.4f}] on "
            f"{pooled['n_test']:,} test matches. Negative = Club Elo helps.",
        )
        c3.metric(
            "Naive check",
            f"{pooled['naive']['clubelo_sign_accuracy']:.1%}",
            help="Picking the higher-Elo side vs always-home "
            f"({pooled['naive']['always_home_accuracy']:.1%}) — a sane "
            "rating should win this and still not beat the market.",
        )


def pricing_engine_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "pricing_engine.json")
    if results is None:
        st.caption("Run `python -m src.pricing_engine` to generate.")
        return
    cols = st.columns(max(len(results["markets"]), 1))
    names = {"1x2": "1X2", "totals": "O/U 2.5", "ah": "Asian handicap"}
    for col, (key, m) in zip(cols, results["markets"].items()):
        col.metric(
            f"{names.get(key, key)} gap",
            f"{m['gap']:+.4f}",
            help=f"engine {m['model_log_loss']:.4f} vs market "
            f"{m['market_log_loss']:.4f} on {m['n']:,} matches; "
            f"CI [{m['lo']:+.4f}, {m['hi']:+.4f}]",
        )


def model_stack_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "model_stack.json")
    if results is None:
        st.caption("Run `python -m src.model_stack` to generate.")
        return
    race = {r["model"]: r for r in results["race"]}
    stack = race.get("STACK (odds-free zoo)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test matches", f"{results['n_test']:,}", help=results["test_seasons"])
    if stack:
        c2.metric(
            "Odds-free stack vs market",
            f"{stack['diff_vs_market']:+.4f}",
            help=f"CI [{stack['lo']:+.4f}, {stack['hi']:+.4f}]; "
            f"stack log loss {stack['log_loss']:.4f}.",
        )
    weights = results.get("weights_with_market", [])
    if weights:
        import pandas as _pd

        w = _pd.DataFrame(weights)
        share = w[["mkt_logit_H", "mkt_logit_D"]].sum(axis=1).mean()
        c3.metric("Market share of learned weight", f"{share:.0%}",
                  help="When the market joins the stack.")


def live_ledger_research() -> None:
    from src.ledger import BACKTEST_BASELINE, load_ledger, scoreboard

    board = scoreboard(load_ledger())
    c1, c2, c3 = st.columns(3)
    c1.metric("Settled predictions", f"{board['n_settled']:,}")
    c2.metric("Awaiting results", f"{board.get('n_open', 0):,}")
    if board["n_settled"]:
        c3.metric(
            "Live log loss",
            f"{board['log_loss']:.4f}",
            help=f"vs backtest baseline {BACKTEST_BASELINE:.4f}; "
            f"CI [{board['log_loss_lo']:.4f}, {board['log_loss_hi']:.4f}]",
        )
    else:
        c3.metric("Live log loss", "—", help="No settled predictions yet.")


def match_stats_research() -> None:
    from common import load_matches as _load

    from src.halftime import comeback_rates, half_comparison

    df = _load()
    halves = half_comparison(df)
    backs = comeback_rates(df)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Goals: 1st → 2nd half",
        f"{halves['first_half_goals']:.2f} → {halves['second_half_goals']:.2f}",
    )
    c2.metric("Half-time leads overturned", f"{backs['comeback_rate']:.1%}",
              help=f"{backs['leads_overturned']:,} of "
              f"{backs['matches_with_a_lead']:,} leads.")
    home_flip = backs["home_led_and_lost"] / max(backs["home_led_at_ht"], 1)
    away_flip = backs["away_led_and_lost"] / max(backs["away_led_at_ht"], 1)
    c3.metric("Away vs home lead fragility",
              f"{away_flip:.1%} vs {home_flip:.1%}",
              help="Away leads are lost roughly twice as often.")


def shot_strength_research() -> None:
    results = _load_json(config.PROCESSED_DIR / "shot_strength.json")
    if results is None:
        st.caption("Run `python -m src.shot_strength` to generate.")
        return
    leagues = results.get("leagues", {})
    if not leagues:
        st.caption("No league results recorded.")
        return
    wins = sum(1 for r in leagues.values() if r["shots_better"])
    rows = [
        {"league": lg, "shots": r["shots_log_loss"], "goals": r["goals_log_loss"],
         "diff": r["diff"], "n": r["n_test"]}
        for lg, r in leagues.items()
    ]
    frame = pd.DataFrame(rows).set_index("league")
    c1, c2 = st.columns(2)
    c1.metric("Leagues where shots beat goals", f"{wins} / {len(leagues)}")
    c2.metric("Mean gap (shots − goals)", f"{frame['diff'].mean():+.4f}",
              help="Positive = the goals-based model forecasts better.")
    st.dataframe(
        frame.style.format({"shots": "{:.4f}", "goals": "{:.4f}",
                            "diff": "{:+.4f}", "n": "{:,}"}),
        use_container_width=True,
    )


# Registry: (title, verdict one-liner, dashboard page, renderer).
STUDIES = [
    {
        "title": "Can anything predict outcomes beyond the odds? (EPL)",
        "date": "2026-07-05",
        "verdict": (
            "Negative. Recalibration, line drift, cross-market totals and "
            "their ensemble never beat raw Pinnacle closing; drift and "
            "ensemble are significantly worse. Recipe: sharpest quote, "
            "Shin de-margined, stop there."
        ),
        "page": "Beat the Market / docs",
        "render": prediction_research,
    },
    {
        "title": "Do big pre-close moves predict results that defy the consensus? (EPL)",
        "date": "2026-07-05",
        "verdict": (
            "No, once favorite strength is controlled (Pinnacle pair "
            "p = 0.13; every other book pair p > 0.25). Big moves shift "
            "who is favored slightly — they do not flag chaotic matches. "
            "The closing favorite is surprised marginally less than the "
            "stale consensus favorite, consistent with moves being "
            "directional information already absorbed at the close."
        ),
        "page": "Drift vs Surprises",
        "render": drift_surprise_research,
    },
    {
        "title": "Are there confirmed mispriced pockets? (all 6 leagues)",
        "date": "2026-07-07",
        "verdict": (
            "No. A deliberate dredge across leagues, outcomes, odds bands, "
            "eras and season phases produced 7 FDR-significant candidates "
            "in the discovery era (≤2017-18) — and zero replicated on "
            "later seasons. Across ~183k (match, outcome) rows the market's "
            "calibration survives even a targeted search for its cracks."
        ),
        "page": "Mismatch Explorer",
        "render": mismatch_research,
    },
    {
        "title": "Do bookmakers copy each other's odds? (all 6 leagues)",
        "date": "2026-07-07",
        "verdict": (
            "Yes — with a strict hierarchy. Pinnacle prices almost "
            "independently (herding 0.10) while Bet365 and William Hill "
            "erase ~70% of their disagreement with the pack by kickoff. "
            "Leadership is one-way: Pinnacle's early prices predict the "
            "soft books' closing prices (z ≈ +23 to +25) but not the "
            "reverse; within the followers Bet365 leads William Hill. "
            "Snapshot resolution — see the page caveat."
        ),
        "page": "Bookmaker Influence",
        "render": influence_research,
    },
    {
        "title": "The Serie A anomaly: real crack or mirage? (I1)",
        "date": "2026-07-07",
        "verdict": (
            "Faded, not confirmed. Serie A showed the only notable anomaly "
            "of the six-league battery: calibration slope 1.103 and a "
            "nominal out-of-sample win for consensus+Elo. Era-split: the "
            "miscalibration was strong before 2018 (slope 1.127, z +4.2) "
            "but is gone in recent seasons (1.059, z +1.4) — the market "
            "tightened. The Elo edge lives almost entirely post-2018 and "
            "has no independent era to replicate on (369 discovery test "
            "matches). Verdict: a historical inefficiency that closed, "
            "plus an unconfirmed lead. Nothing actionable."
        ),
        "page": "Beat the Market (Serie A) / Calibration",
        "render": seriea_lead_research,
    },
    {
        "title": "Is the Asian-handicap market efficiently priced? (all 6 leagues)",
        "date": "2026-07-10",
        "verdict": (
            "Yes — and it is the tightest market of the three. With the draw "
            "removed and the line set near even money, the de-margined "
            "P(home covers) spans the full 0–1 range, so calibration is "
            "tested where the 2.5 totals line cannot. A logistic fit of "
            "realized cover on logit(implied) lands on slope ≈ 1 with "
            "intercept ≈ 0 across leagues; Pinnacle carries the thinnest "
            "margin and the sharpest score. No exploitable calibration gap. "
            "Same recipe as everywhere else: take the sharp two-way price, "
            "Shin de-margin, stop."
        ),
        "page": "Asian Handicap",
        "render": asian_handicap_research,
    },
    {
        "title": "Does external Club Elo add anything to the odds? (top-5 leagues)",
        "date": "2026-07-11",
        "verdict": (
            "DEFERRED — first EXTERNAL information source (clubelo.com "
            "ratings, built purely from results, never odds). Code, page "
            "and tests are complete, but api.clubelo.com was unreachable "
            "on 2026-07-11, so ratings coverage is ~0% pending a re-fetch "
            "(`python -m src.clubelo` when the API responds). The verdict "
            "slot stays open until the data lands."
        ),
        "page": "Club Elo",
        "render": club_elo_research,
    },
    {
        "title": "One engine, three markets: where does a stats model trail least?",
        "date": "2026-07-11",
        "verdict": (
            "A bookmaker-style origination engine (Dixon-Coles scoreline "
            "grid pricing 1X2, totals and Asian handicap consistently) was "
            "raced against the de-margined book prices per market on "
            "identical 2019-26 matches. The per-market gaps below show "
            "where public results data carries furthest against the "
            "market's private information."
        ),
        "page": "Pricing Engine",
        "render": pricing_engine_research,
    },
    {
        "title": "The model of models: can a stack of every predictor beat the market?",
        "date": "2026-07-11",
        "verdict": (
            "Stacking works — and still loses. On 32,780 test matches the "
            "meta-model over five odds-free systems (0.9997) beat every one "
            "of its members (best solo: Pi-ratings 1.0008; worst: "
            "Dixon-Coles 1.0118), confirming that their disagreements carry "
            "real information. But it lands +0.0165 behind the market "
            "(0.9832) — the platform's cleanest measurement of the market's "
            "informational moat. Let the market join the stack and the "
            "meta-model ties it exactly (+0.0000, CI [-0.0004, +0.0004]) "
            "while shifting 72.5% of its learned weight onto the market: "
            "the odds already contain the zoo. Recency-driven systems (Pi, "
            "Elo) beat window-snapshot ones (Massey, Colley) throughout."
        ),
        "page": "Model Stack",
        "render": model_stack_research,
    },
    {
        "title": "Forward test: does the market's calibration survive the future? (ongoing)",
        "date": "2026-07-24 — open",
        "verdict": (
            "The only study here with no verdict yet, by design. Every "
            "other finding is retrospective; this one records predictions "
            "before kickoff, freezes them when the match starts, and scores "
            "them afterwards — evidence hindsight cannot contaminate. The "
            "reference is the market consensus' backtest log loss (0.9832 "
            "over 32,780 historical matches). Below ~200 settled "
            "predictions the interval is too wide to conclude anything; "
            "the study is designed to be patient."
        ),
        "page": "Live Ledger",
        "render": live_ledger_research,
    },
    {
        "title": "The unused half of the data: half time, corners, cards, referees",
        "date": "2026-07-24",
        "verdict": (
            "The source has carried 12 match-statistic columns since 2000 "
            "that this platform never ingested. Findings: second halves "
            "produce ~27% more goals than first; half-time draws are far "
            "commoner than full-time ones (42% vs 26%); only ~8% of "
            "half-time leads are overturned, but **away leads are lost "
            "twice as often as home ones** — home advantage expressed as "
            "recovery. Corner and card rates drift by era, so recency "
            "weighting halves the calibration bias; even so, team "
            "tendencies barely beat a flat league average. Referees do "
            "show a real personal spread (0.64×–1.24×) once league and "
            "season confounds are removed."
        ),
        "page": "Half Time / Corners, Cards & Referees",
        "render": match_stats_research,
    },
    {
        "title": "Poor man's xG: is shot volume a better strength signal than goals?",
        "date": "2026-07-24",
        "verdict": (
            "**Negative, unanimously — goals win in all six leagues.** The "
            "identical Dixon-Coles machinery was fitted on shots on target "
            "(converted at the league rate) and on goals, then both were "
            "scored on the goals that actually followed. Shots lose by "
            "+0.017 to +0.030 nats, every confidence interval excluding "
            "zero. The lesson is about *which* shot information matters: "
            "real xG weights each attempt by where it was taken from, "
            "while a raw shot count treats a tap-in and a 30-yarder "
            "identically — and that discarded quality signal is evidently "
            "worth more than the extra sample size shots provide. It does "
            "not refute xG; it shows the cheap proxy for it fails."
        ),
        "page": "Research Log",
        "render": shot_strength_research,
    },
]

for study in STUDIES:
    st.subheader(study["title"])
    st.caption(f"{study['date']} · interactive version: **{study['page']}** page")
    st.markdown(study["verdict"])
    study["render"]()
    st.divider()
