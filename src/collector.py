"""Private local collection. Network is used only by an explicit run."""
from __future__ import annotations
import argparse
import json
import hashlib
import os
from pathlib import Path
import tempfile
from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd
import fcntl
from src.live import SPORT_KEYS
from src.live_leagues import LIVE_LEAGUES, ISRAEL_LEAGUE

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / 'data/live/settings.json'
DB_PATH = ROOT / 'data/live/collector.sqlite3'


def load_settings(path=None, *, _include_env=True):
    path = Path(path or SETTINGS_PATH)
    result = {'enabled': False, 'leagues': list(SPORT_KEYS), 'hours': [0, 8, 16], 'api_key': '', 'api_football_key': ''}
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError('Invalid private settings')
        result.update({k: loaded[k] for k in result if k in loaded})
    if not isinstance(result['leagues'], list) or any(x not in LIVE_LEAGUES for x in result['leagues']):
        raise ValueError('Invalid league selection')
    result['hours'] = [0, 8, 16]
    if _include_env and os.getenv('API_FOOTBALL_KEY'):
        result['api_football_key'] = os.environ['API_FOOTBALL_KEY']
    return result


def save_settings(settings, path=None):
    path = Path(path or SETTINGS_PATH)
    current = load_settings(path, _include_env=False)
    for key in ('enabled', 'leagues'):
        if key in settings:
            current[key] = settings[key]
    if settings.get('clear_key'):
        current['api_key'] = ''
    elif settings.get('api_key', '').strip():
        current['api_key'] = settings['api_key'].strip()
    if settings.get('clear_api_football_key'):
        current['api_football_key'] = ''
    elif settings.get('api_football_key', '').strip():
        current['api_football_key'] = settings['api_football_key'].strip()
    if not isinstance(current['leagues'], list) or any(x not in LIVE_LEAGUES for x in current['leagues']):
        raise ValueError('Invalid league selection')
    current['enabled'] = bool(current['enabled'])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.settings-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(current, handle)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {k: v for k, v in current.items() if k not in ('api_key', 'api_football_key')}


def _utc(value=None):
    stamp = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Clock must be timezone aware')
    return stamp.tz_convert('UTC')


def _local_timezone():
    name = os.environ.get('TZ', '').removeprefix(':')
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    # Read the system tzfile rather than datetime's fixed current offset.
    with open('/etc/localtime', 'rb') as handle:
        return ZoneInfo.from_file(handle)


def scheduled_slot(now=None):
    local = _utc(now).to_pydatetime().astimezone(_local_timezone())
    hour = max(h for h in (0, 8, 16) if h <= local.hour)
    return local.replace(hour=hour, minute=0, second=0, microsecond=0, fold=0).isoformat()


def next_schedule(now=None):
    local = _utc(now).to_pydatetime().astimezone(_local_timezone())
    for hour in (0, 8, 16):
        future = local.replace(hour=hour, minute=0, second=0, microsecond=0, fold=0)
        if future > local:
            return future.isoformat()
    return (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0, fold=0).isoformat()


def _error(exc):
    # Request exception strings can contain query-string credentials and bodies.
    response = getattr(exc, 'response', None)
    code = getattr(exc, 'status_code', None) or getattr(response, 'status_code', None)
    details = {'error': type(exc).__name__, 'http_status': code if isinstance(code, int) else None}
    reason = getattr(exc, 'reason', None)
    if reason in ('authentication', 'quota', 'coverage', 'provider_error', 'ambiguous_league', 'pagination', 'invalid_response', 'network'):
        details['reason'] = reason
    return details


def _quota(value):
    return {k: str(value[k])[:24] for k in ('used', 'remaining', 'last', 'limit', 'minute_remaining', 'minute_limit')
            if k in value and str(value[k]).replace('.', '', 1).isdigit()}


def public_status(db_path=None, settings_path=None):
    from src.live_store import LiveStore
    settings = load_settings(settings_path)
    store = LiveStore(Path(db_path or DB_PATH))
    try:
        collection = store.status()
        if collection.get('model'):
            original = collection['model']
            fields = ('version', 'trained_at', 'training_cutoff', 'validation')
            collection['model'] = {k: original.get(k) for k in fields}
            collection['model']['league_models'] = {code: {k: model.get(k) for k in fields}
                for code, model in original.get('league_models', {}).items()}
        odds_count = sum(LIVE_LEAGUES[code]['provider'] == 'odds_api' for code in settings['leagues'])
        return {'enabled': settings['enabled'], 'key_configured': bool(os.getenv('ODDS_API_KEY') or settings['api_key']),
                'api_football_key_configured': bool(settings['api_football_key']),
                'provider_estimates': {'odds_api': {'credits_daily': odds_count * 9, 'credits_30_days': odds_count * 270},
                                       'api_football': {'requests_depend_on_paging': True, 'coverage': 'unverified'}},
                'leagues': settings['leagues'], 'hours': [0, 8, 16], 'next_schedule': next_schedule(),
                'estimated_credits_daily': odds_count * 9,
                'estimated_credits_30_days': odds_count * 270, 'collection': collection}
    finally:
        store.close()


def run_collection(api_key=None, leagues=None, db_path=None, now=None, settings_path=None, api_football_key=None):
    from src.live import fetch_events, fetch_scores
    from src.live_store import LiveStore
    from src.live_learning import fit_candidate, predict_candidate
    settings = load_settings(settings_path)
    key = os.getenv('ODDS_API_KEY') or api_key or settings['api_key']
    keys = {'odds_api': key, 'api_football': os.getenv('API_FOOTBALL_KEY') or api_football_key or settings['api_football_key']}
    selected = settings['leagues'] if leagues is None else list(leagues)
    if any(x not in LIVE_LEAGUES for x in selected):
        raise ValueError('Invalid league selection')
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    clock = (lambda: _utc(now() if callable(now) else now))
    lock = open(str(path) + '.lock', 'a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {'status': 'busy'}
    store = None
    try:
        store = LiveStore(path)
        run_id = store.begin_run(scheduled_slot(clock()), clock())
        if run_id is None:
            return {'status': 'already_completed'}
        if not any(keys[LIVE_LEAGUES[league]['provider']] for league in selected):
            store.finish_run(run_id, 'needs_key', clock(), {'reason': 'Configure a private API key'})
            return {'status': 'needs_key', 'run_id': run_id}
        errors, quotas = {}, {}
        remaining = {'odds_api': None, 'api_football': None}
        blocked = {provider: None if value else 'missing_key' for provider, value in keys.items()}
        football = None

        def request(league, kind):
            nonlocal football
            if LIVE_LEAGUES[league]['provider'] == 'api_football':
                if football is None:
                    from src.api_football import ApiFootballClient
                    football = ApiFootballClient(keys['api_football'], now=clock)
                return getattr(football, 'fetch_' + ('scores' if kind == 'scores' else 'events'))()
            if kind == 'scores':
                return fetch_scores(keys['odds_api'], sport=SPORT_KEYS[league], days_from=3)
            return fetch_events(keys['odds_api'], sport=SPORT_KEYS[league])

        def skip(league, kind, cost):
            provider = LIVE_LEAGUES[league]['provider']
            available = remaining[provider]
            reason = blocked[provider] or ('insufficient_quota' if available is not None and available < cost else None)
            if reason:
                details = {'reason': reason, 'provider': provider}
                errors[league + ':' + kind] = details
                store.mark_step(run_id, league, kind, 'skipped', details)
                return True
            return False

        def budget(league, quota):
            try:
                available = float(quota.get('remaining'))
                if available >= 0:
                    remaining[LIVE_LEAGUES[league]['provider']] = available
            except (TypeError, ValueError):
                pass

        def stop_on_error(league, exc):
            provider = LIVE_LEAGUES[league]['provider']
            code = _error(exc)['http_status']
            reason = getattr(exc, 'reason', None)
            if code in (401, 403, 429) or reason in ('authentication', 'quota'):
                blocked[provider] = 'authentication_failed' if code in (401, 403) or reason == 'authentication' else 'quota_exhausted'

        def failure_details(league, kind, exc):
            details = _error(exc)
            if LIVE_LEAGUES[league]['provider'] == 'api_football' and football is not None:
                quota = _quota(football.quota)
                if quota:
                    budget(league, quota)
                    quotas[league + ':' + kind] = quota
                    details['quota'] = quota
            return details

        for league in selected:
            if store.step_done(run_id, league, 'scores'):
                continue
            if skip(league, 'scores', 2 if LIVE_LEAGUES[league]['provider'] == 'odds_api' else 1):
                continue
            try:
                events, quota = request(league, 'scores')
                budget(league, quota)
                store.save_results(league, events, observed_at=clock())
                quotas[league + ':scores'] = _quota(quota)
                store.mark_step(run_id, league, 'scores', 'completed', {'quota': _quota(quota)})
            except Exception as exc:
                stop_on_error(league, exc)
                details = failure_details(league, 'scores', exc)
                errors[league + ':scores'] = details
                store.mark_step(run_id, league, 'scores', 'failed', details)
        as_of = clock()
        try:
            from src import config
            historical_path = config.PROCESSED_DIR / 'enriched.parquet'
            if not historical_path.exists() and config.PROCESSED_DIR == config.PROJECT_ROOT / 'data/processed':
                from src.bundle import bootstrap
                bootstrap()
            historical = pd.read_parquet(historical_path) if historical_path.exists() else pd.DataFrame()
            live_training = store.training_frame(as_of)
            european = live_training.loc[live_training.league != ISRAEL_LEAGUE] if 'league' in live_training else live_training
            historical = historical.loc[historical.league != ISRAEL_LEAGUE] if 'league' in historical else historical
            model = fit_candidate(historical, european, as_of)
            # Always preserve an existing Israel candidate, even on a Europe-only run.
            previous = store.latest_model() or {}
            if ISRAEL_LEAGUE in selected or ISRAEL_LEAGUE in previous.get('league_models', {}):
                israel_training = live_training.loc[live_training.league == ISRAEL_LEAGUE] if 'league' in live_training else pd.DataFrame()
                model['league_models'] = {ISRAEL_LEAGUE: fit_candidate(pd.DataFrame(), israel_training, as_of)}
                payload = {k: v for k, v in model.items() if k != 'version'}
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
                model['version'] = 'live-recalibration-v1-' + digest
            store.save_model(model)
        except Exception as exc:
            errors['learning'] = _error(exc)
            model = None
        for league in selected:
            if store.step_done(run_id, league, 'odds'):
                continue
            if skip(league, 'odds', 1):
                continue
            try:
                events, quota = request(league, 'odds')
                captured = clock()
                budget(league, quota)
                rows = store.save_odds_snapshot(league, events, captured_at=captured, run_id=run_id)
                selected_model = model.get('league_models', {}).get(ISRAEL_LEAGUE) if league == ISRAEL_LEAGUE and model else model
                for row in rows:
                    baseline = [row['p_home'], row['p_draw'], row['p_away']]
                    metadata = {k: str(row[k]) for k in ('league', 'home_team', 'away_team', 'captured_at', 'kickoff')}
                    metadata['features'] = baseline
                    metadata['training_cutoff'] = selected_model.get('training_cutoff') if selected_model else None
                    store.save_prediction(row['snapshot_id'], 'market', baseline, 'baseline', metadata=metadata)
                    # A retry may encounter old snapshots: never backfill with a newer selected_model.
                    eligible = selected_model is not None and _utc(selected_model['trained_at']) <= _utc(row['captured_at'])
                    candidate = predict_candidate(selected_model, pd.DataFrame([row]))[0].tolist() if eligible else baseline
                    version = selected_model['version'] if eligible else 'market'
                    trained_at = selected_model['trained_at'] if eligible else None
                    if eligible:
                        store.save_prediction(row['snapshot_id'], version, candidate, 'candidate', trained_at=trained_at, metadata=metadata)
                    promoted = eligible and selected_model.get('validation', {}).get('promoted', False)
                    store.save_prediction(row['snapshot_id'], version if promoted else 'market', candidate if promoted else baseline,
                                          'active', trained_at=trained_at if promoted else None, metadata=metadata)
                quotas[league + ':odds'] = _quota(quota)
                store.mark_step(run_id, league, 'odds', 'completed', {'quota': _quota(quota)})
            except Exception as exc:
                stop_on_error(league, exc)
                details = failure_details(league, 'odds', exc)
                errors[league + ':odds'] = details
                store.mark_step(run_id, league, 'odds', 'failed', details)
        status = 'partial' if errors else 'completed'
        summary = {'status': status, 'run_id': run_id, 'errors': errors, 'quota': quotas}
        store.finish_run(run_id, status, clock(), summary)
        return summary
    finally:
        if store is not None:
            store.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status', 'configure', 'schedule'])
    parser.add_argument('action', nargs='?', choices=['preview', 'install', 'uninstall'])
    parser.add_argument('--enabled', choices=['yes', 'no'])
    parser.add_argument('--leagues', nargs='+', choices=list(LIVE_LEAGUES))
    parser.add_argument('--set-key', action='store_true', help='Prompt privately; key never appears in process arguments')
    parser.add_argument('--clear-key', action='store_true')
    parser.add_argument('--set-api-football-key', action='store_true', help='Prompt privately for the direct API-Football key')
    parser.add_argument('--clear-api-football-key', action='store_true')
    parser.add_argument('--scheduled', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        scheduled = args.scheduled or os.getenv('ODDS_LAB_SCHEDULED') == '1'
        out = {'status': 'disabled'} if scheduled and not load_settings()['enabled'] else run_collection()
    elif args.command == 'status':
        out = public_status()
    elif args.command == 'configure':
        import getpass
        changes = {'clear_key': args.clear_key, 'clear_api_football_key': args.clear_api_football_key}
        if args.enabled:
            changes['enabled'] = args.enabled == 'yes'
        if args.leagues is not None:
            changes['leagues'] = args.leagues
        if args.set_key:
            changes['api_key'] = getpass.getpass('Private Odds API key: ')
        if args.set_api_football_key:
            changes['api_football_key'] = getpass.getpass('Private API-Football key: ')
        out = save_settings(changes)
    else:
        from src.collection_schedule import render_launch_agent, install_schedule, uninstall_schedule
        out = {'preview': render_launch_agent, 'install': install_schedule, 'uninstall': uninstall_schedule}[args.action or 'preview']()
    print(json.dumps(out, default=str))

if __name__ == '__main__':
    main()
