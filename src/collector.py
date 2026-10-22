"""Private local collection. Network is used only by an explicit run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile
from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd
import fcntl
from src.live import SPORT_KEYS

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / 'data/live/settings.json'
DB_PATH = ROOT / 'data/live/collector.sqlite3'


def load_settings(path=None):
    path = Path(path or SETTINGS_PATH)
    result = {'enabled': False, 'leagues': list(SPORT_KEYS), 'hours': [0, 8, 16], 'api_key': ''}
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError('Invalid private settings')
        result.update({k: loaded[k] for k in result if k in loaded})
    if not isinstance(result['leagues'], list) or any(x not in SPORT_KEYS for x in result['leagues']):
        raise ValueError('Invalid league selection')
    result['hours'] = [0, 8, 16]
    return result


def save_settings(settings, path=None):
    path = Path(path or SETTINGS_PATH)
    current = load_settings(path)
    for key in ('enabled', 'leagues'):
        if key in settings:
            current[key] = settings[key]
    if settings.get('clear_key'):
        current['api_key'] = ''
    elif settings.get('api_key', '').strip():
        current['api_key'] = settings['api_key'].strip()
    if not isinstance(current['leagues'], list) or any(x not in SPORT_KEYS for x in current['leagues']):
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
    return {k: v for k, v in current.items() if k != 'api_key'}


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
    code = getattr(response, 'status_code', None)
    return {'error': type(exc).__name__, 'http_status': code if isinstance(code, int) else None}


def _quota(value):
    return {k: str(value[k])[:24] for k in ('used', 'remaining', 'last')
            if k in value and str(value[k]).replace('.', '', 1).isdigit()}


def public_status(db_path=None, settings_path=None):
    from src.live_store import LiveStore
    settings = load_settings(settings_path)
    store = LiveStore(Path(db_path or DB_PATH))
    try:
        collection = store.status()
        if collection.get('model'):
            collection['model'] = {k: collection['model'].get(k) for k in ('version', 'trained_at', 'training_cutoff', 'validation')}
        return {'enabled': settings['enabled'], 'key_configured': bool(os.getenv('ODDS_API_KEY') or settings['api_key']),
                'leagues': settings['leagues'], 'hours': [0, 8, 16], 'next_schedule': next_schedule(),
                'estimated_credits_daily': len(settings['leagues']) * 9,
                'estimated_credits_30_days': len(settings['leagues']) * 270, 'collection': collection}
    finally:
        store.close()


def run_collection(api_key=None, leagues=None, db_path=None, now=None, settings_path=None):
    from src.live import fetch_events, fetch_scores
    from src.live_store import LiveStore
    from src.live_learning import fit_candidate, predict_candidate
    settings = load_settings(settings_path)
    key = os.getenv('ODDS_API_KEY') or api_key or settings['api_key']
    selected = settings['leagues'] if leagues is None else list(leagues)
    if any(x not in SPORT_KEYS for x in selected):
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
        if not key:
            store.finish_run(run_id, 'needs_key', clock(), {'reason': 'Configure a private API key'})
            return {'status': 'needs_key', 'run_id': run_id}
        errors, quotas = {}, {}
        remaining, blocked = None, None

        def skip(league, kind, cost):
            reason = blocked or ('insufficient_quota' if remaining is not None and remaining < cost else None)
            if reason:
                details = {'reason': reason}
                errors[league + ':' + kind] = details
                store.mark_step(run_id, league, kind, 'skipped', details)
                return True
            return False

        def budget(quota):
            nonlocal remaining
            try:
                available = float(quota.get('remaining'))
                if available >= 0:
                    remaining = available
            except (TypeError, ValueError):
                pass

        def stop_on_error(exc):
            nonlocal blocked
            code = _error(exc)['http_status']
            if code in (401, 429):
                blocked = 'authentication_failed' if code == 401 else 'quota_exhausted'

        for league in selected:
            if store.step_done(run_id, league, 'scores'):
                continue
            if skip(league, 'scores', 2):
                continue
            try:
                events, quota = fetch_scores(key, sport=SPORT_KEYS[league], days_from=3)
                budget(quota)
                store.save_results(league, events, observed_at=clock())
                quotas[league + ':scores'] = _quota(quota)
                store.mark_step(run_id, league, 'scores', 'completed', {'quota': _quota(quota)})
            except Exception as exc:
                stop_on_error(exc)
                errors[league + ':scores'] = _error(exc)
                store.mark_step(run_id, league, 'scores', 'failed', _error(exc))
        as_of = clock()
        try:
            from src import config
            historical_path = config.PROCESSED_DIR / 'enriched.parquet'
            if not historical_path.exists() and config.PROCESSED_DIR == config.PROJECT_ROOT / 'data/processed':
                from src.bundle import bootstrap
                bootstrap()
            historical = pd.read_parquet(historical_path) if historical_path.exists() else pd.DataFrame()
            model = fit_candidate(historical, store.training_frame(as_of), as_of)
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
                events, quota = fetch_events(key, sport=SPORT_KEYS[league])
                captured = clock()
                budget(quota)
                rows = store.save_odds_snapshot(league, events, captured_at=captured, run_id=run_id)
                for row in rows:
                    baseline = [row['p_home'], row['p_draw'], row['p_away']]
                    metadata = {k: str(row[k]) for k in ('league', 'home_team', 'away_team', 'captured_at', 'kickoff')}
                    metadata['features'] = baseline
                    metadata['training_cutoff'] = model.get('training_cutoff') if model else None
                    store.save_prediction(row['snapshot_id'], 'market', baseline, 'baseline', metadata=metadata)
                    # A retry may encounter old snapshots: never backfill with a newer model.
                    eligible = model is not None and _utc(model['trained_at']) <= _utc(row['captured_at'])
                    candidate = predict_candidate(model, pd.DataFrame([row]))[0].tolist() if eligible else baseline
                    version = model['version'] if eligible else 'market'
                    trained_at = model['trained_at'] if eligible else None
                    if eligible:
                        store.save_prediction(row['snapshot_id'], version, candidate, 'candidate', trained_at=trained_at, metadata=metadata)
                    promoted = eligible and model.get('validation', {}).get('promoted', False)
                    store.save_prediction(row['snapshot_id'], version if promoted else 'market', candidate if promoted else baseline,
                                          'active', trained_at=trained_at if promoted else None, metadata=metadata)
                quotas[league + ':odds'] = _quota(quota)
                store.mark_step(run_id, league, 'odds', 'completed', {'quota': _quota(quota)})
            except Exception as exc:
                stop_on_error(exc)
                errors[league + ':odds'] = _error(exc)
                store.mark_step(run_id, league, 'odds', 'failed', _error(exc))
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
    parser.add_argument('--leagues', nargs='+', choices=list(SPORT_KEYS))
    parser.add_argument('--set-key', action='store_true', help='Prompt privately; key never appears in process arguments')
    parser.add_argument('--clear-key', action='store_true')
    parser.add_argument('--scheduled', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        scheduled = args.scheduled or os.getenv('ODDS_LAB_SCHEDULED') == '1'
        out = {'status': 'disabled'} if scheduled and not load_settings()['enabled'] else run_collection()
    elif args.command == 'status':
        out = public_status()
    elif args.command == 'configure':
        import getpass
        changes = {'clear_key': args.clear_key}
        if args.enabled:
            changes['enabled'] = args.enabled == 'yes'
        if args.leagues is not None:
            changes['leagues'] = args.leagues
        if args.set_key:
            changes['api_key'] = getpass.getpass('Private Odds API key: ')
        out = save_settings(changes)
    else:
        from src.collection_schedule import render_launch_agent, install_schedule, uninstall_schedule
        out = {'preview': render_launch_agent, 'install': install_schedule, 'uninstall': uninstall_schedule}[args.action or 'preview']()
    print(json.dumps(out, default=str))

if __name__ == '__main__':
    main()
