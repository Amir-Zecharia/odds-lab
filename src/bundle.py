"""Integrity checked, offline research snapshot; never replaces working data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config

STUDIES = {'analysis', 'asian_handicap', 'clubelo', 'drift_surprise', 'influence',
           'mismatch_pockets', 'model_stack', 'pricing_engine', 'research_prediction',
           'seriea_lead', 'shot_strength'}
CORE = {'match_id', 'league', 'season', 'date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result'}


# Snapshot cache inventory is pinned in code so deleting manifest entries cannot
# silently downgrade the offline distribution. Header-only histories are included.
CACHE_NAMES = {'Aachen', 'Ajaccio', 'Ajaccio_GFCO', 'Alaves', 'Albacete',
               'Almeria', 'Amiens', 'Ancona', 'Angers', 'Arles'}
REQUIRED_ASSETS = ({'processed/matches.parquet', 'processed/enriched.parquet'} |
                   {f'processed/{name}.json' for name in STUDIES} |
                   {f'raw/clubelo/{name}.csv' for name in CACHE_NAMES})
STUDY_TYPES = {
    'analysis': {'leagues': dict, 'computed_at': str},
    'asian_handicap': {'coverage': int, 'books': dict, 'efficiency_by_league': dict, 'computed_at': str},
    'clubelo': {'coverage': dict, 'leagues': dict, 'computed_at': str, 'unmapped_teams': list},
    'drift_surprise': {'pairs': dict},
    'influence': {'convergence': list, 'leadership': list, 'follow_the_sharp': list},
    'mismatch_pockets': {'confirmed_pockets': list, 'source': str},
    'model_stack': {'race': list, 'weights_with_market': list, 'computed_at': str},
    'pricing_engine': {'markets': dict, 'computed_at': str},
    'research_prediction': {'universe': dict, 'baselines': dict, 'models': dict},
    'seriea_lead': {'league': str, 'cutoff': str, 'slope': dict, 'elo_vs_consensus': dict},
    'shot_strength': {'leagues': dict, 'computed_at': str},
}


def _typed_fields(obj, fields, location):
    if not isinstance(obj, dict):
        raise ValueError(f'Expected study object: {location}')
    for key, expected in fields.items():
        if key not in obj or type(obj[key]) is not expected:
            raise ValueError(f'Invalid study field type: {location}.{key}')
        if expected is str and not obj[key].strip():
            raise ValueError(f'Empty study field: {location}.{key}')


def _study(obj, name):
    _typed_fields(obj, STUDY_TYPES[name], name)
    # Row collections feed DataFrames or direct per-row dictionary lookups.
    for key, expected in STUDY_TYPES[name].items():
        value = obj[key]
        if expected is list and key != 'unmapped_teams' and any(not isinstance(row, dict) for row in value):
            raise ValueError(f'Invalid study row: {name}.{key}')
    for key in ('leagues', 'pairs', 'markets'):
        if key in obj and (not isinstance(obj[key], dict) or any(not isinstance(v, dict) for v in obj[key].values())):
            raise ValueError(f'Invalid study mapping: {name}.{key}')
    if name == 'analysis':
        if not (set(config.LEAGUES) | {config.ALL_LEAGUES}).issubset(obj['leagues']):
            raise ValueError('Analysis missing configured league results')
        fields = {'efficiency': dict, 'efficiency_by_period': list, 'dispersion_error': dict,
                  'lr_tests': list, 'model_rows': int, 'oos_comparison': list, 'dixon_coles': dict}
        for league, result in obj['leagues'].items():
            _typed_fields(result, fields, f'analysis.leagues.{league}')
            for key in ('efficiency_by_period', 'lr_tests', 'oos_comparison'):
                if any(not isinstance(row, dict) for row in result[key]):
                    raise ValueError(f'Invalid analysis row: {league}.{key}')
            for key in ('intercept', 'slope', 'slope_se'):
                value = result['efficiency'].get(key)
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ValueError(f'Invalid calibration value: {league}.{key}')


def _relative(value):
    if not isinstance(value, str) or '\\' in value:
        raise ValueError('Invalid bundle path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError(f'Unsafe bundle path: {value}')
    allowed = (len(p.parts) == 2 and p.parts[0] == 'processed' and
               (p.name in {'matches.parquet', 'enriched.parquet'} or
                (p.suffix == '.json' and p.stem in STUDIES)))
    allowed |= (len(p.parts) == 3 and p.parts[:2] == ('raw', 'clubelo') and p.suffix == '.csv')
    if not allowed:
        raise ValueError(f'Unapproved bundle path: {value}')
    return p


def _safe(root, relative):
    # Ancestors may be OS aliases (for example macOS /tmp -> /private/tmp).
    # Only the chosen root and components inside it must not be symlinks.
    path = root
    for component in (None, *PurePosixPath(relative).parts):
        if component is not None:
            path = path / component
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed: {relative}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe bundle path: {relative}')
    return path


def _stats(frame):
    dates = pd.to_datetime(frame['date'], errors='raise')
    return {'rows': len(frame), 'first_date': dates.min().strftime('%Y-%m-%d'),
            'last_date': dates.max().strftime('%Y-%m-%d'),
            'leagues': {str(k): int(v) for k, v in frame['league'].value_counts().sort_index().items()}}


def _frame(path):
    try:
        df = pd.read_parquet(path)
        if df.empty or not CORE.issubset(df.columns) or df[list(CORE)].isna().any().any():
            raise ValueError('Missing required match data')
        if df['match_id'].duplicated().any():
            raise ValueError('Duplicate match identifiers')
        if not df['league'].isin(config.LEAGUES).all():
            raise ValueError('Unknown league')
        if not df['season'].astype(str).str.fullmatch(r'\d{4}-\d{2}').all():
            raise ValueError('Invalid season')
        for col in ['home_goals', 'away_goals']:
            values = pd.to_numeric(df[col], errors='raise')
            if not (np.isfinite(values).all() and (values >= 0).all() and (values % 1 == 0).all()):
                raise ValueError('Invalid goals')
        results = np.where(df.home_goals > df.away_goals, 'H', np.where(df.home_goals < df.away_goals, 'A', 'D'))
        if not (df.result == results).all():
            raise ValueError('Results disagree with scores')
        keys = ['league', 'season', 'date', 'home_team', 'away_team']
        if df.duplicated(keys).any():
            raise ValueError('Duplicate matches')
        _stats(df)
        if path.name == 'enriched.parquet':
            cols = ['p_cons_H', 'p_cons_D', 'p_cons_A']
            if not set(cols).issubset(df.columns):
                raise ValueError('Missing consensus probabilities')
            probabilities = df[cols].dropna()
            if (not np.isfinite(probabilities.to_numpy()).all() or
                not ((probabilities >= 0) & (probabilities <= 1)).all().all() or
                not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
                raise ValueError('Invalid consensus probabilities')
        return df
    except Exception as exc:
        raise ValueError(f'Invalid parquet schema or semantics: {path.name}: {exc}') from exc


def verify_bundle(bundle_dir: Path | None = None) -> dict:
    """Validate every byte and core match semantics before returning the manifest."""
    root = Path(bundle_dir) if bundle_dir is not None else config.BUNDLE_DIR
    try:
        manifest = json.loads(_safe(root, 'manifest.json').read_text())
        if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1 or not isinstance(manifest['files'], list):
            raise ValueError('Unsupported manifest schema')
        inventory = manifest.get('required_assets')
        if (not isinstance(inventory, list) or any(not isinstance(p, str) for p in inventory) or
                len(inventory) != len(set(inventory)) or not REQUIRED_ASSETS.issubset(inventory)):
            raise ValueError('Missing or incomplete required asset inventory')
        seen = set()
        frames = {}
        for entry in manifest['files']:
            relative = str(_relative(entry['path']))
            if relative in seen:
                raise ValueError('Duplicate manifest path')
            seen.add(relative)
            path = _safe(root, relative)
            content = path.read_bytes()
            if type(entry['bytes']) is not int or len(content) != entry['bytes']:
                raise ValueError(f'Byte size mismatch: {relative}')
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {relative}')
            if path.suffix == '.parquet':
                frames[path.name] = _frame(path)
                schema = {str(k): str(v) for k, v in frames[path.name].dtypes.items()}
            elif path.suffix == '.json':
                obj = json.loads(content)
                _study(obj, path.stem)
                schema = {'type': 'object', 'keys': sorted(obj)}
            else:
                csv = pd.read_csv(path)
                if not {'Club', 'Elo', 'From', 'To'}.issubset(csv.columns):
                    raise ValueError(f'Invalid Club Elo schema: {relative}')
                if not np.isfinite(pd.to_numeric(csv.Elo, errors='raise')).all():
                    raise ValueError(f'Invalid Club Elo values: {relative}')
                pd.to_datetime(csv['From'], errors='raise')
                pd.to_datetime(csv['To'], errors='raise')
                schema = {str(k): str(v) for k, v in csv.dtypes.items()}
            if 'schema' in entry and entry['schema'] != schema:
                raise ValueError(f'Schema metadata mismatch: {relative}')
        if seen != set(inventory):
            raise ValueError('Manifest files do not match required asset inventory')
        if not {'matches.parquet', 'enriched.parquet'}.issubset(frames):
            raise ValueError('Required match files missing')
        matches, enriched = frames['matches.parquet'], frames['enriched.parquet']
        if _stats(matches) != manifest['dataset']:
            # Production manifests also record season coverage.
            if any(manifest['dataset'].get(k) != v for k, v in _stats(matches).items()):
                raise ValueError('Dataset metadata mismatch')
        if 'seasons' in manifest['dataset'] and manifest['dataset']['seasons'] != sorted(matches.season.unique().tolist()):
            raise ValueError('Season coverage metadata mismatch')
        keys = sorted(CORE)
        if not matches[keys].reset_index(drop=True).equals(enriched[keys].reset_index(drop=True)):
            raise ValueError('Enriched data does not align with matches')
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid or missing bundle: {exc}') from exc


def bootstrap(bundle_dir: Path | None = None, data_dir: Path | None = None) -> list[str]:
    """Install missing files atomically. Existing working files are preserved."""
    root = Path(bundle_dir) if bundle_dir is not None else config.BUNDLE_DIR
    target = Path(data_dir) if data_dir is not None else config.PROJECT_ROOT / 'data'
    manifest = verify_bundle(root)
    installed = []
    # Check all destinations before creating anything.
    for entry in manifest['files']:
        _safe(target, entry['path'])
    for entry in manifest['files']:
        relative = entry['path']
        dest = _safe(target, relative)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.bundle-', dir=dest.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                content = _safe(root, relative).read_bytes()
                if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
                    raise ValueError(f'Bundle changed during installation: {relative}')
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link publishes complete bytes atomically without clobbering
            # another process or a working file created since the existence test.
            try:
                os.link(temporary, dest)
                installed.append(relative)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    return installed


def build_bundle(bundle_dir: Path | None = None, data_dir: Path | None = None) -> dict:
    """Explicit maintainer operation: copy allowlisted inputs, then validate."""
    root = Path(bundle_dir) if bundle_dir is not None else config.BUNDLE_DIR
    source = Path(data_dir) if data_dir is not None else config.PROJECT_ROOT / 'data'
    if root.exists():
        raise ValueError('Build destination already exists; choose a new directory')
    paths = [source / 'processed' / f for f in ['matches.parquet', 'enriched.parquet']]
    paths += [source / 'processed' / f'{name}.json' for name in sorted(STUDIES)]
    paths += sorted((source / 'raw' / 'clubelo').glob('*.csv'))
    entries = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        _relative(relative)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        content = dest.read_bytes()
        if path.suffix == '.parquet':
            schema = {str(k): str(v) for k, v in pd.read_parquet(path).dtypes.items()}
        elif path.suffix == '.json':
            schema = {'type': 'object', 'keys': sorted(json.loads(content))}
        else:
            schema = {str(k): str(v) for k, v in pd.read_csv(path).dtypes.items()}
        entries.append({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                        'schema': schema, 'source': 'Club Elo' if path.suffix == '.csv' else 'Football-Data.co.uk; local derived studies',
                        'original_retrieved_at': None})
    frame = _frame(root / 'processed/matches.parquet')
    dataset = _stats(frame)
    dataset['seasons'] = sorted(frame.season.unique().tolist())
    manifest = {'schema_version': 1, 'assembled_at': datetime.now(timezone.utc).isoformat(),
                'original_retrieved_at': None, 'dataset': dataset, 'files': entries,
                'required_assets': sorted(entry['path'] for entry in entries),
                'validation': {'checks': ['sha256', 'byte_size', 'allowlisted_paths', 'schema', 'score_result_consistency',
                  'unique_fixtures', 'enriched_alignment', 'consensus_probability_bounds_and_sum', 'coverage_metadata'],
                  'status': 'passed', 'scope': 'Integrity and internal consistency; not independent factual accuracy verification'},
                'sources': {'football_data': 'https://www.football-data.co.uk/', 'club_elo': 'http://clubelo.com/',
                  'terms': 'Personal/research terms do not establish an explicit redistribution license; see DATA_SOURCES.md'}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_bundle(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'install', 'build'])
    parser.add_argument('--bundle-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        result = verify_bundle(args.bundle_dir)
        print(f"Verified {len(result['files'])} files; {result['dataset']['rows']:,} matches")
    elif args.command == 'install':
        print(f'Installed {len(bootstrap(args.bundle_dir, args.data_dir))} missing files')
    else:
        result = build_bundle(args.bundle_dir, args.data_dir)
        print(f"Built and verified {len(result['files'])} files")


if __name__ == '__main__':
    main()
