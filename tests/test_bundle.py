import hashlib
import importlib
import json
import shutil

import pandas as pd
import pytest


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'
    (root / 'processed').mkdir(parents=True)
    frame = pd.DataFrame({'match_id': ['m1'], 'league': ['E0'], 'season': ['2019-20'],
        'date': pd.to_datetime(['2020-01-01']), 'home_team': ['Home'], 'away_team': ['Away'],
        'home_goals': [1], 'away_goals': [0], 'result': ['H']})
    frame.to_parquet(root / 'processed/matches.parquet', index=False)
    frame.assign(p_cons_H=.5, p_cons_D=.3, p_cons_A=.2).to_parquet(root / 'processed/enriched.parquet', index=False)
    # Keep production inventory; only match tables use a tiny synthetic frame.
    from src import config
    for source in config.BUNDLE_DIR.rglob('*'):
        if source.is_file() and source.suffix in {'.json', '.csv'} and source.name != 'manifest.json':
            dest = root / source.relative_to(config.BUNDLE_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    files = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        data = path.read_bytes()
        files.append({'path': path.relative_to(root).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'schema_version': 1, 'files': files, 'required_assets': [f['path'] for f in files], 'dataset': {'rows': 1,
        'first_date': '2020-01-01', 'last_date': '2020-01-01', 'leagues': {'E0': 1}}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root, manifest


def test_bootstrap_installs_missing_files_and_is_resumable(bundle, tmp_path):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    target = tmp_path / 'data'
    assert api.verify_bundle(root) == manifest
    assert set(api.bootstrap(root, target)) == {x['path'] for x in manifest['files']}
    assert api.bootstrap(root, target) == []
    for entry in manifest['files']:
        assert (target / entry['path']).read_bytes() == (root / entry['path']).read_bytes()


def test_bootstrap_preserves_existing_work_and_ledger(bundle, tmp_path):
    api = importlib.import_module('src.bundle')
    root, _ = bundle
    target = tmp_path / 'data'
    existing = target / 'processed/matches.parquet'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'personal working data')
    ledger = target / 'ledger.csv'
    ledger.write_text('personal ledger')
    installed = api.bootstrap(root, target)
    assert 'processed/matches.parquet' not in installed
    assert 'processed/enriched.parquet' in installed
    assert existing.read_bytes() == b'personal working data'
    assert ledger.read_text() == 'personal ledger'


@pytest.mark.parametrize('mutation', ['checksum', 'size', 'schema', 'unsafe_path'])
def test_bad_bundle_rejected_before_any_install(bundle, tmp_path, mutation):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    if mutation == 'checksum':
        manifest['files'][0]['sha256'] = '0' * 64
    elif mutation == 'size':
        manifest['files'][0]['bytes'] += 1
    elif mutation == 'schema':
        manifest['schema_version'] = 999
    else:
        manifest['files'][0]['path'] = '../escape.parquet'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    target = tmp_path / 'data'
    with pytest.raises(ValueError):
        api.bootstrap(root, target)
    assert not list(target.rglob('*.parquet'))


def _rewrite_entry(root, manifest, relative):
    entry = next(item for item in manifest['files'] if item['path'] == relative)
    content = (root / relative).read_bytes()
    entry.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    (root / 'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('mutation', ['result', 'probability', 'metadata', 'columns'])
def test_semantically_invalid_data_rejected_even_with_valid_checksums(bundle, mutation):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    relative = 'processed/enriched.parquet'
    frame = pd.read_parquet(root / relative)
    if mutation == 'result':
        frame['result'] = 'A'
    elif mutation == 'probability':
        frame['p_cons_H'] = float('inf')
    elif mutation == 'metadata':
        manifest['dataset']['rows'] = 2
    else:
        frame = frame.drop(columns=['home_team'])
    frame.to_parquet(root / relative, index=False)
    _rewrite_entry(root, manifest, relative)
    with pytest.raises(ValueError):
        api.verify_bundle(root)


def test_symlink_destination_rejected_before_install(bundle, tmp_path):
    api = importlib.import_module('src.bundle')
    root, _ = bundle
    target = tmp_path / 'target'
    outside = tmp_path / 'outside'
    target.mkdir()
    outside.mkdir()
    (target / 'processed').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        api.bootstrap(root, target)
    assert list(outside.iterdir()) == []


def test_duplicate_or_forbidden_paths_rejected(bundle):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    manifest['files'].append(dict(manifest['files'][0]))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api.verify_bundle(root)
    manifest['files'][-1]['path'] = 'ledger.csv'
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api.verify_bundle(root)


def test_root_parent_alias_allowed_but_internal_symlink_rejected(bundle, tmp_path):
    api = importlib.import_module('src.bundle')
    root, _ = bundle
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path.parent, target_is_directory=True)
    aliased_parent = alias / tmp_path.name
    # /tmp on macOS is this kind of trusted ancestor alias.
    assert api.verify_bundle(aliased_parent / root.name)
    assert len(api.bootstrap(aliased_parent / root.name, aliased_parent / 'target')) == len(json.loads((root / 'manifest.json').read_text())['files'])
    (root / 'processed/enriched.parquet').unlink()
    (root / 'processed/enriched.parquet').symlink_to(aliased_parent / 'target/processed/enriched.parquet')
    with pytest.raises(ValueError):
        api.verify_bundle(aliased_parent / root.name)


@pytest.mark.parametrize('mutation', ['missing', 'null', 'duplicate', 'misaligned'])
def test_match_ids_required_unique_and_aligned(bundle, mutation):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    relative = 'processed/enriched.parquet'
    frame = pd.read_parquet(root / relative)
    if mutation == 'missing':
        frame = frame.drop(columns=['match_id'])
    elif mutation == 'null':
        frame['match_id'] = None
    elif mutation == 'duplicate':
        extra = frame.copy()
        extra['home_team'] = 'Other'
        frame = pd.concat([frame, extra], ignore_index=True)
    else:
        frame['match_id'] = 'different'
    frame.to_parquet(root / relative, index=False)
    _rewrite_entry(root, manifest, relative)
    with pytest.raises(ValueError):
        api.verify_bundle(root)


@pytest.mark.parametrize('mutation', ['null_leagues', 'null_timestamp', 'null_league_result', 'null_efficiency', 'bad_efficiency_slope'])
def test_consumed_analysis_types_rejected_with_valid_checksum(bundle, mutation):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    relative = 'processed/analysis.json'
    obj = json.loads((root / relative).read_text())
    if mutation == 'null_leagues':
        obj['leagues'] = None
    elif mutation == 'null_timestamp':
        obj['computed_at'] = None
    elif mutation == 'null_league_result':
        obj['leagues']['E0'] = None
    elif mutation == 'null_efficiency':
        obj['leagues']['E0']['efficiency'] = None
    else:
        obj['leagues']['E0']['efficiency']['slope'] = 'invalid'
    (root / relative).write_text(json.dumps(obj))
    _rewrite_entry(root, manifest, relative)
    with pytest.raises(ValueError):
        api.verify_bundle(root)


@pytest.mark.parametrize('mutation', ['parquet_only', 'missing_study', 'missing_cache', 'missing_inventory_marker'])
def test_production_inventory_cannot_be_downgraded(bundle, mutation):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    if mutation == 'parquet_only':
        manifest['files'] = [e for e in manifest['files'] if e['path'].endswith('.parquet')]
    elif mutation == 'missing_study':
        manifest['files'] = [e for e in manifest['files'] if e['path'] != 'processed/analysis.json']
    elif mutation == 'missing_cache':
        manifest['files'] = [e for e in manifest['files'] if e['path'] != 'raw/clubelo/Aachen.csv']
    else:
        manifest.pop('required_assets')
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api.verify_bundle(root)


@pytest.mark.parametrize('name,key,value', [
    ('influence', 'leadership', [None]),
    ('model_stack', 'race', {}),
    ('pricing_engine', 'markets', {'goals': None}),
    ('clubelo', 'coverage', None),
    ('shot_strength', 'leagues', {'E0': None}),
])
def test_other_consumed_study_structures(bundle, name, key, value):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    relative = f'processed/{name}.json'
    obj = json.loads((root / relative).read_text())
    obj[key] = value
    (root / relative).write_text(json.dumps(obj))
    _rewrite_entry(root, manifest, relative)
    with pytest.raises(ValueError):
        api.verify_bundle(root)


def test_manifest_cannot_redefine_required_assets_to_remove_studies(bundle):
    api = importlib.import_module('src.bundle')
    root, manifest = bundle
    manifest['files'] = [e for e in manifest['files'] if e['path'].endswith('.parquet')]
    manifest['required_assets'] = [e['path'] for e in manifest['files']]
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api.verify_bundle(root)
