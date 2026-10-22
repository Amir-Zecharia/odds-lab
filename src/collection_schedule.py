"""Owned macOS LaunchAgent; calendar values use the Mac's local wall clock."""
from pathlib import Path
import os
import plistlib
import subprocess
import sys

LABEL = 'com.oddslab.collector'
ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')


def render_launch_agent(project_root=None, python_executable=None):
    root = Path(project_root or ROOT).absolute()
    executable = str(python_executable or (root / '.venv/bin/python'))
    return {'Label': LABEL, 'ProgramArguments': [executable, '-m', 'src.collector', 'run'],
            'WorkingDirectory': str(root), 'EnvironmentVariables': {'ODDS_LAB_SCHEDULED': '1'},
            'StartCalendarInterval': [{'Hour': hour, 'Minute': 0} for hour in (0, 8, 16)],
            'ProcessType': 'Background'}


def _owned(path):
    try:
        current = plistlib.loads(path.read_bytes())
    except Exception:
        return False
    return (current.get('Label') == LABEL and current.get('WorkingDirectory') == str(ROOT.resolve())
            and current.get('ProgramArguments', [])[1:] == ['-m', 'src.collector', 'run'])


def install_schedule():
    from src.collector import save_settings
    agent = render_launch_agent()
    payload = plistlib.dumps(agent)
    assert plistlib.loads(payload) == agent
    if AGENT_PATH.exists() and not _owned(AGENT_PATH):
        raise RuntimeError('Refusing to overwrite an unrelated LaunchAgent')
    AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(AGENT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
    os.chmod(AGENT_PATH, 0o600)
    target = f'gui/{os.getuid()}'
    subprocess.run(['launchctl', 'bootout', target + '/' + LABEL], capture_output=True)
    subprocess.run(['launchctl', 'bootstrap', target, str(AGENT_PATH)], check=True, capture_output=True)
    subprocess.run(['launchctl', 'print', target + '/' + LABEL], check=True, capture_output=True)
    save_settings({'enabled': True})
    return {'status': 'installed', 'path': str(AGENT_PATH), 'hours': [0, 8, 16]}


def uninstall_schedule():
    from src.collector import save_settings
    if AGENT_PATH.exists():
        if not _owned(AGENT_PATH):
            raise RuntimeError('Refusing to remove an unrelated LaunchAgent')
        subprocess.run(['launchctl', 'bootout', f'gui/{os.getuid()}/{LABEL}'], capture_output=True)
        AGENT_PATH.unlink()
    save_settings({'enabled': False})
    return {'status': 'uninstalled'}
