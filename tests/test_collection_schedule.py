import plistlib
from pathlib import Path


def test_launch_agent_calendar_is_three_local_wall_clock_times():
    from src.collection_schedule import render_launch_agent
    agent = render_launch_agent(project_root=Path('/tmp/project with spaces'), python_executable='/tmp/python')
    assert agent['Label'] == 'com.oddslab.collector'
    assert sorted((item['Hour'], item['Minute']) for item in agent['StartCalendarInterval']) == [(0,0),(8,0),(16,0)]
    assert agent['ProgramArguments'] == ['/tmp/python', '-m', 'src.collector', 'run']
    assert agent['WorkingDirectory'] == '/tmp/project with spaces'
    assert plistlib.loads(plistlib.dumps(agent)) == agent


def test_launch_agent_does_not_embed_environment_key(monkeypatch):
    from src.collection_schedule import render_launch_agent
    monkeypatch.setenv('ODDS_API_KEY', 'secret-sentinel')
    agent = render_launch_agent(project_root=Path('/tmp/project'), python_executable='/tmp/python')
    assert b'secret-sentinel' not in plistlib.dumps(agent)
    assert 'ODDS_API_KEY' not in agent.get('EnvironmentVariables', {})


def test_schedule_never_runs_on_load_and_disabled_guard():
    from src.collection_schedule import render_launch_agent
    agent = render_launch_agent()
    assert not agent.get('RunAtLoad', False)
    assert agent['EnvironmentVariables'] == {'ODDS_LAB_SCHEDULED': '1'}


def test_refuse_unrelated_agent(tmp_path, monkeypatch):
    import pytest
    import src.collection_schedule as schedule
    path = tmp_path/'agent.plist'
    path.write_bytes(plistlib.dumps({'Label':'someone.else'}))
    monkeypatch.setattr(schedule, 'AGENT_PATH', path)
    with pytest.raises(RuntimeError, match='unrelated'):
        schedule.install_schedule()
    with pytest.raises(RuntimeError, match='unrelated'):
        schedule.uninstall_schedule()
    assert plistlib.loads(path.read_bytes())['Label'] == 'someone.else'
