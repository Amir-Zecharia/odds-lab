"""Transactional, append-only live observation and forecast ledger."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sqlite3
import uuid

import numpy as np
import pandas as pd

from src.live import parse_events, parse_completed_scores


def _time(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('A timezone-aware timestamp is required')
    return stamp.tz_convert('UTC').isoformat()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()
                if not any(part in str(k).lower().replace('_', '') for part in ('apikey', 'secret', 'token', 'password', 'authorization'))}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'(?i)(api[_-]?key[= :]+)[^&\s]+', r'\1[REDACTED]', value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_safe(value), sort_keys=True, allow_nan=False)


class LiveStore:
    def __init__(self, db_path: Path | None = None):
        path = Path(db_path) if db_path is not None else Path(__file__).resolve().parents[1] / 'data/live/collector.sqlite3'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS archives(id INTEGER PRIMARY KEY, league TEXT, run_id TEXT, captured_at TEXT, payload TEXT, UNIQUE(league,run_id));
        CREATE TABLE IF NOT EXISTS snapshots(snapshot_id INTEGER PRIMARY KEY, league TEXT, event_id TEXT, run_id TEXT, kickoff TEXT, captured_at TEXT, home_team TEXT, away_team TEXT, p_home REAL,p_draw REAL,p_away REAL,source TEXT,books INTEGER, UNIQUE(league,event_id,run_id));
        CREATE TABLE IF NOT EXISTS predictions(prediction_id INTEGER PRIMARY KEY,snapshot_id INTEGER REFERENCES snapshots(snapshot_id),model_version TEXT,kind TEXT,trained_at TEXT,metadata TEXT,p_home REAL,p_draw REAL,p_away REAL, UNIQUE(snapshot_id,kind));
        CREATE TABLE IF NOT EXISTS finals(observation_id INTEGER PRIMARY KEY,league TEXT,event_id TEXT,observed_at TEXT,kickoff TEXT,home_team TEXT,away_team TEXT,home_goals INTEGER,away_goals INTEGER,result TEXT,provider_updated_at TEXT);
        CREATE TABLE IF NOT EXISTS models(version TEXT PRIMARY KEY,trained_at TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,slot TEXT UNIQUE,started_at TEXT,finished_at TEXT,status TEXT,summary TEXT);
        CREATE TABLE IF NOT EXISTS steps(run_id TEXT,league TEXT,kind TEXT,status TEXT,details TEXT,PRIMARY KEY(run_id,league,kind));
        ''')
        self.conn.commit()

    def _frame(self, table):
        frame = pd.read_sql_query('SELECT * FROM ' + table, self.conn)
        for column in ('kickoff', 'captured_at', 'observed_at', 'provider_updated_at', 'trained_at', 'started_at', 'finished_at'):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def save_odds_snapshot(self, league, events, captured_at, run_id):
        captured = _time(captured_at)
        parsed = parse_events(events)
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO archives(league,run_id,captured_at,payload) VALUES(?,?,?,?)', (league, run_id, captured, _json(events)))
            for row in parsed.to_dict('records'):
                if not isinstance(row.get('event_id'), str) or not row['event_id'].strip():
                    continue
                self.conn.execute('''INSERT OR IGNORE INTO snapshots(league,event_id,run_id,kickoff,captured_at,home_team,away_team,p_home,p_draw,p_away,source,books) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                                  (league,row['event_id'],run_id,_time(row['kickoff']),captured,row['home_team'],row['away_team'],row['p_home'],row['p_draw'],row['p_away'],row['source'],row['books']))
        frame = self.snapshot_frame()
        frame = frame[(frame.league == league) & (frame.run_id == run_id) & (frame.captured_at < frame.kickoff)]
        return frame.to_dict('records')

    def save_prediction(self, snapshot_id, model_version, probabilities, kind, trained_at=None, metadata=None):
        if kind not in ('baseline', 'candidate', 'active'):
            raise ValueError('Invalid prediction kind')
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape != (3,) or not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1):
            raise ValueError('Expected normalized finite H/D/A probabilities')
        snapshot = self.conn.execute('SELECT captured_at,kickoff FROM snapshots WHERE snapshot_id=?', (snapshot_id,)).fetchone()
        if snapshot is None:
            raise ValueError('Unknown snapshot')
        trained = _time(trained_at) if trained_at is not None else None
        if snapshot[0] >= snapshot[1] or (trained is not None and trained > snapshot[0]):
            return False
        with self.conn:
            cursor = self.conn.execute('INSERT OR IGNORE INTO predictions(snapshot_id,model_version,kind,trained_at,metadata,p_home,p_draw,p_away) VALUES(?,?,?,?,?,?,?,?)',
                (snapshot_id,model_version,kind,trained,_json(metadata or {}),*probs.tolist()))
        return bool(cursor.rowcount)

    def save_results(self, league, events, observed_at):
        observed = _time(observed_at)
        count = 0
        with self.conn:
            for row in parse_completed_scores(events).to_dict('records'):
                if _time(row['kickoff']) > observed:
                    continue
                updated = None if pd.isna(row['provider_updated_at']) else _time(row['provider_updated_at'])
                values = (league,row['event_id'],observed,_time(row['kickoff']),row['home_team'],row['away_team'],row['home_goals'],row['away_goals'],row['result'],updated)
                previous = self.conn.execute('SELECT kickoff,home_team,away_team,home_goals,away_goals,result,provider_updated_at FROM finals WHERE league=? AND event_id=? ORDER BY observed_at DESC,observation_id DESC LIMIT 1', (league,row['event_id'])).fetchone()
                if previous == values[3:]:
                    continue
                self.conn.execute('INSERT INTO finals(league,event_id,observed_at,kickoff,home_team,away_team,home_goals,away_goals,result,provider_updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',values)
                count += 1
        return count

    def _truth(self, as_of=None):
        finals = self._frame('finals')
        if as_of is not None:
            cutoff = pd.Timestamp(_time(as_of))
            finals = finals[(finals.observed_at <= cutoff) & (finals.kickoff <= cutoff)]
        # Observation availability is filtered before ranking: future corrections
        # cannot leak into an earlier training cutoff. A delayed stale response
        # remains in history but cannot displace a newer provider version.
        # Missing provider times rank below known versions; if all are missing,
        # observation time and insertion identity provide a deterministic fallback.
        return finals.sort_values(['provider_updated_at','observed_at','observation_id'], na_position='first').drop_duplicates(['league','event_id'],keep='last')

    def training_frame(self, as_of):
        columns = ['event_id','league','home_team','away_team','kickoff','captured_at','p_home','p_draw','p_away','result','result_observed_at']
        cutoff = pd.Timestamp(_time(as_of))
        snapshots = self.snapshot_frame()
        truth = self._truth(as_of)
        joined = snapshots.merge(truth[['league','event_id','kickoff','result','observed_at']], on=['league','event_id'], suffixes=('','_final'))
        joined = joined[(joined.captured_at < joined.kickoff) & (joined.captured_at < joined.kickoff_final) & (joined.captured_at <= cutoff)]
        joined = joined.sort_values(['captured_at','snapshot_id']).drop_duplicates(['league','event_id'],keep='last')
        joined = joined.rename(columns={'observed_at':'result_observed_at'})
        return joined.reindex(columns=columns)

    def comparison_frame(self):
        predictions = self._frame('predictions')
        snapshots = self.snapshot_frame().drop(columns=['p_home','p_draw','p_away'])
        joined = predictions.merge(snapshots,on='snapshot_id')
        truth = self._truth()[['league','event_id','kickoff','result','observed_at']].rename(columns={'kickoff':'actual_kickoff','observed_at':'result_observed_at'})
        joined = joined.merge(truth,on=['league','event_id'],how='left')
        eligible = (joined.captured_at < joined.kickoff) & (joined.captured_at < joined.actual_kickoff)
        joined.loc[~eligible,'result'] = None
        joined['log_loss'] = np.nan
        joined['brier'] = np.nan
        for index, row in joined[joined.result.isin(['H','D','A'])].iterrows():
            p = np.asarray([row.p_home,row.p_draw,row.p_away],float)
            target = ['H','D','A'].index(row.result)
            joined.loc[index,'log_loss'] = -math.log(max(float(p[target]),1e-15))
            joined.loc[index,'brier'] = float(np.sum((p-np.eye(3)[target])**2))
        return joined

    def snapshot_frame(self):
        return self._frame('snapshots')

    def results_frame(self):
        return self._frame('finals')

    def runs_frame(self):
        return self._frame('runs')

    def save_model(self, model):
        version = model['version']
        trained = _time(model['trained_at'])
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO models VALUES(?,?,?)', (version,trained,_json(model)))

    def latest_model(self):
        row = self.conn.execute('SELECT payload FROM models ORDER BY trained_at DESC,rowid DESC LIMIT 1').fetchone()
        return json.loads(row[0]) if row else None

    def begin_run(self, slot, started_at):
        slot, started = _time(slot), _time(started_at)
        with self.conn:
            self.conn.execute('INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?)', (uuid.uuid4().hex,slot,started,None,'running','{}'))
            row = self.conn.execute('SELECT run_id,status FROM runs WHERE slot=?',(slot,)).fetchone()
            if row[1] == 'completed':
                return None
            self.conn.execute('UPDATE runs SET status=? WHERE run_id=?',('running',row[0]))
        return row[0]

    def finish_run(self, run_id, status, finished_at, summary):
        if status not in ('completed','partial','failed','needs_key'):
            raise ValueError('Invalid run status')
        with self.conn:
            self.conn.execute('UPDATE runs SET status=?,finished_at=?,summary=? WHERE run_id=?',(status,_time(finished_at),_json(summary),run_id))

    def step_done(self, run_id, league, kind):
        row = self.conn.execute('SELECT status FROM steps WHERE run_id=? AND league=? AND kind=?',(run_id,league,kind)).fetchone()
        return row is not None and row[0] == 'completed'

    def mark_step(self, run_id, league, kind, status, details=None):
        if kind not in ('odds','scores'):
            raise ValueError('Invalid step kind')
        with self.conn:
            self.conn.execute('INSERT INTO steps VALUES(?,?,?,?,?) ON CONFLICT(run_id,league,kind) DO UPDATE SET status=excluded.status,details=excluded.details',(run_id,league,kind,status,_json(details or {})))

    def status(self):
        runs = self.runs_frame()
        return {'last_run': None if runs.empty else _safe(runs.iloc[-1].to_dict()),
                'snapshots':len(self.snapshot_frame()),'results':len(self.results_frame()),
                'model':self.latest_model()}

    def close(self):
        self.conn.close()
