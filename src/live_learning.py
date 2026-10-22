"""Chronological, JSON-only market recalibration for subsequent fixtures."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluate import brier_per_match, log_loss_per_match
from src.ledger import _match_key

CLASSES = ['H', 'D', 'A']


def _utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError('as_of must be timezone aware')
    return value.tz_convert('UTC')


def _market(frame):
    p = frame[['p_home', 'p_draw', 'p_away']].to_numpy(dtype=float)
    p = np.nan_to_num(p, nan=1 / 3, posinf=1, neginf=0)
    p = np.clip(p, 1e-8, 1)
    return p / p.sum(axis=1, keepdims=True)


def _features(frame):
    logs = np.log(np.clip(_market(frame), 1e-8, 1))
    rows = []
    for i, row in enumerate(frame.itertuples()):
        league = str(row.league)
        home, away = _match_key(row.home_team, row.away_team).split('|')
        rows.append({'log_H': float(logs[i, 0]), 'log_D': float(logs[i, 1]),
                     'log_A': float(logs[i, 2]), 'league': league,
                     'home': league + ':' + home, 'away': league + ':' + away})
    return rows


def _prepare(historical, live, cutoff):
    parts = []
    for source, original in [('historical', historical), ('live', live)]:
        if original.empty:
            continue
        f = original.copy()
        if source == 'historical':
            if not {'date', 'result', 'league', 'home_team', 'away_team'} <= set(f):
                continue
            # Historical dates have day precision: conservatively available next day.
            f['kickoff'] = pd.to_datetime(f['date'], utc=True, errors='coerce')
            f['result_observed_at'] = f['kickoff'].dt.floor('D') + pd.Timedelta(days=1)
            f['captured_at'] = f['kickoff'] - pd.Timedelta(seconds=1)
            cons = f.reindex(columns=['p_cons_H', 'p_cons_D', 'p_cons_A']).to_numpy(float)
            sharp = f.reindex(columns=['p_PS_H', 'p_PS_D', 'p_PS_A']).to_numpy(float)
            valid = np.isfinite(sharp).all(axis=1) & (sharp > 0).all(axis=1)
            probs = np.where(valid[:, None], sharp, cons)
            f[['p_home', 'p_draw', 'p_away']] = probs
            f['event_id'] = ['historical:' + str(r.league) + ':' + str(r.kickoff) + ':' +
                             _match_key(r.home_team, r.away_team) for r in f.itertuples()]
        required = {'kickoff', 'captured_at', 'result_observed_at', 'event_id', 'result',
                    'league', 'home_team', 'away_team', 'p_home', 'p_draw', 'p_away'}
        if not required <= set(f):
            continue
        for col in ['kickoff', 'captured_at', 'result_observed_at']:
            f[col] = pd.to_datetime(f[col], utc=True, errors='coerce')
        probs = f[['p_home', 'p_draw', 'p_away']].apply(pd.to_numeric, errors='coerce').to_numpy()
        valid = np.isfinite(probs).all(axis=1) & (probs > 0).all(axis=1)
        valid &= (f.captured_at < f.kickoff) & (f.kickoff < cutoff)
        valid &= (f.captured_at <= cutoff) & (f.result_observed_at <= cutoff)
        valid &= (f.result_observed_at >= f.kickoff) & f.result.isin(CLASSES)
        valid &= f.event_id.notna() & f.league.notna() & f.home_team.notna() & f.away_team.notna()
        f = f.loc[valid].copy()
        f['source'] = source
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True).sort_values(['captured_at', 'result_observed_at'])
    # Match identities prevent both snapshot weighting and historical/live overlap.
    f['fixture_key'] = [str(r.league) + ':' + str(r.kickoff.date()) + ':' +
                        _match_key(r.home_team, r.away_team) for r in f.itertuples()]
    f = f.drop_duplicates(['league', 'event_id'], keep='last')
    # Prefer provider-identified live observations when historical fixtures overlap.
    f['source_rank'] = (f.source == 'live').astype(int)
    return (f.sort_values(['source_rank', 'captured_at'])
            .drop_duplicates('fixture_key', keep='last').sort_values('kickoff').reset_index(drop=True))


def _fit(frame):
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(_features(frame))
    estimator = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs', random_state=20260704)
    estimator.fit(x, frame.result)
    return {'coefficients': estimator.coef_.tolist(), 'intercept': estimator.intercept_.tolist(),
            'vocabulary': dict(vectorizer.vocabulary_), 'classes': estimator.classes_.tolist()}


def predict_candidate(model, frame):
    """Candidate H/D/A probabilities; caller applies the promotion decision."""
    if frame.empty:
        return np.empty((0, 3))
    if model.get('kind') != 'multinomial':
        return _market(frame)
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.vocabulary_ = model['vocabulary']
    vectorizer.feature_names_ = sorted(model['vocabulary'], key=model['vocabulary'].get)
    logits = vectorizer.transform(_features(frame)) @ np.asarray(model['coefficients']).T
    logits = np.clip(logits + np.asarray(model['intercept']), -50, 50)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p[:, [model['classes'].index(c) for c in CLASSES]]


def _paired_ci(delta):
    rng = np.random.default_rng(20260704)
    # Bound temporary allocation even for very large bundled datasets.
    means = []
    chunk = max(1, min(64, 250_000 // len(delta)))
    for start in range(0, 2000, chunk):
        ix = rng.integers(0, len(delta), size=(min(chunk, 2000 - start), len(delta)))
        means.extend(delta[ix].mean(axis=1).tolist())
    lo, hi = np.quantile(means, [.025, .975])
    return {'diff': float(delta.mean()), 'lo': float(lo), 'hi': float(hi)}



def _historical_lineage(frame):
    """Compact, deterministic identity of the eligible historical training rows.

    Hash inputs include identities, availability, market features and labels, so
    corrections change the fingerprint without repeating the dataset in models.
    """
    digest = hashlib.sha256()
    if frame.empty:
        return dict(count=0, date_start=None, date_end=None,
                    sha256=digest.hexdigest(), fingerprint_version=1)
    for row in frame.sort_values(['league', 'event_id']).itertuples():
        payload = [str(row.league), str(row.event_id), row.kickoff.isoformat(),
                   row.result_observed_at.isoformat(),
                   _match_key(row.home_team, row.away_team), str(row.result),
                   float(row.p_home), float(row.p_draw), float(row.p_away)]
        digest.update(json.dumps(payload, ensure_ascii=True, separators=(',', ':'),
                                 allow_nan=False).encode('utf-8'))
        digest.update(b'\n')
    return dict(count=len(frame), date_start=frame.kickoff.min().isoformat(),
                date_end=frame.kickoff.max().isoformat(), sha256=digest.hexdigest(),
                fingerprint_version=1)


def fit_candidate(historical, live_training, as_of):
    """Validate on later dates, then refit on all eligible past matches."""
    cutoff = _utc(as_of)
    frame = _prepare(historical, live_training, cutoff)
    validation = dict(promoted=False, reason='insufficient chronological training/validation data',
                      n_train=0, n_validation=0, baseline_log_loss=None, candidate_log_loss=None,
                      baseline_brier=None, candidate_brier=None, ci=None)
    model = dict(schema_version=1, kind='market', trained_at=cutoff.isoformat(),
                 training_cutoff=cutoff.isoformat(), training_event_ids=[], event_lineage=[],
                 validation=validation, classes=CLASSES,
                 historical_lineage=_historical_lineage(pd.DataFrame()))
    if not frame.empty:
        historical_rows = frame.loc[frame.source == 'historical']
        live_rows = frame.loc[frame.source == 'live']
        model['historical_lineage'] = _historical_lineage(historical_rows)
        model['training_event_ids'] = live_rows.event_id.astype(str).tolist()
        model['event_lineage'] = [dict(event_id=str(r.event_id), league=str(r.league),
                                     source=r.source, kickoff=r.kickoff.isoformat(),
                                     result_observed_at=r.result_observed_at.isoformat())
                                  for r in live_rows.itertuples()]
        dates = frame.kickoff.dt.floor('D')
        target = max(0, int(len(frame) * .75))
        split_day = dates.iloc[min(target, len(frame) - 1)]
        train, val = frame.loc[dates < split_day], frame.loc[dates >= split_day]
        validation.update(n_train=len(train), n_validation=len(val),
                          split_date=split_day.isoformat())
        if len(train) >= 500 and len(val) >= 200:
            if set(train.result) != set(CLASSES) or set(val.result) != set(CLASSES):
                validation['reason'] = 'all H/D/A outcomes required in both chronological partitions'
            else:
                # Training labels must have been available before the holdout begins.
                train = train.loc[train.result_observed_at < split_day]
                validation['n_train'] = len(train)
                if len(train) < 500 or set(train.result) != set(CLASSES):
                    validation['reason'] = 'insufficient labels available before validation cutoff'
                else:
                    candidate = dict(kind='multinomial', **_fit(train))
                    baseline, predicted = _market(val), predict_candidate(candidate, val)
                    base_ll, cand_ll = log_loss_per_match(baseline, val.result), log_loss_per_match(predicted, val.result)
                    base_b, cand_b = brier_per_match(baseline, val.result), brier_per_match(predicted, val.result)
                    ci = _paired_ci(cand_ll - base_ll)
                    promoted = bool(ci['hi'] < 0 and cand_ll.mean() < base_ll.mean() and cand_b.mean() <= base_b.mean())
                    validation.update(promoted=promoted, reason='passed chronological paired evaluation' if promoted else
                                      'candidate did not pass paired log-loss CI and Brier safeguards',
                                      baseline_log_loss=float(base_ll.mean()), candidate_log_loss=float(cand_ll.mean()),
                                      baseline_brier=float(base_b.mean()), candidate_brier=float(cand_b.mean()), ci=ci)
                    model.update(kind='multinomial', **_fit(frame))
    digest = hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    model['version'] = 'live-recalibration-v1-' + digest
    json.dumps(model, allow_nan=False)
    return model
