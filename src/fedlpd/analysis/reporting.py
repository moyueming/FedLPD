from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from fedlpd.config import ABLATION_GROUPS, ABLATIONS as ABLATION_METHODS, METHOD_LABELS, METHODS
SEEDS = {2026, 2027, 2028}
ABLATIONS = ABLATION_GROUPS
COLORS = dict(zip(METHODS, ('gray', '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#56B4E9', '#8c564b', '#9467bd', '#e377c2', '#7f7f7f')))
COLORS['donor_filter_off'] = '#6a51a3'
COLORS.update(fedrag='#332288', gpfedrec='#882255')

def declared_baselines(runs):
    return tuple((m for m in ('fedrag', 'gpfedrec') if any((r['method'] == m or m in r.get('config', {}).get('comparison', {}).get('baselines', []) for r in runs))))

def summarize_values(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Cannot summarize missing or nonfinite observations')
    return (float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else float('nan'))

def validate_runs(runs, allow_incomplete=False):
    issues, groups, env_hashes = ([], {}, {})
    if not runs:
        raise ValueError('No runs found')
    if len({bool(r.get('smoke', False)) for r in runs}) != 1:
        raise ValueError('Cannot mix smoke and formal runs')
    for r in runs:
        validate_publication_eligibility(r)
        if r['method'] not in METHODS:
            raise ValueError('Method is outside the v2.0 registry: ' + r['method'])
        fields = ('task_hash', 'evaluation_hash', 'protocol_hash') if r.get('smoke') else ('task_hash', 'evaluation_hash', 'training_protocol_hash', 'source_hash', 'backbone_hash')
        for field in fields:
            value = r.get(field)
            key = (r['environment'], field)
            if not value:
                raise ValueError(f'Missing {field} in run')
            if key in env_hashes and env_hashes[key] != value:
                raise ValueError(f"Incompatible {field} in {r['environment']}")
            env_hashes[key] = value
        key = (r['environment'], r['method'])
        group = groups.setdefault(key, [])
        if r['seed'] in group:
            raise ValueError(f"Duplicate run for {key}, seed {r['seed']}")
        group.append(r['seed'])
        if r.get('status') != 'completed':
            issues.append(f"Unfinished run {key}, seed {r['seed']}")
    for key, seeds in groups.items():
        if set(seeds) != SEEDS:
            issues.append(f'{key}: expected seeds {sorted(SEEDS)}, got {sorted(seeds)}')
    if issues and (not allow_incomplete):
        raise ValueError('; '.join(issues))
    return issues

def paired_gains(records, method='full', reference='local', budget=40000):
    selected = {}
    for r in records:
        if r['requests_per_client'] == budget and r['method'] in (method, reference):
            key = (r['environment'], r['client_id'], r['seed'], r['method'])
            if key in selected:
                raise ValueError(f'Duplicate paired-gain record {key}')
            selected[key] = float(r['J'])
    groups = {}
    for (env, client, seed, name), value in selected.items():
        if name != method:
            continue
        refkey = (env, client, seed, reference)
        if refkey not in selected:
            continue
        groups.setdefault((env, client), []).append((seed, value - selected[refkey]))
    out = []
    for (env, client), vals in sorted(groups.items()):
        mean, sd = summarize_values([v for _, v in vals])
        out.append(dict(environment=env, method=method, reference=reference, client_id=client, budget=budget, mean_gain=mean, sd_gain=sd, n_seeds=len(vals), seeds=','.join((str(s) for s, _ in sorted(vals)))))
    return out

def teacher_matrix(rounds, budget=40000, last_rounds=5):
    if budget >= 40000:
        window = set(range(budget - (last_rounds - 1) * 2000, budget + 1, 2000))
        rounds = [r for r in rounds if r.get('requests_per_client') in window]
    else:
        rounds = [r for r in rounds if r.get('requests_per_client', 0) <= budget][-last_rounds:]
    shapes = [np.asarray(r['teacher_weights']).shape for r in rounds if 'teacher_weights' in r and np.asarray(r['teacher_weights']).ndim >= 2]
    if not shapes:
        return (np.empty((0, 0)), np.empty(0, dtype=int))
    n = shapes[0][0]
    total, count = (np.zeros((n, n)), np.zeros(n, dtype=int))
    for row in rounds:
        if 'teacher_weights' not in row:
            continue
        w = np.asarray(row['teacher_weights'], dtype=float)
        if w.size == 0:
            continue
        if w.ndim == 2:
            w = w[:, :, None]
        if w.shape[:2] != (n, n):
            raise ValueError('Teacher weights must have shape [target, source, probe]')
        confidence = np.asarray(row.get('probe_confidence', np.ones((n, w.shape[2]))), dtype=float)
        if confidence.shape != (n, w.shape[2]):
            raise ValueError('Probe confidence shape differs from teacher weights')
        for k in range(n):
            valid = (confidence[k] > 0) & np.isfinite(w[k]).all(axis=0) & (w[k].sum(axis=0) > 0)
            if valid.any():
                total[k] += w[k][:, valid].mean(axis=1)
                count[k] += 1
    result = np.full((n, n), np.nan)
    result[count > 0] = total[count > 0] / count[count > 0, None]
    np.fill_diagonal(result, np.nan)
    return (result, count)

def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def _record_rows(run):
    records = {}
    priorities = {'curve': 0, 'comparison': 1, 'final': 2}
    for record in _jsonl(run['_path'] / 'metrics.jsonl'):
        if record.get('split', 'test') != 'test':
            continue
        if record.get('phase', 'evaluation') in ('train', 'training'):
            continue
        request = int(record['requests_per_client'])
        old = records.get(request)
        if old is None or priorities.get(record.get('eval_kind'), 0) >= priorities.get(old.get('eval_kind'), 0):
            records[request] = record
    rows = []
    for request, record in sorted(records.items()):
        clients = record.get('clients', [])
        if not clients:
            continue
        ids = [c.get('client_id', c.get('id', i)) for i, c in enumerate(clients)]
        expected = int(run.get('config', {}).get('env', {}).get('num_clients', len(clients)))
        if len(set(ids)) != len(ids) or len(ids) != expected:
            raise ValueError(f"Missing/duplicate client evaluations in {run['_path']}")
        for i, c in enumerate(clients):
            vals = {metric: float(c[metric]) for metric in ('J', 'A', 'L')}
            if not np.isfinite(list(vals.values())).all():
                raise ValueError('Nonfinite evaluation metrics')
            rows.append(dict(environment=run['environment'], method=run['method'], seed=run['seed'], requests_per_client=request, eval_kind=record.get('eval_kind', 'curve'), eval_sessions=record.get('eval_sessions', 0), client_id=ids[i], **vals))
    return rows

def comparison_issues(records):
    issues = []
    methods = {'full', 'local'} | {m for group in ABLATIONS.values() for m in group}
    for method in methods:
        for seed in sorted(SEEDS):
            selected = [r for r in records if r['environment'] == 'kuaisim' and r['method'] == method and (r['seed'] == seed) and (r['requests_per_client'] == 40000)]
            if not selected:
                issues.append(f'Missing kuaisim {method} seed {seed} comparison at 40000 requests')
            elif any((r['eval_sessions'] < 500 or r['eval_kind'] not in ('comparison', 'final') for r in selected)):
                issues.append(f'kuaisim {method} seed {seed} 40000 comparison requires 500 sessions/client')
    return issues

def validate_publication_eligibility(run):
    if run.get('smoke', False):
        return
    purpose = run.get('purpose', run.get('config', {}).get('purpose', 'formal'))
    if purpose != 'formal' or run.get('publication_eligible', True) is False:
        raise ValueError('Run is not publication eligible: ' + str(purpose))
    if run.get('design_version') != '2.0':
        raise ValueError('Formal report requires v2.0 run metadata')
    if not run.get('performance_eligible', False):
        raise ValueError('Run is not publication eligible: performance_eligible is not true')
