from __future__ import annotations
import json
from pathlib import Path
import numpy as np
SEEDS = (2026, 2027, 2028)
MAIN = ('local', 'fedavg', 'full')

def rounds_to_target(run, records):
    target = run.get('cost_target') or {}
    jstar = target.get('J_star')
    result = dict(J_star=jstar, first_hit_requests=None, rounds_to_target=None, status='N/A' if run['method'] == 'local' else 'NR')
    if run['method'] == 'local':
        return result
    if jstar is None or not np.isfinite(jstar):
        result['status'] = 'MISSING TARGET'
        return result
    for row in sorted(records, key=lambda r: r['requests_per_client']):
        if row.get('split') != 'validation' or row['requests_per_client'] > run['budget_per_client']:
            continue
        macro = row.get('macro', row.get('macros', {}))
        value = row.get('J_macro', macro.get('J'))
        if value is not None and np.isfinite(value) and (value >= jstar):
            result.update(status='reached', first_hit_requests=row['requests_per_client'], rounds_to_target=int(row['scheduled_round_index']))
            break
    return result

def rounds_summary(rows):
    by_seed = {r['seed']: r for r in rows}
    if set(by_seed) != set(SEEDS):
        return 'INCOMPLETE'
    ordered = [by_seed[s] for s in SEEDS]
    if all((r['status'] == 'N/A' for r in ordered)):
        return 'N/A'
    reached = [r['rounds_to_target'] for r in ordered if r['status'] == 'reached']
    if len(reached) == 3:
        return f'{np.mean(reached):.2f} ± {np.std(reached, ddof=1):.2f}'
    return ' / '.join((str(r['rounds_to_target']) if r['status'] == 'reached' else r['status'] for r in ordered)) + f' ({len(reached)}/3 reached)'

def export_costs(runs, output, methods=None):
    import pandas as pd
    from .reporting import _jsonl, declared_baselines
    from fedlpd.config import METHOD_LABELS
    from fedlpd.artifacts import fingerprint
    output = Path(output)
    costs, communication, validation, targets, issues, sources = ([], [], [], [], [], {})
    main = tuple(methods) if methods is not None else MAIN + declared_baselines(runs)
    deferred = ['gpfedrec'] if methods is None and 'gpfedrec' not in main else []
    performances = {(r.get('run_id'), r.get('attempt_id')): r for r in runs if r.get('purpose') == 'formal'}
    for run in runs:
        if run.get('purpose') != 'cost_remeasure':
            continue
        reference = performances.get((run.get('performance_run_id'), run.get('performance_attempt_id')))
        if reference is None or not run.get('performance_run_id') or (not run.get('performance_attempt_id')):
            raise ValueError('Cost remeasurement must name an available original performance run and attempt')
        for field in ('environment', 'method', 'seed', 'budget_per_client', 'task_hash', 'method_config_hash', 'training_protocol_hash', 'backbone_hash', 'source_hash', 'evaluation_hash'):
            if run.get(field) is None or run[field] != reference.get(field):
                raise ValueError('Incompatible cost remeasurement ' + field)
    selected = {}
    for run in runs:
        if run['method'] not in main:
            continue
        key = (run['environment'], run['method'], run['seed'])
        if key in selected:
            previous = selected[key]
            if previous.get('cost_eligible') and run.get('cost_eligible'):
                raise ValueError(f'Duplicate eligible cost source {key}')
            if previous.get('cost_eligible'):
                continue
        selected[key] = run
    for key, run in sorted(selected.items()):
        env, method, seed = key
        path = Path(run['_path'])
        identity = dict(environment=env, method=method, seed=seed)
        reasons = list(run.get('cost_ineligible_reasons', []))
        eligible = bool(run.get('cost_eligible', False)) and (not run.get('smoke', False))
        if not run.get('cost_eligible', False):
            reasons.append('cost_eligible is not true')
        timing = run.get('timing', {})
        elapsed = timing.get('training_elapsed_seconds', run.get('training_elapsed_seconds'))
        if elapsed is None or not np.isfinite(elapsed) or elapsed < 0:
            reasons.append('missing valid cumulative training elapsed time')
        ledger = _jsonl(path / 'message_ledger.jsonl')
        unique = {}
        for row in ledger:
            eid = row.get('event_id')
            if not eid:
                raise ValueError(f'Missing communication event_id in {path}')
            if eid in unique and unique[eid] != row:
                raise ValueError(f'Conflicting duplicate communication event {eid}')
            unique[eid] = row
        total = sum((int(r['payload_nbytes']) for r in unique.values())) if unique else None
        if method == 'local':
            if total not in (None, 0) or run.get('communication_bytes', 0) != 0:
                raise ValueError('Local RL must have exactly zero communication bytes')
            total = 0
        elif total is None:
            reasons.append('missing federation message ledger')
        if total is not None and run.get('communication_bytes') not in (None, total):
            reasons.append('communication total differs from event ledger')
        for row in unique.values():
            communication.append(dict(**identity, direction=row['direction'], message_type=row.get('message_type', row.get('type')), round_index=row['round_index'], request_step=row['request_step'], sender=row.get('sender'), receiver=row.get('receiver'), initialization=row['round_index'] == 0, payload_nbytes=row['payload_nbytes'], event_id=row['event_id']))
        records = [r for r in _jsonl(path / 'metrics.jsonl') if r.get('split') == 'validation']
        expected_budget = 80000 if env == 'kuaisim' else 40000
        if run['budget_per_client'] != expected_budget:
            reasons.append('not a complete primary-method budget')
        warmup = int(run.get('config', {}).get('training', {}).get('warmup', 8000))
        every = int(run.get('config', {}).get('training', {}).get('communication_every', 2000))
        required = {0, warmup, *range(warmup + every, int(run['budget_per_client']) + 1, every)}
        if {r['requests_per_client'] for r in records} != required:
            reasons.append('missing or unexpected validation checkpoints')
        for r in records:
            macro = r.get('macro', r.get('macros', {}))
            validation.append({**r, **identity, 'J_macro': r.get('J_macro', macro.get('J'))})
        target = rounds_to_target(run, records)
        if target['status'] == 'MISSING TARGET':
            reasons.append('target was not frozen')
        if not run.get('cost_protocol_hash'):
            reasons.append('missing cost protocol identity')
        if reasons:
            eligible = False
            issues.extend((f'{env}/{method}/{seed}: {reason}' for reason in reasons))
        source = str(path)
        hashes = {name: fingerprint(path / name) for name in ('run.json', 'metrics.jsonl', 'message_ledger.jsonl', 'evaluation_sessions.jsonl') if (path / name).is_file()}
        sources['/'.join(map(str, key))] = dict(run=source, files=hashes, target=run.get('cost_target'), cost_protocol_hash=run.get('cost_protocol_hash'))
        targets.append(dict(**identity, **target, cost_eligible=eligible, source_run=source, cost_protocol_hash=run.get('cost_protocol_hash'), evidence_hash=hashes.get('metrics.jsonl')))
        costs.append(dict(**identity, budget=run['budget_per_client'], cost_eligible=eligible, ineligible_reasons='; '.join(reasons), training_elapsed_seconds=elapsed, training_minutes=elapsed / 60 if elapsed is not None else np.nan, communication_bytes=total, communication_GB=total / 1000000000.0 if total is not None else np.nan, source_run=source, cost_protocol_hash=run.get('cost_protocol_hash'), hardware_hash=run.get('hardware_hash'), **{k: v for k, v in timing.items() if k != 'training_elapsed_seconds'}))
    expected = {(e, m, s) for e in ('kuaisim', 'recsim') for m in main for s in SEEDS}
    for key in sorted(expected - set(selected)):
        issues.append('Missing cost source: ' + str(key))
    for env in ('kuaisim', 'recsim'):
        protocols = {r.get('cost_protocol_hash') for r in selected.values() if r['environment'] == env and r.get('cost_eligible')}
        if len(protocols) > 1:
            raise ValueError(f'Incompatible cost protocols in {env}')
        frozen_targets = {json.dumps(r.get('cost_target'), sort_keys=True) for r in selected.values() if r['environment'] == env and r.get('cost_eligible')}
        if len(frozen_targets) > 1:
            raise ValueError(f'Incompatible frozen cost targets in {env}')
    for name, rows in [('costs_by_run', costs), ('communication_by_type', communication), ('validation_curves', validation), ('rounds_to_target_by_seed', targets)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False)
    for name, field in [('training_time', 'training_minutes'), ('communication_volume', 'communication_GB'), ('rounds_to_target', None)]:
        table = []
        for method in (*main, *deferred):
            row = {'method': METHOD_LABELS.get(method, method)}
            for env in ('kuaisim', 'recsim'):
                if method in deferred:
                    row[env] = 'Deferred'
                    continue
                if method == 'local' and field is None:
                    row[env] = 'N/A'
                    continue
                chosen = [r for r in (targets if field is None else costs) if r['method'] == method and r['environment'] == env and r['cost_eligible']]
                if {r['seed'] for r in chosen} != set(SEEDS):
                    row[env] = 'INCOMPLETE'
                elif field is None:
                    row[env] = rounds_summary(chosen)
                else:
                    values = [r[field] for r in chosen]
                    row[env] = f'{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}'
            table.append(row)
        frame = pd.DataFrame(table)
        frame.to_csv(output / (name + '.csv'), index=False)
        (output / (name + '.tex')).write_text(frame.to_latex(index=False), encoding='utf-8')
    manifest = dict(status='incomplete' if issues or any((r.get('smoke') for r in runs)) else 'complete', expected_sources=6 * len(main), eligible_sources=sum((r['cost_eligible'] for r in costs)), issues=issues, units={'training_time': 'minutes', 'communication_volume': 'GB = 1e9 bytes', 'rounds_to_target': 'scheduled coordination index'}, sources=sources, deferred=deferred)
    (output / 'cost_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
