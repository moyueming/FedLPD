from pathlib import Path
import json
import os
import shutil
import tarfile
import zipfile
import numpy as np
import pandas as pd
from .artifacts import atomic_json, fingerprint
from .staged_study import pause_points, verify_pause

def build_report(runs, output, phase):
    from .scheduler import verify_completed_retention
    from .task_study import check_study_freeze
    if phase not in ('half', 'finish'):
        raise ValueError('Unknown study phase')
    paths = sorted(Path(runs).glob('*/run.json'))
    output = Path(output)
    if len(paths) != 18:
        raise ValueError('Staged report requires all 18 matched runs')
    expected = {(e, m, s) for e in ('kuaisim', 'recsim') for m in ('local', 'full', 'central_shared') for s in (2026, 2027, 2028)}
    seen = set()
    identities = {}
    smokes = set()
    endpoints = []
    clients = []
    curves = []
    sampling = []
    for path in paths:
        meta = json.loads(path.read_text(encoding='utf-8'))
        c = meta['config']
        key = (meta['environment'], meta['method'], meta['seed'])
        if key not in expected or key in seen or c['study']['scenario'] != 'complementary':
            raise ValueError('Unmatched staged scope')
        seen.add(key)
        smokes.add(meta['smoke'])
        if not meta['smoke']:
            check_study_freeze(c)
        identity = tuple((meta[n] for n in ('task_hash', 'evaluation_hash', 'training_protocol_hash', 'backbone_hash', 'source_hash', 'budget_per_client')))
        if key[0] in identities and identities[key[0]] != identity:
            raise ValueError('Unmatched methods, tasks, budgets or sources')
        identities[key[0]] = identity
        if phase == 'half':
            step = pause_points([c])[meta['run_id']]
            verify_pause(path.parent, step)
        else:
            if meta['status'] != 'completed':
                raise ValueError('Finish report requires completed runs')
            verify_completed_retention(path.parent, meta['configuration_hash'])
            step = meta['budget_per_client']
        metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        validation = sorted([m for m in metrics if m['split'] == 'validation'], key=lambda m: m['requests_per_client'])
        if any((m['eval_sessions'] != c['evaluation']['validation_sessions'] for m in validation)):
            raise ValueError('Unmatched validation session count')
        target = [m for m in metrics if m['requests_per_client'] == step and (m['split'] == 'validation' if phase == 'half' else m['split'] == 'test' and m['eval_kind'] == 'final')]
        sessions = c['evaluation']['validation_sessions' if phase == 'half' else 'final_sessions']
        if len(target) != 1 or target[0]['eval_sessions'] != sessions:
            raise ValueError('Missing matched endpoint')
        base = dict(environment=key[0], method=key[1], seed=key[2])
        m = target[0]
        recent = validation[-3:]
        endpoints.append(dict(**base, split=m['split'], requests=step, sessions=sessions, **m['macro'], recent3_validation_J=float(np.mean([v['macro']['J'] for v in recent])), recent3_requests=','.join((str(v['requests_per_client']) for v in recent)), training_seconds=meta['timing']['training_elapsed_seconds'], actual_rl_updates=meta['centralized_global_updates'] if key[1] == 'central_shared' else sum(meta['rl_updates_by_client']), distillation_updates=meta['distillation_update_count']))
        clients.extend((dict(**base, **v) for v in m['clients']))
        curves.extend((dict(**base, requests=v['requests_per_client'], sessions=v['eval_sessions'], **v['macro']) for v in validation))
        sampling.append(dict(**base, audit=json.loads((path.parent / 'task-sampling.json').read_text(encoding='utf-8'))))
    if seen != expected or len(smokes) != 1:
        raise ValueError('Incomplete staged scope or mixed smoke/formal results')
    endpoints = pd.DataFrame(endpoints)
    clients = pd.DataFrame(clients)
    curves = pd.DataFrame(curves)
    summary = []
    pairs = []
    client_pairs = []
    for (env, method), g in endpoints.groupby(['environment', 'method']):
        r = dict(environment=env, method=method)
        for metric in ('J', 'A', 'L', 'recent3_validation_J', 'training_seconds', 'actual_rl_updates'):
            r[metric + '_mean'] = float(g[metric].mean())
            r[metric + '_sd'] = float(g[metric].std(ddof=1))
        summary.append(r)
    for env in ('kuaisim', 'recsim'):
        g = endpoints[endpoints.environment == env]
        b = g[g.method == 'local'].set_index('seed')
        for method in ('full', 'central_shared'):
            a = g[g.method == method].set_index('seed')
            record = dict(environment=env, method=method, reference='local')
            for metric in ('J', 'A', 'L', 'recent3_validation_J'):
                delta = (a[metric] - b[metric]).sort_index()
                margin = 4.302652729911275 * float(delta.std(ddof=1)) / np.sqrt(3)
                record[metric] = dict(mean_delta=float(delta.mean()), seed_deltas=delta.to_dict(), wins=int((delta > 0).sum()), ci95_low=float(delta.mean() - margin), ci95_high=float(delta.mean() + margin))
            pairs.append(record)
            for k in range(5):
                cg = clients[(clients.environment == env) & (clients.client_id == k)]
                d = cg[cg.method == method].set_index('seed').J - cg[cg.method == 'local'].set_index('seed').J
                client_pairs.append(dict(environment=env, method=method, client_id=k, J_delta=float(d.mean()), wins=int((d > 0).sum())))
    output.mkdir(parents=True, exist_ok=True)
    for name, df in [('endpoints', endpoints), ('clients', clients), ('validation', curves), ('summary', pd.DataFrame(summary)), ('client-pairs', pd.DataFrame(client_pairs))]:
        df.to_csv(output / (name + '.csv'), index=False)
    result = dict(version='v17.1', phase=phase, runs=18, smoke=smokes.pop(), decision_split='validation', endpoint_split='validation' if phase == 'half' else 'test', summary=summary, paired_comparisons=pairs, identities=identities, sampling=sampling, limitations=['Interim validation is exploratory resource allocation, not a final-budget performance claim.', 'Recent-three-point differences are descriptive; repeated validation points are not independent seeds.', 'CentralShared has privileged shared heads and is neither a federated baseline nor an upper bound.', 'Only complementary scenario and three methods; not the complete paper benchmark or ablation matrix.', 'Three paired seeds; exploratory t intervals, no multiplicity correction. No automatic promotion.'])
    atomic_json(output / 'stage-results.json', result)
    lines = [f'# v17.1 {phase} report', '', f"Software smoke: {result['smoke']}. Endpoint split: {result['endpoint_split']}.", '', '| Environment | Method | J mean ± SD | Recent 3 validation J |', '|---|---|---|---|']
    lines += [f"| {r['environment']} | {r['method']} | {r['J_mean']:.6f} ± {r['J_sd']:.6f} | {r['recent3_validation_J_mean']:.6f} |" for r in summary]
    lines += ['', '| Environment | Comparison | Paired J delta | Seeds won | Recent 3 validation delta |', '|---|---|---|---|---|']
    lines += [f"| {r['environment']} | {r['method']} - Local | {r['J']['mean_delta']:+.6f} | {r['J']['wins']}/3 | {r['recent3_validation_J']['mean_delta']:+.6f} |" for r in pairs]
    lines += [''] + ['- ' + s for s in result['limitations']]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result

def finalize(root, runs, phase):
    from .full_workflow import archive_analysis, _verify_analysis
    root = Path(root)
    runs = Path(runs)
    out = root / phase
    out.mkdir(parents=True, exist_ok=True)
    build_report(runs, out / 'report', phase)
    if phase == 'finish':
        archive_analysis(runs, out / 'analysis-evidence.zip', expected_runs=18)
    else:
        files = {}
        archive = out / 'analysis-evidence.zip'
        pending = archive.with_suffix('.zip.pending')
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for directory in sorted((p.parent for p in runs.glob('*/run.json'))):
                receipt = json.loads((directory / 'pause-retention.json').read_text(encoding='utf-8'))
                for name in sorted(set(receipt['files']) - {'resume.pt'} | {'pause-retention.json'}):
                    path = directory / name
                    member = path.relative_to(runs).as_posix()
                    files[member] = fingerprint(path)
                    z.write(path, member)
            manifest = dict(version='v17.1', phase='half', runs=18, files=files, resume_checkpoints='Retained on the server; hashes in pause-retention.json. Deliberately excluded from evidence export.')
            z.writestr('analysis_manifest.json', json.dumps(manifest, indent=2))
        _verify_analysis(pending)
        os.replace(pending, archive)
    for name in ('study-freeze.json', 'stage-policy.json', 'matrix-task-study.json', 'training-coverage.json', 'workflow_manifest.json', 'workflow-timing.json'):
        if (root / name).is_file():
            shutil.copy2(root / name, out / name)
    if (root / 'audits').exists():
        shutil.copytree(root / 'audits', out / 'audits', dirs_exist_ok=True)
    return out

def export(root, phase):
    root = Path(root)
    if (root / 'workflow-timing.json').exists():
        shutil.copy2(root / 'workflow-timing.json', root / phase / 'workflow-timing.json')
    path = root / f'{phase}-v17-1-task-study.tar.gz'
    pending = path.with_suffix('.gz.pending')
    with tarfile.open(pending, 'w:gz', compresslevel=1) as z:
        z.add(root / phase, arcname=phase)
    os.replace(pending, path)
    path.with_name(path.name + '.sha256').write_bytes((fingerprint(path) + '  ' + path.name + '\n').encode('ascii'))
    print('Return this file: ' + str(path.resolve()), flush=True)
    return path
