from pathlib import Path
import hashlib
import json
import os
import zipfile
import numpy as np
import pandas as pd
from .artifacts import atomic_json, fingerprint
from .config import run_id, stable_hash, validation_steps, METHOD_LABELS
MAIN = ('full', 'transfer_guard_off', 'local', 'fedavg', 'fedrag', 'gpfedrec', 'fedhpd', 'fedslate', 'random')

def validate_trial_members(jobs, names):
    version = jobs[0].get('trial', {}).get('version') if jobs else None
    count = {'v19': 66, 'v20': 66, 'v21': 78, 'v22': 78, 'v23': 90}.get(version)
    expected = {run_id(c) for c in jobs}
    if count is None or len(jobs) != count or len(expected) != len(jobs) or (set(names) != expected) or any((c.get('trial', {}).get('version') != version for c in jobs)):
        raise ValueError('Incomplete or unexpected trial matrix')
    if version in ('v20', 'v21', 'v22', 'v23'):
        from .config import V20_METHODS, V21_METHODS, V23_METHODS, SEEDS
        methods = V23_METHODS if version == 'v23' else V21_METHODS if version in ('v21', 'v22') else V20_METHODS
        required = {(e, m, s) for e in ('kuaisim', 'recsim') for m in (*methods, 'random') for s in SEEDS}
        if {(c['environment'], c['method'], c['seed']) for c in jobs} != required:
            raise ValueError('Incomplete or unexpected trial matrix')

def paired_comparisons(frame, phase):
    rows = []
    for (env, method), g in frame.groupby(['environment', 'method']):
        if method in ('full', 'random'):
            continue
        budgets = g.budget.unique()
        if len(budgets) != 1:
            raise ValueError('Mixed budgets for a method')
        step = int(budgets[0] // 2 if phase == 'half' else budgets[0])
        left = frame[(frame.environment == env) & (frame.method == 'full') & (frame.step == step)].set_index('seed').J
        right = g[g.step == step].set_index('seed').J
        if set(left.index) != {2026, 2027, 2028} or set(right.index) != set(left.index) or left.index.has_duplicates or right.index.has_duplicates:
            raise ValueError('Missing equal-budget paired Full comparison')
        d = left - right
        se = float(d.std(ddof=1) / np.sqrt(3))
        rows.append(dict(environment=env, method=method, step=step, full_mean=float(left.mean()), method_mean=float(right.mean()), full_minus_method=float(d.mean()), relative_pct=100 * float(d.mean() / right.mean()) if right.mean() != 0 else None, full_wins=int((d > 0).sum()), n_seeds=3, paired_95_low=float(d.mean() - 4.30265273 * se), paired_95_high=float(d.mean() + 4.30265273 * se), interpretation='Exploratory paired t interval, n=3, no multiplicity correction.'))
    return rows

def build_report(runs, output, jobs, phase='half'):
    from .optimization_trial import check_trial_freeze
    from .staged_study import verify_pause
    from .scheduler import verify_completed_retention
    from .protocol import source_identity
    runs = Path(runs)
    output = Path(output)
    if phase not in ('half', 'finish'):
        raise ValueError('Unknown phase')
    expected = {run_id(c): c for c in jobs}
    paths = {p.parent.name: p for p in runs.glob('*/run.json')}
    validate_trial_members(jobs, paths)
    records = []
    costs = []
    guards = []
    sources = set()
    smokes = set()
    identity = {}
    common = {}
    backbones = {}
    for name, c in expected.items():
        path = paths[name]
        meta = json.loads(path.read_text(encoding='utf-8'))
        cfg = meta['config']
        sources.add(meta['source_hash'])
        smokes.add(meta['smoke'])
        if stable_hash(cfg) != meta['configuration_hash']:
            raise ValueError('Run configuration hash differs')
        check_trial_freeze(cfg)
        if deep_without_assets(cfg) != deep_without_assets(c):
            raise ValueError('Run differs from planned matrix')
        identity[name] = meta['configuration_hash']
        env, method = (cfg['environment'], cfg['method'])
        budget = cfg['training']['requests_per_client']
        shared = (meta['task_hash'], meta['evaluation_hash'], meta['backbone_hash'], meta['protocol_hash'])
        if cfg.get('trial', {}).get('version') in ('v20', 'v21', 'v22', 'v23'):
            from .config import backbone_hash
            if meta['backbone_hash'] != backbone_hash(cfg):
                raise ValueError('Incorrect capacity identity')
            shared = (meta['task_hash'], meta['evaluation_hash'], meta['protocol_hash'])
            if method not in ('random', 'local_fixed', 'v14_reference'):
                if env in backbones and backbones[env] != meta['backbone_hash']:
                    raise ValueError('Main scorer capacity differs')
                backbones[env] = meta['backbone_hash']
        if env in common and common[env] != shared:
            raise ValueError('Shared comparison protocol differs')
        common[env] = shared
        step = budget // 2 if phase == 'half' else budget
        if method == 'random' or phase == 'finish':
            verify_completed_retention(path.parent, meta['configuration_hash'])
        else:
            verify_pause(path.parent, step)
        if method != 'random' and meta['requests_per_client'] != step:
            raise ValueError('Wrong phase endpoint')
        rows = [json.loads(x) for x in (path.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        val = [r for r in rows if r['split'] == 'validation']
        needed = {0} if method == 'random' else {s for s in validation_steps(cfg) if s <= step}
        if len(val) != len(needed) or {r['requests_per_client'] for r in val} != needed:
            raise ValueError('Incomplete fixed-sample validation trajectory')
        for r in val:
            if r['eval_sessions'] != cfg['evaluation']['validation_sessions']:
                raise ValueError('Validation count differs')
            if not cfg['smoke'] and r['eval_sessions'] != 500:
                raise ValueError('Formal trial needs 500 validation sessions')
            if len(r['clients']) != cfg['env']['num_clients']:
                raise ValueError('Missing client metrics')
            for metric in ('J', 'A', 'L'):
                if not np.isclose(np.mean([x[metric] for x in r['clients']]), r['macro'][metric], rtol=1e-10, atol=1e-10):
                    raise ValueError('Macro differs from clients')
            records.append(dict(environment=env, method=method, seed=cfg['seed'], budget=budget, step=r['requests_per_client'], **r['macro']))
        costs.append(dict(environment=env, method=method, seed=cfg['seed'], requests=0 if method == 'random' else step, training_seconds=meta['timing'].get('training_elapsed_seconds', 0.0), evaluation_seconds=meta['timing'].get('evaluation_seconds', 0.0), communication_bytes=meta.get('communication_bytes', 0), interpretation='Observed trial costs; see costs_by_seed.csv for precommitted rounds' if cfg['cost'].get('require_calibrated_target') else 'Observed trial costs; no calibrated time-to-target claim'))
        rounds = path.parent / 'rounds.jsonl'
        if rounds.exists():
            for line in rounds.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                for client, d in enumerate(r.get('distillation', [])):
                    g = d.get('transfer_guard')
                    if g is not None:
                        guards.append(dict(run=name, environment=env, method=method, seed=cfg['seed'], step=r['requests_per_client'], client=client, accepted_scale=g['accepted_scale'], reason=g['reason'], proposal_steps=d['distillation_update_count'], seconds=g['seconds'], retained_replay_kl=g.get('retained_replay_kl', 0.0), attempts=json.dumps(g['attempts'])))
    if sources != {source_identity()} or len(smokes) != 1:
        raise ValueError('Mixed source or smoke status')
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'validation_by_seed.csv', index=False)
    selected = frame[(frame.step == frame.budget // 2 if phase == 'half' else frame.step == frame.budget) | (frame.method == 'random')]
    summary = selected.groupby(['environment', 'method', 'step']).agg(J_mean=('J', 'mean'), J_sd=('J', 'std'), A_mean=('A', 'mean'), L_mean=('L', 'mean'), n_seeds=('seed', 'count')).reset_index()
    summary.to_csv(output / 'endpoint_results.csv', index=False)
    pairs = pd.DataFrame(paired_comparisons(frame, phase))
    pairs.to_csv(output / 'paired_comparisons.csv', index=False)
    pd.DataFrame(costs).to_csv(output / 'observed_costs.csv', index=False)
    pd.DataFrame(guards).to_csv(output / 'guard_by_round.csv', index=False)
    guard_summary = []
    if guards:
        for (env, method), g in pd.DataFrame(guards).groupby(['environment', 'method']):
            guard_summary.append(dict(environment=env, method=method, calls=len(g), rejection_fraction=float((g.accepted_scale == 0).mean()), mean_accepted_scale=float(g.accepted_scale.mean()), full_step_fraction=float((g.accepted_scale == 1).mean()), mean_replay_kl=float(g.retained_replay_kl.mean()), seconds=float(g.seconds.sum())))
    atomic_json(output / 'guard_summary.json', guard_summary)
    status = 'NONPUBLICATION SMOKE' if True in smokes else 'EXPLORATORY HALF' if phase == 'half' else 'COMPLETED OPTIMIZATION TRIAL'
    report = dict(version=jobs[0]['trial']['version'], status=status, phase=phase, runs=len(jobs), source_hash=source_identity(), identities=identity, primary_split='validation', publication_evidence=False if True in smokes or phase == 'half' else 'requires final scientific review', guard=guard_summary, interpretation='All methods retained; ablations paired at equal budgets. Half ranking is not a final/SOTA claim. No calibrated J* cost claim.')
    if jobs[0]['cost'].get('require_calibrated_target'):
        report.update(precommitted_targets=True, interpretation='All methods retained at equal budgets. Independent validation targets frozen before comparison; calibrated rounds in costs_by_seed.csv. Timing remains observed cost.')
    atomic_json(output / 'stage-results.json', report)
    lines = ['# ' + jobs[0]['trial']['version'] + ' ' + status, '', 'Primary split: fixed-sample validation. Mean +/- sample SD over three training seeds.', '', '|Environment|Method|Requests/client|J mean|SD|', '|---|---|---:|---:|---:|']
    for r in summary.to_dict('records'):
        labels = METHOD_LABELS
        if jobs[0]['trial']['version'] == 'v23':
            from .ablation_reporting import LABELS
            labels = LABELS
        lines.append(f"|{r['environment']}|{labels[r['method']]}|{r['step']}|{r['J_mean']:.5f}|{r['J_sd']:.5f}|")
    budget_note = 'All learned conditions use equal budgets within each environment.' if jobs[0]['trial']['version'] in ('v20', 'v21', 'v22', 'v23') else 'Endpoint rows have different budgets for main methods and ablations. Use paired_comparisons.csv for equal-budget comparisons.'
    lines += ['', budget_note, 'Random uses no training (step 0). Test trajectories are archived separately and do not control training or promotion.', 'Do not treat exploratory n=3 intervals as proof of superiority or equivalence. Check guard rejection and retained KL before attributing changes.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if jobs[0]['trial']['version'] in ('v20', 'v21', 'v22', 'v23'):
        from .v20_reporting import causal_report
        causal_report(frame, runs, output, phase, version=jobs[0]['trial']['version'])
    return report

def deep_without_assets(c):
    from copy import deepcopy
    value = deepcopy(c)
    value.pop('assets', None)
    return value

def archive_evidence(runs, output, names):
    runs = Path(runs).resolve()
    output = Path(output)
    files = {}
    for name in names:
        directory = (runs / name).resolve()
        if not directory.is_relative_to(runs):
            raise ValueError('Unsafe run name')
        for p in sorted(directory.rglob('*')):
            if p.is_file() and (not p.is_symlink()) and (p.suffix in ('.json', '.jsonl') or p.parent.name in ('source', 'checkpoints', 'attempts')):
                files[p.relative_to(runs).as_posix()] = fingerprint(p)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.zip.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in files:
            z.write(runs / name, name)
        z.writestr('manifest.json', json.dumps(dict(version=json.loads((runs / names[0] / 'run.json').read_text(encoding='utf-8')).get('config', {}).get('trial', {}).get('version', 'v19'), files=files, interpretation='Resume/replay retained on server; compact checkpoints and raw evidence included.'), indent=2))
    verify_archive(pending)
    os.replace(pending, output)
    atomic_json(output.with_suffix('.receipt.json'), dict(sha256=fingerprint(output), files=len(files), bytes=output.stat().st_size))

def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read('manifest.json'))
        if set(z.namelist()) != set(m['files']) | {'manifest.json'}:
            raise ValueError('Archive members differ')
        for name, digest in m['files'].items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != digest:
                raise ValueError('Archive checksum failed: ' + name)
    return m
