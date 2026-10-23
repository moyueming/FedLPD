from copy import deepcopy
from pathlib import Path
import json
import zipfile
import numpy as np
from .artifacts import atomic_json, fingerprint, LogBook
from .config import build_matrix, run_id, stable_hash, validation_steps, scheduled_round
from .protocol import calibration_target, validate_target

def pilot_config(config):
    c = deepcopy(config)
    c.pop('trial', None)
    c.update(method='fedavg', purpose='calibration', seed=c['cost']['pilot_seed'])
    c['federation']['probe_selected'] = 4 if c['smoke'] else 16
    c['cost'].update(require_calibrated_target=False, targets_path=None, freeze_path=None)
    return c

def load_trial_target(config):
    if not config['cost'].get('require_calibrated_target'):
        return None
    path = config['cost'].get('targets_path')
    if not path or not Path(path).is_file():
        raise ValueError('Missing required precommitted target file')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    target = data.get('targets', {}).get(config['environment'])
    if target is None:
        raise ValueError('Missing environment target')
    validate_target(target, pilot_config(config))
    return target

def calibrate_trial(base, root, devices, data=None, response=None, affinities=None):
    import torch
    from .scheduler import run_matrix
    from .training import _assets, load_agents_from_checkpoint
    from .evaluation import evaluate, initialization_manifest
    from .calibration_evidence import validate_calibration_evidence
    torch.set_num_threads(base['training']['cpu_threads'])
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = build_matrix(base, True)
    pilots = [pilot_config(next((c for c in jobs if c['environment'] == env and c['method'] == 'fedavg'))) for env in ('kuaisim', 'recsim')]
    path = root / 'cost_targets.json'
    if path.exists():
        for c in pilots:
            checked = _assets(c, data, response)
            checked['cost'].update(require_calibrated_target=True, targets_path=str(path))
            load_trial_target(checked)
        print('Verified existing independent calibration targets', flush=True)
        return path
    result = run_matrix(pilots, root / 'runs', devices, data, response, cpu_affinities=affinities)
    if result['failed']:
        raise RuntimeError('Independent calibration failed; inspect calibration/runs/*/worker.log')
    targets = {}
    for c in pilots:
        run = root / 'runs' / run_id(c)
        budget = c['training']['requests_per_client']
        device = devices[0]
        agents, loaded = load_agents_from_checkpoint(run, budget, device)
        manifest = initialization_manifest(loaded, 'validation', c['evaluation']['calibration_sessions'])
        directory = root / (c['environment'] + '-validation')
        directory.mkdir(exist_ok=True)
        for name in ('metrics.jsonl', 'evaluation_sessions.jsonl'):
            (directory / name).write_text('', encoding='utf-8')
        with LogBook(directory) as logs:
            ref = evaluate(agents, loaded, budget, 'calibration', data, response, device, logs, split='validation', manifest=manifest, sessions=c['evaluation']['calibration_sessions'])
            random = deepcopy(loaded)
            random['method'] = 'random'
            rnd = evaluate([], random, 0, 'calibration', data, response, device, logs, split='validation', manifest=manifest, sessions=c['evaluation']['calibration_sessions'])
            logs.durable()
        evidence = dict(run=str(run), run_sha256=fingerprint(run / 'run.json'), seed=c['seed'], budget=budget, validation_sessions_path=str(directory / 'evaluation_sessions.jsonl'), validation_sha256=fingerprint(directory / 'evaluation_sessions.jsonl'), validation_metrics_path=str(directory / 'metrics.jsonl'), validation_metrics_sha256=fingerprint(directory / 'metrics.jsonl'))
        try:
            target = calibration_target(loaded, ref['macro']['J'], rnd['macro']['J'], evidence)
        except ValueError:
            atomic_json(root / 'calibration_status.json', dict(status='needs_review', environment=c['environment'], J_ref=ref['macro']['J'], J_random=rnd['macro']['J'], evidence=evidence))
            raise
        validate_calibration_evidence(target, loaded)
        targets[c['environment']] = target
        print(f"Frozen target {c['environment']}: J*={target['J_star']:.6f}", flush=True)
    atomic_json(path, dict(schema_version=2, targets=targets, rule='J_random + 0.9 * (J_independent_full_budget_FedAvg - J_random)', interpretation='Frozen before comparison runs; validation only; independent training seed 3026.'))
    return path

def measured_outcome(records, config, target, endpoint):
    from .costs import target_outcome
    val = [r for r in records if r['split'] == 'validation' and r['requests_per_client'] <= endpoint]
    needed = {0} if config['method'] == 'random' else {s for s in validation_steps(config) if s <= endpoint}
    if len(val) != len(needed) or {r['requests_per_client'] for r in val} != needed:
        raise ValueError('Incomplete round-level validation trajectory')
    for r in val:
        if r['eval_sessions'] != config['evaluation']['validation_sessions']:
            raise ValueError('Wrong validation sample count')
        if r['scheduled_round_index'] != scheduled_round(config, r['requests_per_client']):
            raise ValueError('Incorrect validation round index')
        if not np.isfinite(r['macro']['J']):
            raise ValueError('Nonfinite validation return')
    return target_outcome(val, target, config['method'])

def summarize_costs(rows):
    import pandas as pd
    frame = pd.DataFrame(rows)
    out = []
    for (env, method), g in frame.groupby(['environment', 'method']):
        if set(g.seed) != {2026, 2027, 2028} or len(g) != 3:
            raise ValueError('Cost summary needs three unique comparison seeds')
        if g.J_star.nunique() != 1:
            raise ValueError('Different targets within comparison')
        reached = int((g.status == 'reached').sum())
        all_reached = reached == 3
        status = 'reached' if all_reached else 'N/A' if (g.status == 'N/A').all() else f'NR ({reached}/3 reached)'
        out.append(dict(environment=env, method=method, J_star=float(g.J_star.iloc[0]), training_minutes_mean=float(g.training_seconds.mean() / 60), training_minutes_sd=float(g.training_seconds.std(ddof=1) / 60), communication_MB_mean=float(g.communication_bytes.mean() / 1000000.0), communication_MB_sd=float(g.communication_bytes.std(ddof=1) / 1000000.0), reached_seeds=reached, total_seeds=3, rounds_status=status, rounds_mean=float(g.rounds_to_target.mean()) if all_reached else None, rounds_sd=float(g.rounds_to_target.std(ddof=1)) if all_reached else None))
    return out

def cost_report(runs, output, jobs, phase):
    import pandas as pd
    from .ablation_reporting import LABELS
    rows = []
    for c in jobs:
        directory = Path(runs) / run_id(c)
        meta = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        target = load_trial_target(meta['config'])
        if target is None:
            raise ValueError('Cost report requires precommitted targets')
        if meta.get('cost_target_hash') != target['hash']:
            raise ValueError('Training target differs from report target')
        endpoint = c['training']['requests_per_client'] // (2 if phase == 'half' else 1)
        records = [json.loads(l) for l in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        outcome = measured_outcome(records, c, target['J_star'], endpoint)
        if meta.get('rounds_to_target') != outcome:
            raise ValueError('Stored target outcome differs from validation evidence')
        rows.append(dict(environment=c['environment'], method=c['method'], seed=c['seed'], J_star=target['J_star'], target_hash=target['hash'], endpoint=endpoint, censoring_round=scheduled_round(c, endpoint), **outcome, training_seconds=meta['timing']['training_elapsed_seconds'], communication_bytes=meta['communication_bytes'], timing_eligible=meta.get('cost_eligible', False), timing_limitations=json.dumps(meta.get('cost_ineligible_reasons', []))))
    output = Path(output)
    pd.DataFrame(rows).to_csv(output / 'costs_by_seed.csv', index=False)
    summary = summarize_costs(rows)
    pd.DataFrame(summary).to_csv(output / 'cost_summary.csv', index=False)
    lines = ['# Precommitted validation target costs', '', 'Time/communication are observed full-phase costs. Calibration and evaluation time are excluded.', 'NR means not reached by the phase budget; no success-only mean. Local/Random rounds are N/A.', 'Round zero means the target was reached before communication. Training continues to the fixed budget after a hit.', 'Timing eligibility and limitations remain in costs_by_seed.csv; target recovery does not certify controlled wall-time claims.', '', '|Environment|Method|J*|Time (min)|Comm. (MB)|Rounds|', '|---|---|---:|---:|---:|---|']
    for r in summary:
        rounds = f"{r['rounds_mean']:.2f} +/- {r['rounds_sd']:.2f}" if r['rounds_status'] == 'reached' else r['rounds_status']
        lines.append(f"|{r['environment']}|{LABELS[r['method']]}|{r['J_star']:.6f}|{r['training_minutes_mean']:.2f} +/- {r['training_minutes_sd']:.2f}|{r['communication_MB_mean']:.3f} +/- {r['communication_MB_sd']:.3f}|{rounds}|")
    (output / 'COSTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary

def archive_calibration(target_path, output):
    target_path = Path(target_path)
    root = target_path.parent.resolve()
    files = {'cost_targets.json': target_path}
    data = json.loads(target_path.read_text(encoding='utf-8'))
    for env, target in data['targets'].items():
        ev = target['evidence']
        run = Path(ev['run'])
        for name in ('run.json', 'config.json', 'metrics.jsonl', 'resources.jsonl', 'message_ledger.jsonl'):
            if (run / name).exists():
                files[f'{env}/run/{name}'] = run / name
        for key, name in (('validation_sessions_path', 'evaluation_sessions.jsonl'), ('validation_metrics_path', 'metrics.jsonl')):
            files[f'{env}/validation/{name}'] = Path(ev[key])
    manifest = {name: dict(original_path=str(p.resolve()), sha256=fingerprint(p)) for name, p in files.items()}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in files.items():
            z.write(p, name)
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
if __name__ == '__main__':
    import argparse
    from .config import load_config
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'root', 'data-dir', 'response'):
        p.add_argument('--' + key, type=Path, required=key in ('config', 'root'))
    p.add_argument('--devices', default='cpu')
    p.add_argument('--cpu-affinities')
    a = p.parse_args()
    groups = [[int(x) for x in g.split(',')] for g in a.cpu_affinities.split(';')] if a.cpu_affinities else None
    calibrate_trial(load_config(a.config), a.root, a.devices.split(','), a.data_dir, a.response, groups)
