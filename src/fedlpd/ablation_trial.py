from copy import deepcopy
from pathlib import Path
import json
from .artifacts import atomic_json
from .config import build_matrix, run_id, stable_hash, validation_steps

def pause_points(jobs):
    points = {run_id(c): c['training']['requests_per_client'] // 2 for c in jobs}
    if any((c['method'] == 'random' or points[run_id(c)] not in validation_steps(c) for c in jobs)):
        raise ValueError('Invalid trial pause matrix')
    return points

def _identity(c):
    c = deepcopy(c)
    c['trial']['freeze_path'] = None
    return stable_hash(c)

def freeze_trial(jobs, path, data=None, response=None):
    from .protocol import source_identity
    from .training import _assets
    path = Path(path).resolve()
    value = dict(version=jobs[0]['trial']['version'], source_hash=source_identity(), jobs={run_id(c): _identity(_assets(c, data, response)) for c in jobs}, pause_points=pause_points([c for c in jobs if c['method'] != 'random']), interpretation='Precommitted half trial; all methods retained. No J* calibration or automatic promotion.')
    if len(value['jobs']) != len(jobs):
        raise ValueError('Duplicate trial IDs')
    if any((c['cost'].get('require_calibrated_target') for c in jobs)):
        from .trial_costs import load_trial_target
        from .artifacts import fingerprint
        if not all((c['cost'].get('require_calibrated_target') for c in jobs)):
            raise ValueError('Mixed target requirement')
        targets = {}
        for c in jobs:
            target = load_trial_target(_assets(c, data, response))
            targets[c['environment']] = target['hash']
        paths = {c['cost']['targets_path'] for c in jobs}
        if len(paths) != 1:
            raise ValueError('Trial must use one common target file')
        value['cost_targets'] = dict(path=next(iter(paths)), sha256=fingerprint(next(iter(paths))), targets=targets)
        value['interpretation'] = 'Independent validation targets frozen before all comparison runs; no automatic promotion.'
    value['hash'] = stable_hash(value)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Trial freeze differs; preserve old results and use a fresh root')
    atomic_json(path, value)
    for c in jobs:
        c['trial']['freeze_path'] = str(path)
    return value

def check_trial_freeze(c):
    from .protocol import source_identity
    path = c['trial'].get('freeze_path')
    if not path:
        raise ValueError('Missing v23 trial freeze')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('hash') != stable_hash({k: v for k, v in value.items() if k != 'hash'}):
        raise ValueError('Trial freeze hash differs')
    if value.get('version') != c['trial']['version'] or value.get('source_hash') != source_identity():
        raise ValueError('Trial source differs')
    if value['jobs'].get(run_id(c)) != _identity(c):
        raise ValueError('Trial config/assets differ')
    if c['cost'].get('require_calibrated_target'):
        from .artifacts import fingerprint
        target = value.get('cost_targets', {})
        path = c['cost'].get('targets_path')
        if not path or not Path(path).is_file() or target.get('path') != path or (target.get('sha256') != fingerprint(path)):
            raise ValueError('Frozen target file missing or changed')
    return value

def prepare(config, root, data=None, response=None, device='cpu'):
    from .config import load_config
    from .data.preparation import validate_data
    from .envs.validation import validate_environment
    from .artifacts import fingerprint
    root = Path(root)
    c = load_config(config)
    if c['cost'].get('require_calibrated_target'):
        c['cost']['targets_path'] = str((root / 'calibration/cost_targets.json').resolve())
    jobs = build_matrix(c, include_random=True)
    receipt = root / 'audit-receipt.json'
    if not receipt.exists():
        if not c['smoke']:
            validate_data(data)
        for env in ('kuaisim', 'recsim'):
            job = next((j for j in jobs if j['environment'] == env and j['method'] == 'full'))
            validate_environment(env, job, root / 'audits' / env, data_dir=data, response_path=response, device=device, sessions_per_client=10 if c['smoke'] else 100)
        files = {p.relative_to(root).as_posix(): fingerprint(p) for p in (root / 'audits').rglob('*') if p.is_file()}
        atomic_json(receipt, files)
    else:
        for name, digest in json.loads(receipt.read_text(encoding='utf-8')).items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or fingerprint(path) != digest:
                raise ValueError('Environment audit changed')
    freeze_trial(jobs, root / 'trial-freeze.json', data, response)
    atomic_json(root / 'matrix-v23.json', jobs)
    return jobs

def execute_matrix(root, runs, phase, devices, data=None, response=None, affinities=None):
    from .scheduler import run_matrix
    from .artifacts import atomic_json
    root = Path(root)
    runs = Path(runs)
    jobs = json.loads((root / 'matrix-v23.json').read_text(encoding='utf-8'))
    freeze_trial(jobs, root / 'trial-freeze.json', data, response)
    random = [c for c in jobs if c['method'] == 'random']
    training = [c for c in jobs if c['method'] != 'random']
    result = run_matrix(random, runs, devices, data, response, cpu_affinities=affinities)
    atomic_json(root / 'random-schedule.json', result)
    if result['failed']:
        raise RuntimeError('Random evaluation failed; inspect worker.log')
    result = run_matrix(training, runs, devices, data, response, cpu_affinities=affinities, stop_after=pause_points(training) if phase == 'half' else None)
    atomic_json(root / (phase + '-schedule.json'), result)
    if result['failed']:
        raise RuntimeError('Training failed; inspect worker.log')
    return result

def finalize(root, runs, phase):
    import shutil, subprocess, sys
    from .optimization_trial_reporting import build_report, archive_evidence
    root = Path(root)
    dest = root / phase
    jobs = json.loads((root / 'matrix-v23.json').read_text(encoding='utf-8'))
    report = build_report(runs, dest / 'report', jobs, phase)
    script = Path(__file__).resolve().parents[2] / 'scripts/export_results.py'
    subprocess.run([sys.executable, '-u', str(script), '--runs', str(runs.resolve()), '--output', str((dest / 'report/data').resolve()), '--phase', phase], check=True)
    if jobs[0]['cost'].get('require_calibrated_target'):
        from .trial_costs import cost_report, archive_calibration
        from .paper_tables import paper_tables
        cost_report(runs, dest / 'report', jobs, phase)
        paper_tables(dest / 'report', jobs, phase)
        archive_calibration(jobs[0]['cost']['targets_path'], dest / 'calibration-evidence.zip')
    archive_evidence(runs, dest / 'analysis-evidence.zip', [run_id(c) for c in jobs])
    for name in ('trial-freeze.json', 'matrix-v23.json', 'audit-receipt.json', 'workflow_manifest.json', 'random-schedule.json', phase + '-schedule.json'):
        if (root / name).is_file():
            shutil.copy2(root / name, dest / name)
    shutil.copytree(root / 'audits', dest / 'audits', dirs_exist_ok=True)
    return report

def export(root, phase):
    import os, tarfile, shutil
    from .artifacts import fingerprint
    root = Path(root)
    if (root / 'workflow-timing.json').exists():
        shutil.copy2(root / 'workflow-timing.json', root / phase / 'workflow-timing.json')
    target = root / (phase + '-v23.tar.gz')
    pending = target.with_suffix('.gz.pending')
    with tarfile.open(pending, 'w:gz', compresslevel=1) as z:
        z.add(root / phase, arcname=phase)
    os.replace(pending, target)
    target.with_name(target.name + '.sha256').write_bytes((fingerprint(target) + '  ' + target.name + '\n').encode('ascii'))
    print('Return this file: ' + str(target.resolve()), flush=True)
    return target
