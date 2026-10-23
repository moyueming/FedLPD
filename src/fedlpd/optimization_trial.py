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
    value['hash'] = stable_hash(value)
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError('Trial freeze differs; preserve old results and use a fresh root')
    atomic_json(path, value)
    for c in jobs:
        c['trial']['freeze_path'] = str(path)
    return value

def check_trial_freeze(c):
    if c.get('trial', {}).get('version') == 'v23':
        from .ablation_trial import check_trial_freeze as check_v23
        return check_v23(c)
    from .protocol import source_identity
    path = c['trial'].get('freeze_path')
    if not path:
        raise ValueError('Missing v19 trial freeze')
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('hash') != stable_hash({k: v for k, v in value.items() if k != 'hash'}):
        raise ValueError('Trial freeze hash differs')
    if value.get('version') != c['trial']['version'] or value.get('source_hash') != source_identity():
        raise ValueError('Trial source differs')
    if value['jobs'].get(run_id(c)) != _identity(c):
        raise ValueError('Trial config/assets differ')
    return value

def prepare(config, root, data=None, response=None, device='cpu'):
    from .config import load_config
    from .data.preparation import validate_data
    from .envs.validation import validate_environment
    from .artifacts import fingerprint
    root = Path(root)
    c = load_config(config)
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
    atomic_json(root / 'matrix-v19.json', jobs)
    return jobs

def execute_matrix(root, runs, phase, devices, data=None, response=None, affinities=None):
    from .scheduler import run_matrix
    from .artifacts import atomic_json
    root = Path(root)
    runs = Path(runs)
    jobs = json.loads((root / 'matrix-v19.json').read_text(encoding='utf-8'))
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
    import shutil
    from .optimization_trial_reporting import build_report, archive_evidence
    root = Path(root)
    dest = root / phase
    jobs = json.loads((root / 'matrix-v19.json').read_text(encoding='utf-8'))
    report = build_report(runs, dest / 'report', jobs, phase)
    archive_evidence(runs, dest / 'analysis-evidence.zip', [run_id(c) for c in jobs])
    for name in ('trial-freeze.json', 'matrix-v19.json', 'audit-receipt.json', 'workflow_manifest.json', 'random-schedule.json', phase + '-schedule.json'):
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
    target = root / (phase + '-v19.tar.gz')
    pending = target.with_suffix('.gz.pending')
    with tarfile.open(pending, 'w:gz', compresslevel=1) as z:
        z.add(root / phase, arcname=phase)
    os.replace(pending, target)
    target.with_name(target.name + '.sha256').write_bytes((fingerprint(target) + '  ' + target.name + '\n').encode('ascii'))
    print('Return this file: ' + str(target.resolve()), flush=True)
    return target

def main(argv=None):
    import argparse, os, sys
    from .full_workflow import Journal, isolated_runtime_check
    from .config import load_config
    from .protocol import source_identity
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['half', 'finish', 'prepare', 'matrix', 'report'], nargs='?', default='half')
    p.add_argument('--phase', choices=['half', 'finish'], default='half')
    for name in ('root', 'runs', 'config', 'data-dir', 'response'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--devices', default='cpu')
    p.add_argument('--cpu-affinities')
    args = p.parse_args(argv)
    if args.action == 'prepare':
        prepare(args.config, args.root, args.data_dir, args.response, args.device)
        return 0
    if args.action == 'matrix':
        groups = [[int(x) for x in g.split(',')] for g in args.cpu_affinities.split(';')] if args.cpu_affinities else None
        execute_matrix(args.root, args.runs, args.phase, args.devices.split(','), args.data_dir, args.response, groups)
        return 0
    if args.action == 'report':
        finalize(args.root, args.runs, args.phase)
        return 0
    env = os.environ
    root = Path(env.get('FEDLPD_COMPLETE_ROOT', 'results/optimization-v19')).resolve()
    runs = Path(env.get('FEDLPD_RUNS', 'runs/optimization-v19')).resolve()
    config = Path(env['FEDLPD_CONFIG']).resolve()
    data = Path(env['FEDLPD_DATA_DIR']).resolve()
    response = Path(env['FEDLPD_RESPONSE']).resolve()
    devices = env['FEDLPD_DEVICES'].split(',')
    affinities = env['FEDLPD_CPU_AFFINITIES']
    isolated_runtime_check(config, env['FEDLPD_PREFLIGHT'], devices, affinities, 'optimization-v19')
    if not data.is_dir() or not response.is_file():
        raise ValueError('Existing prepared assets required')
    if args.action == 'finish' and (not (root / 'half/report/stage-results.json').is_file()):
        raise ValueError('Complete half stage before explicitly continuing')
    identity = dict(version='v19', source_hash=source_identity(), config=load_config(config), runs=str(runs), data=str(data), response=str(response), devices=devices, affinities=affinities)
    path = root / 'workflow_manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Workflow changed; use a new root')
    atomic_json(path, identity)
    journal = Journal(root / 'workflow-timing.json')
    command = [sys.executable, '-u', '-m', 'fedlpd.optimization_trial']
    assets = ['--data-dir', data, '--response', response]
    common = ['--root', root, '--runs', runs]
    try:
        journal.command('audit_and_freeze', ['taskset', '-c', affinities.split(';')[0]] + command + ['prepare', '--config', config, '--root', root, '--device', devices[0]] + assets)
        journal.command(args.action + '_matrix', command + ['matrix', '--phase', args.action] + common + assets + ['--devices', ','.join(devices), '--cpu-affinities', affinities])
        journal.command(args.action + '_report', command + ['report', '--phase', args.action] + common)
    except BaseException as exc:
        journal.finish('failed', f'{type(exc).__name__}: {exc}')
        raise
    journal.finish('paused_for_review' if args.action == 'half' else 'completed')
    export(root, args.action)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
