import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

def main():
    parser = argparse.ArgumentParser(description='Run the complete FedLPD experiment and export numerical results.')
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiment.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3')
    parser.add_argument('--cpu-affinities')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=('calibrate', 'prepare', 'train', 'report'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    from fedlpd.config import load_config, smoke_config
    from fedlpd.artifacts import atomic_json
    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    if not config['smoke'] and (args.data_dir is None or not args.data_dir.is_dir() or args.response is None or (not args.response.is_file())):
        parser.error('A prepared --data-dir and fitted --response are required; use --smoke only for synthetic validation.')
    devices = args.devices.split(',')
    groups = [[int(x) for x in group.split(',')] for group in args.cpu_affinities.split(';')] if args.cpu_affinities else None
    root = args.output.resolve()
    results = root / 'results'
    runs = root / 'runs'
    if args.stage == 'calibrate':
        from fedlpd.trial_costs import calibrate_trial
        calibrate_trial(config, results / 'calibration', devices, args.data_dir, args.response, groups)
        return
    if args.stage == 'prepare':
        from fedlpd.ablation_trial import prepare
        prepare(args.config, results, args.data_dir, args.response, devices[0])
        return
    if args.stage == 'train':
        from fedlpd.ablation_trial import execute_matrix
        execute_matrix(results, runs, 'finish', devices, args.data_dir, args.response, groups)
        return
    if args.stage == 'report':
        from fedlpd.ablation_trial import finalize, export
        finalize(results, runs, 'finish')
        export(results, 'finish')
        return
    from fedlpd.protocol import source_identity
    from fedlpd.artifacts import fingerprint
    identity = dict(config=config, source_hash=source_identity(), scripts={p.name: fingerprint(p) for p in sorted((PROJECT / 'scripts').glob('*.py'))}, devices=devices, cpu_affinities=groups, data_dir=str(args.data_dir.resolve()) if args.data_dir else None, response=str(args.response.resolve()) if args.response else None)
    manifest = root / 'workflow.json'
    if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')) != identity:
        raise ValueError('Configuration, source, assets, or worker allocation changed; use a new output directory.')
    atomic_json(manifest, identity)
    config_path = root / 'config.json'
    atomic_json(config_path, config)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--config', str(config_path), '--output', str(root), '--devices', args.devices]
    for flag, value in (('--data-dir', args.data_dir), ('--response', args.response)):
        if value is not None:
            command += [flag, str(value.resolve())]
    if args.cpu_affinities:
        command += ['--cpu-affinities', args.cpu_affinities]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(PROJECT / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    timing_path = root / 'workflow-timing.json'
    timing = json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else dict(attempts=[])
    attempt = dict(started_at=time.time(), stages=[], status='running')
    timing['attempts'].append(attempt)
    atomic_json(timing_path, timing)
    try:
        for stage in ('calibrate', 'prepare', 'train', 'report'):
            print('Stage: ' + stage, flush=True)
            start = time.perf_counter()
            subprocess.run(command + ['--stage', stage], env=environment, check=True)
            attempt['stages'].append(dict(stage=stage, elapsed_seconds=time.perf_counter() - start))
            atomic_json(timing_path, timing)
        attempt['status'] = 'completed'
    except BaseException:
        attempt['status'] = 'failed'
        raise
    finally:
        attempt['finished_at'] = time.time()
        atomic_json(timing_path, timing)
    print('Result data: ' + str(results / 'finish/report/data'), flush=True)
if __name__ == '__main__':
    main()
