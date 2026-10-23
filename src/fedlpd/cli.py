from pathlib import Path
import argparse, json, os, platform, shutil, sys
from .config import load_config, smoke_config, build_matrix, validate_config, environment_config
from .artifacts import atomic_json

def _common(parser):
    parser.add_argument('--config', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Tiny synthetic nonpublication run')

def parser():
    p = argparse.ArgumentParser(prog='fedlpd')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor')
    d.add_argument('--device', default='cpu')
    d.add_argument('--strict', action='store_true')
    a = sub.add_parser('prepare')
    _common(a)
    a.add_argument('--logs', nargs='+', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--item-features', type=Path)
    a = sub.add_parser('validate-data')
    a.add_argument('--data-dir', type=Path, required=True)
    a = sub.add_parser('fit-response')
    _common(a)
    a.add_argument('--data-dir', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--epochs', type=int)
    a = sub.add_parser('train')
    _common(a)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--data-dir', type=Path)
    a.add_argument('--response', type=Path)
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--resume', type=Path)
    a.add_argument('--environment', choices=['recsim', 'kuaisim'])
    a.add_argument('--method')
    a.add_argument('--seed', type=int)
    a.add_argument('--requests', type=int)
    a.add_argument('--stop-after', type=int, help='Save a resumable pause after this request count')
    a.add_argument('--freeze', type=Path)
    a.add_argument('--targets', type=Path)
    a.add_argument('--purpose', choices=['pilot', 'calibration', 'formal', 'cost_remeasure', 'smoke'])
    a.add_argument('--controlled-load', action='store_true')
    a.add_argument('--cost-remeasure-of', type=Path)
    a.add_argument('--cpu-affinity', help='Comma-separated allocated Linux logical CPU IDs')
    return p

def doctor(device='cpu', strict=False):
    import torch
    result = {'python': platform.python_version(), 'cpu_logical': os.cpu_count(), 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'gpus': [], 'free_disk_gib': round(shutil.disk_usage('.').free / 2 ** 30, 1)}
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        result['gpus'].append({'index': i, 'name': prop.name, 'memory_gib': round(prop.total_memory / 2 ** 30, 1)})
    failures = []
    try:
        from .envs.recsim_backend import load_backend
        load_backend()
        result['recsim_backend'] = 'available'
    except Exception as e:
        result['recsim_backend'] = f'{type(e).__name__}: {e}'
        failures.append('RecSim backend')
    if device.startswith('cuda') and (not torch.cuda.is_available()):
        failures.append('CUDA')
    print(json.dumps(result, indent=2))
    if strict and failures:
        raise RuntimeError('Unavailable: ' + ', '.join(failures))
    return result

def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'doctor':
        doctor(args.device, args.strict)
        return 0
    if args.command in ('prepare', 'fit-response', 'train', 'matrix', 'validate-env', 'calibrate', 'freeze', 'catalog-audit', 'migration-check'):
        c = load_config(args.config)
        if args.smoke:
            c = smoke_config(c)
        if args.command in ('prepare', 'fit-response', 'catalog-audit'):
            c = environment_config(c, 'kuaisim', concrete=True)
        elif getattr(args, 'environment', None):
            c = environment_config(c, args.environment, concrete=True)
        if getattr(args, 'freeze', None):
            c['cost']['freeze_path'] = str(args.freeze.resolve())
        if getattr(args, 'targets', None):
            c['cost']['targets_path'] = str(args.targets.resolve())
    if args.command == 'prepare':
        from .data import prepare_data
        cfg = dict(c['env'], **c['data'])
        cfg['smoke'] = c['smoke']
        print(prepare_data(args.logs, args.output, cfg, args.item_features))
        return 0
    if args.command == 'validate-data':
        from .data import validate_data
        print(json.dumps(validate_data(args.data_dir), indent=2))
        return 0
    if args.command == 'fit-response':
        from .data import fit_response
        from .data.preparation import validate_prepared_config
        validate_prepared_config(json.loads((args.data_dir / 'metadata.json').read_text(encoding='utf-8-sig')), c)
        cfg = dict(c.get('response', {}))
        if args.epochs is not None:
            cfg['epochs'] = args.epochs
        print(fit_response(args.data_dir, args.output, cfg, args.device))
        return 0
    if args.command == 'train':
        if args.cpu_affinity:
            from .resources import set_cpu_affinity
            set_cpu_affinity([int(x) for x in args.cpu_affinity.split(',')])
        from .training import run_experiment
        if args.cost_remeasure_of:
            original = json.loads((args.cost_remeasure_of / 'run.json').read_text(encoding='utf-8'))
            if original.get('status') != 'completed' or not original.get('performance_eligible'):
                raise ValueError('Cost remeasurement requires a completed eligible performance run')
            c = original['config']
            c.update(purpose='cost_remeasure', performance_run_id=original['run_id'], performance_attempt_id=original['attempt_id'])
            c['cost'].update(controlled_load=True, jobs_per_device=1)
        if args.purpose:
            c['purpose'] = args.purpose
        if args.controlled_load:
            c['cost']['controlled_load'] = True
        for key in ('environment', 'method', 'seed'):
            if getattr(args, key) is not None:
                c[key] = getattr(args, key)
        if args.requests is not None:
            c['training']['requests_per_client'] = args.requests
        print(run_experiment(validate_config(c), args.output, args.data_dir, args.response, args.device, args.resume, args.stop_after))
        return 0
    return 1
