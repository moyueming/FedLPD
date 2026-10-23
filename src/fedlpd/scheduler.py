from collections import deque
from pathlib import Path
import json, os, signal, subprocess, sys, threading, time
from .artifacts import atomic_json, fingerprint
from .config import run_id, stable_hash, validate_config

def _diagnostic_replacement(directory, checkpoint, checksum, configuration_hash):
    link_path = directory / 'diagnostic_retention.json'
    if not link_path.is_file():
        return False
    link = json.loads(link_path.read_text(encoding='utf-8'))
    if link.get('checkpoint') != checkpoint or link.get('checkpoint_sha256') != checksum:
        return False
    path = Path(link.get('receipt_path', ''))
    if not path.is_absolute() or not path.is_file() or fingerprint(path) != link.get('receipt_sha256'):
        return False
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('checkpoint_sha256') != checksum or receipt.get('inputs', {}).get('configuration_hash') != configuration_hash:
        return False
    request = int(Path(checkpoint).stem.split('-')[-1])
    if receipt.get('inputs', {}).get('requests_per_client') != request:
        return False
    required = {'diagnostics.json', 'client_pair_discrepancies.csv', 'latents.npz', 'discrepancies.png', 'discrepancies.pdf', 'latent_pca.png', 'latent_pca.pdf', 'latent_pca_reward.png', 'latent_pca_reward.pdf', 'plot_arrays.npz'}
    files = receipt.get('files', {})
    truth = receipt.get('truth_cache_files', {})
    if not required <= files.keys() or len(truth) != 2:
        return False
    for name, digest in files.items():
        item = (path.parent / name).resolve()
        if not item.is_relative_to(path.parent.resolve()) or not item.is_file() or fingerprint(item) != digest:
            return False
    return all((Path(name).is_file() and fingerprint(name) == digest for name, digest in truth.items()))

def verify_completed_retention(directory, configuration_hash):
    directory = Path(directory).resolve()
    path = directory / 'retention.json'
    if (directory / 'paper_retention.json').is_file():
        from .paper_archive import verify_archived_run
        verify_archived_run(directory, configuration_hash)
        return True
    if not path.is_file():
        raise ValueError('Missing completed retention receipt: ' + str(directory))
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('schema_version') != 2 or receipt.get('configuration_hash') != configuration_hash:
        raise ValueError('Completed retention receipt identity differs: ' + str(directory))
    files = receipt.get('files', {})
    required = {'run.json', 'config.json', 'packages.json', 'metrics.jsonl', 'evaluation_sessions.jsonl', 'message_ledger.jsonl', 'source/source.zip', 'source/dependencies.json', 'test-initializations.json', 'validation-initializations.json'}
    metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if metadata.get('method') not in ('local', 'local_fixed', 'random', 'central_shared'):
        required.add('rounds.jsonl')
    cfg = metadata.get('config', {})
    if cfg.get('study'):
        required.add('task-sampling.json')
    if cfg.get('storage', {}).get('analysis_checkpoints', False) and metadata.get('method') != 'random':
        budget = metadata['requests_per_client']
        required.update((f'checkpoints/request-{step}.pt' for step in {budget, cfg['evaluation']['comparison_requests']} if step <= budget))
    if metadata.get('environment') == 'recsim' and metadata.get('method') in ('full', 'alignment_off'):
        comparison = cfg.get('evaluation', {}).get('comparison_requests')
        budget = metadata.get('requests_per_client', cfg.get('training', {}).get('requests_per_client', 0))
        if comparison is not None and comparison <= budget:
            required.add(f'checkpoints/request-{comparison}.pt')
    if not required <= files.keys():
        raise ValueError('Incomplete completed retention receipt: ' + str(directory))
    for name, digest in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise ValueError('Invalid retention file path')
        if target.is_file() and fingerprint(target) == digest:
            continue
        if not target.exists() and name.startswith('checkpoints/') and _diagnostic_replacement(directory, name, digest, configuration_hash):
            continue
        raise ValueError('Completed retention checksum failed: ' + str(target))
    return True

def run_matrix(jobs, output_root, devices=None, data_dir=None, response_path=None, jobs_per_device=1, dry_run=False, resume=True, fail_fast=True, cpu_affinities=None, stop_after=None):
    devices = ['cpu'] if devices is None else devices
    if not devices or type(jobs_per_device) is not int or jobs_per_device < 1:
        raise ValueError('Invalid worker allocation')
    import torch
    canonical = []
    for device in devices:
        parsed = torch.device(device)
        if parsed.type not in ('cpu', 'cuda'):
            raise ValueError('Unsupported worker device: ' + str(device))
        canonical.append('cpu' if parsed.type == 'cpu' else f'cuda:{parsed.index or 0}')
    if len(set(canonical)) != len(canonical):
        raise ValueError('Duplicate worker device allocation')
    devices = canonical
    if cpu_affinities is not None:
        if len(cpu_affinities) != len(devices) * jobs_per_device or any((not g for g in cpu_affinities)):
            raise ValueError('Need one CPU allocation per worker')
        flat = [x for group in cpu_affinities for x in group]
        if any((type(x) is not int or x < 0 for x in flat)):
            raise ValueError('CPU allocations require nonnegative integer IDs')
        if len(set(flat)) != len(flat):
            raise ValueError('Worker CPU allocations overlap')
    for c in jobs:
        if c.get('cost', {}).get('jobs_per_device', 1) != jobs_per_device:
            raise ValueError('Actual jobs_per_device differs from declared worker allocation')
        if c.get('cost', {}).get('controlled_load') and jobs_per_device != 1:
            raise ValueError('Controlled cost runs require one worker per GPU')
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identifiers = [run_id(c) for c in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate run identifiers in matrix')
    if stop_after is not None:
        from .config import validation_steps
        from .staged_study import seal_pause, verify_pause
        if set(stop_after) != set(identifiers):
            raise ValueError('Pause policy must cover exactly the scheduled matrix')
        for c in jobs:
            step = stop_after[run_id(c)]
            if not (c.get('study') or c.get('trial')) or type(step) is not int or (not 0 < step < c['training']['requests_per_client']) or (step not in validation_steps(c)):
                raise ValueError('Study pause must precede final budget at a scheduled validation point')
    configs = root / '_configs'
    configs.mkdir(exist_ok=True)
    queue = deque()
    skipped = []
    for c in jobs:
        validate_config(c)
        name = run_id(c)
        directory = root / name
        config_path = configs / (name + '.json')
        atomic_json(config_path, c)
        meta_path = directory / 'run.json'
        if meta_path.exists():
            old = json.loads(meta_path.read_text(encoding='utf-8'))
            from .training import _assets
            if old.get('configuration_hash') != stable_hash(_assets(c, data_dir, response_path)):
                raise ValueError(f'Existing run configuration/assets differ: {name}. Use a new output root.')
            if old['status'] == 'completed':
                if stop_after is not None:
                    raise ValueError('Run already completed; use finish to export final results')
                verify_completed_retention(directory, old['configuration_hash'])
                skipped.append(name)
                continue
            if stop_after is not None and old.get('requests_per_client', 0) > stop_after[name]:
                raise ValueError('Run has passed the interim point; use finish')
            if old['status'] == 'paused' and (directory / 'pause-retention.json').exists():
                from .staged_study import verify_pause
                receipt = json.loads((directory / 'pause-retention.json').read_text(encoding='utf-8'))
                if receipt['requests_per_client'] == old['requests_per_client']:
                    verify_pause(directory, old['requests_per_client'])
            if stop_after is not None and old['status'] == 'paused' and (old.get('requests_per_client') == stop_after[name]):
                if not (directory / 'pause-retention.json').exists():
                    seal_pause(directory, stop_after[name])
                skipped.append(name)
                continue
            if not resume:
                raise ValueError(f'Incomplete run exists: {name}; enable resume or use a new directory')
            if not (directory / 'resume.pt').exists():
                raise ValueError(f'{name} failed before a checkpoint. Inspect its error and use a new output root.')
        queue.append((name, c, config_path, directory))
    summary = {'planned': len(jobs), 'skipped': skipped, 'completed': [], 'failed': 0, 'failures': [], 'devices': devices, 'jobs_per_device': jobs_per_device, 'dry_run': dry_run}
    if stop_after is not None:
        summary.update(paused=[], stop_after=stop_after)
    atomic_json(root / 'schedule.json', summary)
    if dry_run:
        return summary
    from .deployment import isolated_worker_bindings
    bindings = {b['parent_device']: b for b in isolated_worker_bindings(devices)}
    summary['worker_bindings'] = list(bindings.values())
    summary['assignments'] = []
    slots = list(enumerate((device for device in devices for _ in range(jobs_per_device))))
    active = []
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.getsignal(signal.SIGTERM)

        def request_stop(signum, frame):
            raise KeyboardInterrupt('Scheduler termination requested')
        signal.signal(signal.SIGTERM, request_stop)
    try:
        while queue or active:
            free = slots.copy()
            for worker in active:
                free.remove((worker['slot'], worker['device']))
            for slot, device in free:
                if not queue:
                    break
                name, c, config_path, directory = queue.popleft()
                directory.mkdir(exist_ok=True)
                binding = bindings[device]
                command = [sys.executable, '-m', 'fedlpd', 'train', '--config', str(config_path), '--output', str(directory), '--device', binding['child_device']]
                if data_dir:
                    command += ['--data-dir', str(Path(data_dir).resolve())]
                if response_path:
                    command += ['--response', str(Path(response_path).resolve())]
                if cpu_affinities:
                    command += ['--cpu-affinity', ','.join(map(str, cpu_affinities[slot]))]
                if (directory / 'resume.pt').exists():
                    command += ['--resume', str(directory / 'resume.pt')]
                if stop_after is not None:
                    command += ['--stop-after', str(stop_after[name])]
                env = os.environ.copy()
                env['OMP_NUM_THREADS'] = str(c['training']['cpu_threads'])
                env['MKL_NUM_THREADS'] = str(c['training']['cpu_threads'])
                env['OPENBLAS_NUM_THREADS'] = str(c['training']['cpu_threads'])
                env['CUDA_VISIBLE_DEVICES'] = binding['uuid'] or ''
                env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                log = (directory / 'worker.log').open('a', encoding='utf-8')
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close()
                    raise
                active.append({'process': process, 'log': log, 'name': name, 'device': device, 'slot': slot})
                summary['assignments'].append(dict(run=name, slot=slot, **binding))
                print(f'Started {name} on {device}', flush=True)
            for worker in list(active):
                status = worker['process'].poll()
                if status is None:
                    continue
                worker['log'].close()
                active.remove(worker)
                if status == 0:
                    meta = json.loads((root / worker['name'] / 'run.json').read_text())
                    if stop_after is not None:
                        try:
                            seal_pause(root / worker['name'], stop_after[worker['name']])
                        except Exception as exc:
                            print(f"Invalid pause for {worker['name']}: {exc}", flush=True)
                            status = 2
                        else:
                            summary['paused'].append(worker['name'])
                    elif meta.get('status') == 'completed':
                        summary['completed'].append(worker['name'])
                    else:
                        status = 2
                if status != 0:
                    summary['failed'] += 1
                    summary['failures'].append({'run': worker['name'], 'exit_code': status})
                    if fail_fast:
                        queue.clear()
                print(f"Finished {worker['name']} exit={status}", flush=True)
                atomic_json(root / 'schedule.json', summary)
            if active:
                time.sleep(0.25)
    except BaseException:
        for worker in active:
            worker['process'].terminate()
        for worker in active:
            try:
                worker['process'].wait(timeout=60)
            except subprocess.TimeoutExpired:
                worker['process'].kill()
                worker['process'].wait()
            worker['log'].close()
        summary['interrupted'] = True
        atomic_json(root / 'schedule.json', summary)
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
    atomic_json(root / 'schedule.json', summary)
    return summary
