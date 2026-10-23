from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import json
import platform
import zipfile
import numpy as np
from .artifacts import atomic_json, fingerprint
from .config import build_matrix, stable_hash, task_hash, protocol_hash, evaluation_hash, method_config_hash
from .evaluation import initialization_manifest

def source_identity():
    root = Path(__file__).parent
    digest = sha256()
    for file in sorted(root.rglob('*.py')):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()

def dependency_manifest():
    return {'python': platform.python_version(), 'packages': dict(sorted(((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))))}

def snapshot_source(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'source.zip'
    root = Path(__file__).parent
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*.py')):
            archive.write(file, 'fedlpd/' + file.relative_to(root).as_posix())
        project = root.parents[1]
        for component in ('scripts', 'configs', 'docs'):
            for file in sorted((project / component).rglob('*')):
                if file.is_file() and file.suffix in ('.py', '.sh', '.json', '.md'):
                    archive.write(file, file.relative_to(project).as_posix())
        for name in ('pyproject.toml', 'requirements.txt', 'requirements-server.txt', 'README.md'):
            if (project / name).is_file():
                archive.write(project / name, name)
    dependencies = dependency_manifest()
    atomic_json(directory / 'dependencies.json', dependencies)
    return dict(source_hash=source_identity(), source_archive_sha256=fingerprint(path), dependency_manifest_hash=stable_hash(dependencies))

def calibration_target(config, reference, random_return, evidence):
    if not np.isfinite([reference, random_return]).all() or reference <= random_return:
        raise ValueError('needs_review: calibration FedAvg must exceed Random on validation')
    manifest = initialization_manifest(config, 'validation', config['evaluation']['calibration_sessions'])
    value = dict(status='frozen', environment=config['environment'], J_ref=float(reference), J_random=float(random_return), J_star=float(random_return + config['cost']['target_fraction'] * (reference - random_return)), fraction=config['cost']['target_fraction'], task_hash=task_hash(config), validation_manifest_hash=manifest['hash'], calibration_seed=config['cost']['pilot_seed'], evidence=evidence, source_hash=source_identity(), smoke=bool(config['smoke']))
    return dict(value, hash=stable_hash(value))

def _entry(c):
    return dict(environment=c['environment'], method=c['method'], seed=c['seed'], budget=c['training']['requests_per_client'], task_hash=task_hash(c), training_protocol_hash=protocol_hash(c), evaluation_hash=evaluation_hash(c), method_config_hash=method_config_hash(c))

def validate_target(target, config):
    if target.get('hash') != stable_hash({k: v for k, v in target.items() if k != 'hash'}):
        raise ValueError('Calibration target checksum mismatch')
    expected = calibration_target(config, target['J_ref'], target['J_random'], target.get('evidence', {}))
    for key in ('J_star', 'fraction', 'task_hash', 'validation_manifest_hash', 'calibration_seed', 'smoke', 'environment', 'source_hash'):
        if target.get(key) != expected[key]:
            raise ValueError('Calibration target rule mismatch: ' + key)
    if not config['smoke']:
        from .calibration_evidence import validate_calibration_evidence
        validate_calibration_evidence(target, config)

def execution_config(config):
    return dict(cpu_threads=config['training']['cpu_threads'], precision='float32', client_order=list(range(config['env']['num_clients'])), checkpoint_every=config['training']['checkpoint_every'], resume_every=config['training']['resume_every'], storage=config['storage'])

def _preflight_evidence(base, pilot_root, environment_reports, data_dir, response_path):
    from .training import _assets
    if base['smoke']:
        return {'nonpublication_smoke': True, 'files': {}, 'mode': base.get('preflight', {}).get('mode', 'all_paths'), 'pilot_execution_verified': False}
    mode = base.get('preflight', {}).get('mode', 'all_paths')
    if mode not in ('all_paths', 'environment_audit_only'):
        raise ValueError('Invalid preflight.mode')
    direct = mode == 'environment_audit_only'
    if not direct and (not pilot_root) or not environment_reports:
        raise ValueError('Formal freeze requires all-path pilot and environment audit evidence')
    expected = {}
    for c in build_matrix(base):
        if c['seed'] == 2026:
            c = _assets(c, data_dir if c['environment'] == 'kuaisim' else None, response_path if c['environment'] == 'kuaisim' else None)
            expected[c['environment'], c['method']] = c
    files = {}
    seen = set()
    execution_summaries = {}
    from .pilot_evidence import validate_pilot_configuration, validate_pilot_rounds, compatible_source
    compatibility = Path(__file__).resolve().parents[2] / 'configs/pilot-source-review.json'
    if not direct and compatibility.is_file():
        files[str(compatibility.resolve())] = fingerprint(compatibility)
    for path in [] if direct else sorted(Path(pilot_root).rglob('run.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        key = (meta.get('environment'), meta.get('method'))
        if key not in expected or meta.get('phase') not in ('all_paths', 'quick_paths', 'learning_diagnostic'):
            continue
        if key in seen:
            raise ValueError('Duplicate all-path pilot evidence: ' + str(key))
        c = expected[key]
        minimum = c['training']['warmup'] + 2 * c['training']['communication_every']
        short = meta.get('phase') in ('quick_paths', 'learning_diagnostic')
        if short:
            validate_pilot_configuration(meta, c, base)
            execution_summaries['/'.join(key)] = validate_pilot_rounds(path.parent, meta)
        elif meta.get('status') != 'completed' or meta.get('smoke') or meta.get('purpose') != 'pilot' or (meta.get('seed') != base['cost']['pilot_seed']) or (meta.get('requests_per_client', 0) < minimum) or (meta.get('source_hash') != source_identity()) or (meta.get('task_hash') != task_hash(c)) or (meta.get('method_config_hash') != method_config_hash(c)):
            raise ValueError('Incompatible or incomplete pilot: ' + str(path))
        recorded = meta['config']['training']
        wanted = c['training']
        if not short and any((recorded.get(k) != v for k, v in wanted.items() if k != 'requests_per_client')):
            raise ValueError('Pilot training configuration changed: ' + str(path))
        receipt = path.parent / 'retention.json'
        if not receipt.exists():
            raise ValueError('Missing pilot retention receipt')
        retained = json.loads(receipt.read_text(encoding='utf-8'))['files']
        if short:
            from .scheduler import verify_completed_retention
            from .pilot_evidence import verify_pilot_source_snapshot
            verify_completed_retention(path.parent, meta['configuration_hash'])
            verify_pilot_source_snapshot(path.parent, meta['source_hash'])
        for name, digest in retained.items():
            asset = path.parent / name
            if not asset.is_file() or fingerprint(asset) != digest:
                raise ValueError('Pilot evidence changed: ' + str(asset))
            files[str(asset.resolve())] = digest
        files[str(receipt.resolve())] = fingerprint(receipt)
        if key[1] not in ('local', 'fedavg'):
            rounds = [json.loads(x) for x in (path.parent / 'rounds.jsonl').read_text(encoding='utf-8').splitlines()]
            active = [r for r in rounds if not r.get('skipped') and r.get('completed_aggregation_count', 0) > 0]
            if not active or not any((r.get('geometry') for r in active)):
                raise ValueError('Pilot never executed geometry: ' + str(key))
            summary = execution_summaries.get('/'.join(key), {})
            verified_abstention = short and key[1] != 'full' and (summary.get('rank_abstention_rounds') == len(active))
            if not any((r.get('selected_probes') for r in active)) and (not verified_abstention):
                raise ValueError('Pilot has no effective probes: ' + str(key))
            updates = sum((r.get('distillation_update_count', 0) for r in active))
            if key[1] == 'distillation_off' and updates != 0 or (key[1] != 'distillation_off' and updates <= 0 and (not verified_abstention)):
                raise ValueError('Pilot distillation path did not execute as configured: ' + str(key))
        seen.add(key)
    if not direct and seen != set(expected):
        raise ValueError('Missing all-path pilots: ' + str(sorted(set(expected) - seen)))
    audited = set()
    for value in environment_reports:
        path = Path(value)
        path = path / 'environment_validation.json' if path.is_dir() else path
        report = json.loads(path.read_text(encoding='utf-8'))
        env = report.get('environment')
        if env not in ('kuaisim', 'recsim') or env in audited:
            raise ValueError('Unknown or duplicate environment audit')
        c = expected[env, 'full']
        if report.get('purpose') != 'environment_validation' or report.get('split') != 'validation' or report.get('task_hash') != task_hash(c) or (not (report.get('source_hash') == source_identity() if direct else compatible_source(report.get('source_hash')))) or (set(report.get('checks', {})) != {'legal_candidates', 'visible_information', 'bounded_reward', 'horizon'}) or (not all(report['checks'].values())):
            raise ValueError('Environment audit is stale or incompatible')
        sessions = path.parent / 'environment_sessions.jsonl'
        if not sessions.is_file() or fingerprint(sessions) != report.get('sessions_sha256'):
            raise ValueError('Environment audit session evidence changed')
        files[str(path.resolve())] = fingerprint(path)
        files[str(sessions.resolve())] = fingerprint(sessions)
        audited.add(env)
    if audited != {'kuaisim', 'recsim'}:
        raise ValueError('Both environment audits are required')
    response_report = Path(response_path).with_suffix('.metrics.json') if response_path else None
    if not response_report or not response_report.is_file():
        raise ValueError('Missing fitted response quality report')
    quality = json.loads(response_report.read_text(encoding='utf-8'))
    if quality.get('checkpoint_sha256') != fingerprint(response_path):
        raise ValueError('Response quality report checkpoint changed')
    files[str(response_report.resolve())] = fingerprint(response_report)
    return {'nonpublication_smoke': False, 'files': files, 'pilot_count': len(seen), 'environment_count': len(audited), 'execution_summaries': execution_summaries, 'mode': mode, 'pilot_execution_verified': not direct, 'pilot_omission_reason': 'User selected direct full experiments without short pilots' if direct else None}

def _runtime_evidence(base, path):
    if base['smoke']:
        return {'nonpublication_smoke': True}
    if path is None:
        return {'status': 'not_attached', 'cost_eligible': False, 'interpretation': 'Runtime forecasts are optional scheduling aids. Formal measured cost eligibility is unchanged.'}
    path = Path(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    files = value.get('benchmark_files', {})
    if value.get('training_jobs') != 42 or value.get('random_evaluation_jobs') != 6 or (not files):
        raise ValueError('Runtime estimate must cover the v2 workload and benchmark evidence')
    if not np.isfinite(value.get('serial_hours', float('nan'))) or value['serial_hours'] <= 0:
        raise ValueError('Invalid runtime estimate')
    seen = set()
    hardware = []
    for filename, digest in files.items():
        file = Path(filename)
        if not file.is_file() or fingerprint(file) != digest:
            raise ValueError('Benchmark evidence changed: ' + filename)
        if file.name == 'run.json':
            meta = json.loads(file.read_text(encoding='utf-8'))
            key = (meta.get('environment'), meta.get('method'))
            if key in seen:
                raise ValueError('Duplicate runtime benchmark')
            if meta.get('smoke') or meta.get('status') != 'completed' or meta.get('purpose') != 'pilot' or (meta.get('phase') != 'benchmark') or (meta.get('seed') != base['cost']['pilot_seed']) or (meta.get('source_hash') != source_identity()) or (meta.get('requests_per_client', 0) < 12000):
                raise ValueError('Runtime benchmark identity changed')
            seen.add(key)
            hardware.append(meta['hardware'])
    if seen != {(e, m) for e in ('kuaisim', 'recsim') for m in ('local', 'fedavg', 'full')}:
        raise ValueError('Runtime estimate requires six independent benchmark configurations')
    return dict(path=str(path.resolve()), sha256=fingerprint(path), files=files, hardware=hardware, serial_hours=value['serial_hours'], assumed_effective_speedup=value.get('assumed_effective_speedup'))

def freeze_protocol(base, targets_path, output_dir, data_dir=None, response_path=None, pilot_root=None, environment_reports=None, runtime_estimate=None):
    from .training import _assets
    targets_path = Path(targets_path).resolve()
    targets = json.loads(targets_path.read_text(encoding='utf-8'))
    directory = Path(output_dir).resolve()
    if (directory / 'protocol_freeze.json').exists():
        raise FileExistsError('Protocol already frozen; use a new version directory')
    preflight = _preflight_evidence(base, pilot_root, environment_reports, data_dir, response_path)
    runtime = _runtime_evidence(base, runtime_estimate)
    entries = []
    manifests = {}
    for c in build_matrix(base, include_random=True):
        c = _assets(c, data_dir if c['environment'] == 'kuaisim' else None, response_path if c['environment'] == 'kuaisim' else None)
        target = targets['targets'].get(c['environment'])
        if not target or target['status'] != 'frozen' or target['task_hash'] != task_hash(c):
            raise ValueError('Calibration target task differs from frozen configuration')
        if bool(target['smoke']) != bool(c['smoke']):
            raise ValueError('Cannot freeze formal protocol using smoke targets')
        if target.get('source_hash') != source_identity():
            raise ValueError('Calibration source differs; repeat independent calibration')
        validate_target(target, c)
        entries.append(_entry(c))
        for split in ('validation', 'test'):
            key = c['environment'] + '-' + split
            manifests[key] = initialization_manifest(c, split)
    directory.mkdir(parents=True, exist_ok=True)
    identity = snapshot_source(directory)
    for key, value in manifests.items():
        atomic_json(directory / (key + '-initializations.json'), value)
    design = Path(__file__).resolve().parents[2] / 'docs/FedLPD-experiment-design-v1.md'
    payload = dict(schema_version=2, design_version='2.0', design_sha256=fingerprint(design) if design.exists() else None, entries=entries, targets_path=str(targets_path), targets_sha256=fingerprint(targets_path), initialization_manifests={k: v['hash'] for k, v in manifests.items()}, execution_config=execution_config(base), preflight_evidence=preflight, runtime_estimate=runtime, cost_config={k: v for k, v in base['cost'].items() if k not in ('freeze_path', 'targets_path')}, **identity)
    atomic_json(directory / 'protocol_freeze.json', payload)
    return directory / 'protocol_freeze.json'

def check_frozen_protocol(config):
    path = config.get('cost', {}).get('freeze_path')
    if not path:
        raise ValueError('Formal runs require a precommitted protocol freeze (--freeze)')
    path = Path(path)
    freeze = json.loads(path.read_text(encoding='utf-8'))
    target_path = Path(config['cost'].get('targets_path') or freeze['targets_path'])
    if fingerprint(target_path) != freeze['targets_sha256']:
        raise ValueError('Frozen target file changed')
    if source_identity() != freeze['source_hash']:
        raise ValueError('Frozen protocol source changed')
    design = Path(__file__).resolve().parents[2] / 'docs/FedLPD-experiment-design-v1.md'
    if design.exists() and fingerprint(design) != freeze.get('design_sha256'):
        raise ValueError('Frozen experiment design changed')
    if stable_hash(dependency_manifest()) != freeze['dependency_manifest_hash']:
        raise ValueError('Frozen protocol dependencies changed')
    if fingerprint(path.parent / 'source.zip') != freeze['source_archive_sha256']:
        raise ValueError('Frozen source archive changed')
    evidence = dict(freeze.get('preflight_evidence', {}).get('files', {}))
    runtime = freeze.get('runtime_estimate', {})
    evidence.update(runtime.get('files', {}))
    if runtime.get('path'):
        evidence[runtime['path']] = runtime['sha256']
    for filename, digest in evidence.items():
        asset = Path(filename)
        if not asset.is_file() or fingerprint(asset) != digest:
            raise ValueError('Frozen preflight evidence changed: ' + filename)
    if _entry(config) not in freeze['entries']:
        raise ValueError('Run configuration differs from frozen protocol')
    if execution_config(config) != freeze['execution_config']:
        raise ValueError('Frozen execution quota changed')
    if config.get('preflight', {}).get('mode', 'all_paths') != freeze['preflight_evidence'].get('mode', 'all_paths'):
        raise ValueError('Frozen preflight mode changed')
    cost = {k: v for k, v in config['cost'].items() if k not in ('freeze_path', 'targets_path')}
    expected_cost = dict(freeze['cost_config'])
    if config.get('purpose') == 'cost_remeasure':
        expected_cost['controlled_load'] = True
        expected_cost['jobs_per_device'] = 1
    if cost != expected_cost:
        raise ValueError('Cost configuration differs from frozen protocol')
    target = json.loads(target_path.read_text(encoding='utf-8'))['targets'][config['environment']]
    validate_target(target, config)
    for split in ('test', 'validation'):
        key = config['environment'] + '-' + split
        recorded = json.loads((path.parent / (key + '-initializations.json')).read_text(encoding='utf-8'))
        expected = initialization_manifest(config, split)
        if recorded != expected or recorded['hash'] != freeze['initialization_manifests'][key]:
            raise ValueError('Frozen evaluation manifest changed')
    return dict(target=target, freeze_sha256=fingerprint(path), freeze=freeze)

def calibrate(base, output_dir, data_dir=None, response_path=None, device='cpu'):
    from .training import run_experiment, load_agents_from_checkpoint, _assets
    from .evaluation import evaluate
    from .artifacts import LogBook
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    targets = {}
    for environment in ('kuaisim', 'recsim'):
        c = deepcopy(base)
        c.update(environment=environment, method='fedavg', seed=c['cost']['pilot_seed'], purpose='calibration')
        if not c['smoke']:
            c['training']['requests_per_client'] = 80000 if environment == 'kuaisim' else 40000
        c['storage']['mode'] = 'full'
        data = data_dir if environment == 'kuaisim' else None
        response = response_path if environment == 'kuaisim' else None
        run = directory / (environment + '-fedavg-pilot')
        resume = None
        if (run / 'run.json').exists():
            previous = json.loads((run / 'run.json').read_text(encoding='utf-8'))
            if previous['configuration_hash'] != stable_hash(_assets(c, data, response)):
                raise ValueError('Calibration configuration/assets changed')
            if previous['status'] != 'completed':
                resume = run / 'resume.pt'
        if not (run / 'run.json').exists() or resume:
            run_experiment(c, run, data, response, device, resume)
        agents, loaded = load_agents_from_checkpoint(run, c['training']['requests_per_client'], device)
        manifest = initialization_manifest(loaded, 'validation', c['evaluation']['calibration_sessions'])
        evaluation_dir = directory / (environment + '-validation')
        if evaluation_dir.exists():
            for file in evaluation_dir.glob('*.jsonl'):
                file.unlink()
        evaluation_dir.mkdir(exist_ok=True)
        with LogBook(evaluation_dir) as logs:
            reference = evaluate(agents, loaded, c['training']['requests_per_client'], 'calibration', data, response, device, logs, split='validation', manifest=manifest, sessions=c['evaluation']['calibration_sessions'])
            random_config = deepcopy(loaded)
            random_config['method'] = 'random'
            random_result = evaluate([], random_config, 0, 'calibration', data, response, device, logs, split='validation', manifest=manifest, sessions=c['evaluation']['calibration_sessions'])
            logs.durable()
        evidence = dict(run=str(run.resolve()), run_sha256=fingerprint(run / 'run.json'), seed=c['seed'], validation_sessions_path=str((evaluation_dir / 'evaluation_sessions.jsonl').resolve()), validation_sha256=fingerprint(evaluation_dir / 'evaluation_sessions.jsonl'), validation_metrics_path=str((evaluation_dir / 'metrics.jsonl').resolve()), validation_metrics_sha256=fingerprint(evaluation_dir / 'metrics.jsonl'), budget=c['training']['requests_per_client'])
        try:
            targets[environment] = calibration_target(loaded, reference['macro']['J'], random_result['macro']['J'], evidence)
        except ValueError:
            atomic_json(directory / 'calibration_status.json', dict(status='needs_review', environment=environment, J_ref=reference['macro']['J'], J_random=random_result['macro']['J'], evidence=evidence))
            raise
    path = directory / 'cost_targets.json'
    atomic_json(path, dict(schema_version=2, targets=targets))
    return path
