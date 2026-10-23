from __future__ import annotations
import json
import math
from pathlib import Path
from .artifacts import fingerprint
from .config import stable_hash, task_hash, SEEDS
from .evaluation import initialization_manifest
from .metrics import client_metrics, macro_metrics

def _fail(message):
    raise ValueError('Calibration evidence: ' + message)

def _equal_fields(record, expected, context):
    for key, value in expected.items():
        if record.get(key) != value:
            _fail(f'{context} identity mismatch: {key}')

def _close(actual, expected, context):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        _fail(f'{context} must be a finite number')
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        _fail(f'{context} disagrees with recomputed raw-session value')

def _checked_path(value, digest, context):
    if not isinstance(value, str) or not value:
        _fail(f'{context} path is missing')
    path = Path(value)
    if not path.is_absolute():
        _fail(f'{context} path must be absolute')
    if not path.is_file() or not isinstance(digest, str) or fingerprint(path) != digest:
        _fail(f'{context} file is missing or its checksum changed')
    return path

def _json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f'Invalid JSON in {path}: {exc}')
    if not isinstance(value, dict):
        _fail(f'{path} must contain a JSON object')
    return value

def _records(path):
    try:
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    _fail(f'Blank record in {path.name}:{number}')
                record = json.loads(line)
                if not isinstance(record, dict):
                    _fail(f'Invalid record in {path.name}:{number}')
                yield record
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f'Invalid records in {path}: {exc}')

def validate_calibration_evidence(target, config):
    from .protocol import source_identity
    evidence = target.get('evidence')
    if not isinstance(evidence, dict):
        _fail('missing evidence object')
    smoke = bool(config.get('smoke', False))
    environment = config['environment']
    if environment not in ('kuaisim', 'recsim'):
        _fail('unknown environment')
    budget = int(config['training']['requests_per_client']) if smoke else 80000 if environment == 'kuaisim' else 40000
    count = int(config['evaluation']['calibration_sessions'])
    clients = int(config['env']['num_clients'])
    seed = int(config['cost']['pilot_seed'])
    if not smoke and (count != 500 or clients != 5 or seed in SEEDS):
        _fail('formal calibration requires 500 sessions, five clients and an independent seed')
    source = source_identity()
    task = task_hash(config)
    manifest = initialization_manifest(config, 'validation', count)
    _equal_fields(target, dict(status='frozen', environment=environment, smoke=smoke, task_hash=task, source_hash=source, calibration_seed=seed, validation_manifest_hash=manifest['hash']), 'target')
    _equal_fields(evidence, dict(seed=seed, budget=budget), 'declared run')
    run_dir = evidence.get('run')
    if not isinstance(run_dir, str) or not Path(run_dir).is_absolute():
        _fail('run path must be absolute')
    run_path = _checked_path(str(Path(run_dir) / 'run.json'), evidence.get('run_sha256'), 'run metadata')
    run = _json(run_path)
    _equal_fields(run, dict(status='completed', purpose='calibration', method='fedavg', seed=seed, budget_per_client=budget, requests_per_client=budget, environment=environment, task_hash=task, source_hash=source, smoke=smoke), 'completed run')
    recorded = run.get('config')
    if not isinstance(recorded, dict):
        _fail('run configuration is missing')
    _equal_fields(recorded, dict(method='fedavg', purpose='calibration', seed=seed, environment=environment, smoke=smoke), 'recorded configuration')
    if run.get('configuration_hash') != stable_hash(recorded):
        _fail('recorded configuration checksum mismatch')
    for section in ('agent', 'federation'):
        if recorded.get(section) != config.get(section):
            _fail('calibration learning configuration differs: ' + section)
    excluded = {'requests_per_client', 'cpu_threads', 'checkpoint_every', 'resume_every'}
    if {k: v for k, v in recorded.get('training', {}).items() if k not in excluded} != {k: v for k, v in config['training'].items() if k not in excluded}:
        _fail('calibration learning schedule differs')
    try:
        if task_hash(recorded) != task or recorded['training']['requests_per_client'] != budget:
            _fail('recorded task or training budget differs')
    except (KeyError, TypeError):
        _fail('incomplete recorded task configuration')
    sessions_path = _checked_path(evidence.get('validation_sessions_path'), evidence.get('validation_sha256'), 'validation sessions')
    metrics_path = _checked_path(evidence.get('validation_metrics_path'), evidence.get('validation_metrics_sha256'), 'validation metrics')
    common = dict(environment=environment, seed=seed, purpose='calibration', task_hash=task, source_hash=source, split='validation', eval_kind='calibration', eval_manifest_hash=manifest['hash'])
    grouped = {method: {client: {} for client in range(clients)} for method in ('fedavg', 'random')}
    gamma = float(config['agent']['gamma'])
    horizon = int(config['env']['horizon'])
    for record in _records(sessions_path):
        method = record.get('method')
        if method not in grouped:
            _fail('session method must be FedAvg or Random')
        _equal_fields(record, dict(common, requests_per_client=budget if method == 'fedavg' else 0), 'session')
        client, episode = (record.get('client_id'), record.get('session'))
        if type(client) is not int or type(episode) is not int or client not in range(clients) or (episode not in range(count)):
            _fail('session client or index is outside the complete manifest')
        if episode in grouped[method][client]:
            _fail('duplicate method/client/session evidence')
        if record.get('initialization_seed') != manifest['clients'][client][episode]:
            _fail('session initialization differs from the common validation manifest')
        rewards = record.get('rewards')
        if not isinstance(rewards, list) or not 1 <= len(rewards) <= horizon:
            _fail('session must contain a nonempty bounded reward trajectory')
        if any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (not 0 <= value <= 1) for value in rewards)):
            _fail('invalid session reward')
        if record.get('length') != len(rewards):
            _fail('session length differs from its reward trajectory')
        if record.get('termination_reason') not in ('feedback_leave', 'horizon'):
            _fail('session has no valid termination reason')
        if record['termination_reason'] == 'horizon' and len(rewards) != horizon:
            _fail('horizon-terminated session is incomplete')
        _close(record.get('discounted_return'), sum((gamma ** index * value for index, value in enumerate(rewards))), 'session return')
        _close(record.get('reward_sum'), sum(rewards), 'session reward sum')
        grouped[method][client][episode] = rewards
    recomputed = {}
    for method in grouped:
        values = []
        for client in range(clients):
            episodes = grouped[method][client]
            if set(episodes) != set(range(count)):
                _fail(f'{method} client {client} lacks the complete {count}-session manifest')
            values.append(dict(client_metrics([episodes[index] for index in range(count)], gamma), client_id=client))
        recomputed[method] = dict(clients=values, macro=macro_metrics(values))
    seen = set()
    for record in _records(metrics_path):
        method = record.get('method')
        if method not in grouped or method in seen:
            _fail('metrics must contain exactly one FedAvg and one Random calibration record')
        seen.add(method)
        _equal_fields(record, dict(common, requests_per_client=budget if method == 'fedavg' else 0, eval_sessions=count, sessions_per_client=count), 'metrics')
        actual_clients = record.get('clients')
        if not isinstance(actual_clients, list) or len(actual_clients) != clients:
            _fail('metrics client coverage is incomplete')
        ids = [item.get('client_id') for item in actual_clients if isinstance(item, dict)]
        if len(ids) != clients or any((type(client) is not int for client in ids)) or set(ids) != set(range(clients)):
            _fail('metrics client identities are duplicated or incomplete')
        for actual in actual_clients:
            expected = recomputed[method]['clients'][actual['client_id']]
            for key in ('J', 'A', 'L', 'sessions', 'requests'):
                _close(actual.get(key), expected[key], f"{method} client {actual['client_id']} {key}")
        if not isinstance(record.get('macro'), dict):
            _fail('metrics macro summary is missing')
        for key in ('J', 'A', 'L'):
            _close(record['macro'].get(key), recomputed[method]['macro'][key], f'{method} macro {key}')
        _close(record.get('J_macro'), recomputed[method]['macro']['J'], f'{method} J_macro')
        by_client = record.get('J_by_client')
        if not isinstance(by_client, list) or len(by_client) != clients:
            _fail('metrics J_by_client is incomplete')
        for client, value in enumerate(by_client):
            _close(value, recomputed[method]['clients'][client]['J'], f'{method} J_by_client')
    if seen != set(grouped):
        _fail('both calibration metrics records are required')
    reference, random_return = (recomputed['fedavg']['macro']['J'], recomputed['random']['macro']['J'])
    _close(target.get('J_ref'), reference, 'target J_ref')
    _close(target.get('J_random'), random_return, 'target J_random')
    return dict(validated=True, J_ref=reference, J_random=random_return, session_counts={method: clients * count for method in grouped}, files={str(run_path): evidence['run_sha256'], str(sessions_path): evidence['validation_sha256'], str(metrics_path): evidence['validation_metrics_sha256']})
