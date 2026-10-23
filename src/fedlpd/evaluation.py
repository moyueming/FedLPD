from contextlib import contextmanager
import random
import time
import numpy as np
import torch
from .config import stable_hash, task_hash, scheduled_round
from .metrics import client_metrics, macro_metrics

def initialization_manifest(config, split, sessions=None):
    if split not in ('test', 'validation'):
        raise ValueError('Evaluation split must be test or validation')
    cfg = config['evaluation']
    base = cfg['seed'] if split == 'test' else cfg['validation_seed']
    count = int(sessions or (cfg['final_sessions'] if split == 'test' else max(cfg['validation_sessions'], cfg['calibration_sessions'])))
    namespace = 32343 if split == 'test' else 41245
    clients = [[int(np.random.SeedSequence([base, namespace, k, e]).generate_state(1)[0]) for e in range(count)] for k in range(config['env']['num_clients'])]
    value = dict(schema_version=2, split=split, task_hash=task_hash(config), seed=base, clients=clients)
    return dict(value, hash=stable_hash(value))

@contextmanager
def isolated_inference(agents, device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    modes = []
    for agent in agents:
        for value in vars(agent).values():
            if isinstance(value, torch.nn.Module):
                modes.append((value, value.training))
                value.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for model, mode in modes:
            model.train(mode)

def evaluate(agents, config, requests_per_client, eval_kind, data_dir=None, response_path=None, device='cpu', logs=None, *, split='test', manifest=None, sessions=None, completed_aggregation_count=0, target=None):
    from .envs import make_env
    cfg = config['evaluation']
    sessions = int(sessions or (cfg['validation_sessions'] if split == 'validation' else cfg['curve_sessions'] if eval_kind == 'curve' else cfg['final_sessions']))
    manifest = manifest or initialization_manifest(config, split, max(sessions, cfg['final_sessions'] if split == 'test' else cfg['calibration_sessions']))
    if manifest['split'] != split or manifest['task_hash'] != task_hash(config):
        raise ValueError('Evaluation manifest belongs to different split/task')
    check = {k: v for k, v in manifest.items() if k != 'hash'}
    if stable_hash(check) != manifest['hash']:
        raise ValueError('Evaluation manifest checksum mismatch')
    from .protocol import source_identity
    identity = dict(environment=config['environment'], method=config['method'], seed=config['seed'], purpose=config.get('purpose', 'formal'), task_hash=task_hash(config), source_hash=config.get('assets', {}).get('implementation_sha256') or source_identity())
    start = time.perf_counter()
    metrics = []
    with isolated_inference(agents, device):
        for k in range(config['env']['num_clients']):
            if len(manifest['clients'][k]) < sessions:
                raise ValueError('Evaluation manifest lacks sessions')
            ev = make_env(config['environment'], k, manifest['seed'] + k, config, data_dir=data_dir, split=split, response_path=response_path, device=device)
            rewards = []
            for e, session_seed in enumerate(manifest['clients'][k][:sessions]):
                obs = ev.reset(session_seed=session_seed)
                rng = np.random.default_rng(np.random.SeedSequence([session_seed, 42, config['seed'] if config['method'] == 'random' else 0]))
                values = []
                reason = None
                for _ in range(config['env']['horizon']):
                    slate = rng.choice(obs.candidate_ids, config['env']['slate_size'], replace=False) if config['method'] == 'random' else agents[k].act(obs, rng, explore=False)[0]
                    result = ev.step(slate)
                    values.append(float(result.reward))
                    obs = result.observation
                    reason = result.info.get('termination_reason', result.info.get('terminal_reason'))
                    if result.done:
                        break
                rewards.append(values)
                if logs:
                    logs.write('evaluation_sessions.jsonl', dict(identity, schema_version=2, event_id=f'{split}:{requests_per_client}:{eval_kind}:{k}:{e}', split=split, requests_per_client=requests_per_client, eval_kind=eval_kind, client_id=k, session=e, initialization_seed=session_seed, eval_manifest_hash=manifest['hash'], rewards=values, discounted_return=sum((config['agent']['gamma'] ** i * r for i, r in enumerate(values))), reward_sum=sum(values), length=len(values), termination_reason=reason))
            item = client_metrics(rewards, config['agent']['gamma'])
            item['client_id'] = k
            metrics.append(item)
    record = dict(identity, split=split, requests_per_client=requests_per_client, eval_kind=eval_kind, eval_sessions=sessions, sessions_per_client=sessions, clients=metrics, macro=macro_metrics(metrics), eval_manifest_hash=manifest['hash'], scheduled_round_index=scheduled_round(config, requests_per_client), completed_aggregation_count=completed_aggregation_count, evaluation_seconds=time.perf_counter() - start)
    if split == 'validation':
        record.update(J_by_client=[x['J'] for x in metrics], J_macro=record['macro']['J'], J_star=target, reached=target is not None and record['macro']['J'] >= target, validation_seconds=record['evaluation_seconds'])
    if logs:
        logs.write('metrics.jsonl', record)
    return record
