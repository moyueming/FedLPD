from collections import Counter
import json
from pathlib import Path
import numpy as np

def _summary(sessions, horizon):
    lengths = np.asarray([s['length'] for s in sessions], dtype=int)
    returns = np.asarray([s['discounted_return'] for s in sessions], dtype=float)
    rewards = np.asarray([s['mean_reward'] for s in sessions], dtype=float)
    return {'sessions': len(sessions), 'length_mean': float(lengths.mean()), 'length_min': int(lengths.min()), 'length_max': int(lengths.max()), 'length_histogram': dict(sorted(Counter(map(int, lengths)).items())), 'horizon_fraction': float((lengths == horizon).mean()), 'one_request_fraction': float((lengths == 1).mean()), 'termination_reasons': dict(Counter((s['termination_reason'] for s in sessions))), 'discounted_return_mean': float(returns.mean()), 'discounted_return_std': float(returns.std(ddof=1)) if len(returns) > 1 else None, 'mean_session_reward': float(rewards.mean())}

def validate_environment(name, config, output_dir, sessions_per_client=100, seed=3026, data_dir=None, response_path=None, device='cpu', split='validation'):
    from . import make_env
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown audit split')
    if int(sessions_per_client) < 1:
        raise ValueError('sessions_per_client must be positive')
    cfg = dict(config.get('env', config))
    gamma = float(config.get('agent', {}).get('gamma', 0.95))
    records = []
    provenance = {}
    clients = {}
    for client in range(int(cfg.get('num_clients', 5))):
        env = make_env(name, client, seed, config, data_dir, split, response_path, device, evaluation=False)
        provenance[str(client)] = env.provenance
        action_rng = np.random.default_rng(np.random.SeedSequence([int(seed), client, 41169]))
        local = []
        for episode in range(int(sessions_per_client)):
            session_seed = int(np.random.SeedSequence([int(seed), client, episode, 41170]).generate_state(1)[0])
            obs = env.reset(session_seed=session_seed)
            exposed = set()
            total = 0.0
            rewards = []
            for request in range(env.horizon):
                candidate_ids = np.asarray(obs.candidate_ids)
                if len(candidate_ids) != env.candidate_size or len(set(candidate_ids.tolist())) != len(candidate_ids):
                    raise ValueError('Candidate size or uniqueness contract failed')
                if np.any(candidate_ids < 0) or np.any(candidate_ids >= len(env.catalog_features)) or set(candidate_ids) & exposed:
                    raise ValueError('Candidate legality contract failed')
                expected_history = 10 if client % 5 in (0, 3) else 0 if client % 5 == 2 else 20
                expected_context = env.catalog_features.shape[1] + 4 if expected_history == 0 else 2
                if obs.history.shape != (expected_history, env.catalog_features.shape[1] + 3) or obs.context.shape != (expected_context,):
                    raise ValueError('Client observation schema contract failed')
                if not np.isfinite(obs.history).all() or not np.isfinite(obs.context).all():
                    raise ValueError('Nonfinite visible observation')
                slate = action_rng.choice(candidate_ids, env.slate_size, replace=False)
                result = env.step(slate)
                exposed.update(slate.tolist())
                if set(result.info) - {'request', 'termination_reason'}:
                    raise ValueError('Privileged information in learner feedback')
                if not np.isfinite(result.reward) or not 0 <= result.reward <= 1:
                    raise ValueError('Reward contract failed')
                if set(result.observation.candidate_ids) & exposed:
                    raise ValueError('Exposed item in next candidates')
                total += gamma ** request * result.reward
                rewards.append(result.reward)
                obs = result.observation
                if result.done:
                    break
            if not result.done:
                raise ValueError('Session exceeded horizon')
            record = {'client_id': client, 'episode': episode, 'session_seed': session_seed, 'length': len(rewards), 'discounted_return': total, 'mean_reward': float(np.mean(rewards)), 'termination_reason': result.info['termination_reason']}
            records.append(record)
            local.append(record)
        clients[str(client)] = _summary(local, env.horizon)
    report = {'format_version': 1, 'purpose': 'environment_validation', 'environment': name, 'policy': 'random', 'split': split, 'seed': int(seed), 'gamma': gamma, 'publication_eligible': False, 'checks': {'legal_candidates': True, 'visible_information': True, 'bounded_reward': True, 'horizon': True}, 'all': _summary(records, env.horizon), 'clients': clients, 'provenance': provenance, 'interpretation': 'Descriptive pilot audit, not a formal policy comparison. Inspect session lengths and response quality; no automatic performance acceptance threshold and no tuning from FedLPD outcomes.'}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'environment_validation.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'environment_sessions.jsonl').write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in records)), encoding='utf-8')
    return path
