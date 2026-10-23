from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

def empirical_wasserstein(x, y):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment, linprog
    from scipy.sparse import lil_matrix
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or (not len(x)) or (not len(y)):
        raise ValueError('Expected nonempty empirical distributions with matching dimensions')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite ground-truth samples')
    cost = cdist(x, y)
    n, m = cost.shape
    if n == m:
        i, j = linear_sum_assignment(cost)
        return float(cost[i, j].mean())
    constraints = lil_matrix((n + m, n * m))
    for i in range(n):
        constraints[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        constraints[n + j, j::m] = 1
    fit = linprog(cost.ravel(), A_eq=constraints.tocsr(), b_eq=np.r_[np.full(n, 1 / n), np.full(m, 1 / m)], bounds=(0, None), method='highs')
    if not fit.success:
        raise RuntimeError('Empirical Wasserstein solve failed: ' + fit.message)
    return float(fit.fun)

def evaluate_matching(latents, expected_rewards, next_samples, random_seed=None):
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(random_seed)
    pairs = []
    for k, z in enumerate(latents):
        for j, target in enumerate(latents):
            if j == k:
                continue
            matches = cdist(z, target).argmin(axis=1) if random_seed is None else rng.integers(len(target), size=len(z))
            reward_delta = np.abs(expected_rewards[k] - expected_rewards[j][matches]).mean()
            distances = []
            for p, q in enumerate(matches):
                for action in range(next_samples[k].shape[1]):
                    distances.append(empirical_wasserstein(next_samples[k][p, action], next_samples[j][q, action]))
            pairs.append(dict(source_client=k, target_client=j, reward_discrepancy=float(reward_delta), transition_discrepancy=float(np.mean(distances)), n_states=len(z), target_indices=matches.tolist(), reward_discrepancy_by_state_action=np.abs(expected_rewards[k] - expected_rewards[j][matches]).tolist(), transition_discrepancy_by_state_action=np.asarray(distances).reshape(len(z), -1).tolist()))
    if not pairs:
        raise ValueError('At least two clients required')
    return dict(reward_discrepancy=float(np.mean([p['reward_discrepancy'] for p in pairs])), transition_discrepancy=float(np.mean([p['transition_discrepancy'] for p in pairs])), client_pairs=pairs)

def _pool(config, env_factory, cache_path, states_per_client, actions, repeats, seed):
    from fedlpd.types import Observation
    from fedlpd.artifacts import fingerprint
    import inspect
    cache_path = Path(cache_path)
    package = Path(__file__).parents[1]
    source_files = sorted((package / 'envs').glob('*.py')) + [Path(__file__)]
    factory_file = inspect.getsourcefile(env_factory)
    if factory_file and Path(factory_file).is_file():
        source_files.append(Path(factory_file))
    source_hash = hashlib.sha256(''.join((fingerprint(p) for p in source_files)).encode()).hexdigest()
    identity = dict(config=config, states=states_per_client, actions=actions, repeats=repeats, seed=seed, version=2, source_hash=source_hash)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    cache_path.mkdir(parents=True, exist_ok=True)
    npz = cache_path / (digest + '.npz')
    metadata = cache_path / (digest + '.json')
    if npz.exists() and metadata.exists():
        recorded = json.loads(metadata.read_text(encoding='utf-8'))
        if recorded.get('identity') != identity or recorded.get('digest') != digest or recorded.get('npz_sha256') != fingerprint(npz):
            raise ValueError('Diagnostic truth cache identity/schema/content mismatch')
        data = dict(np.load(npz, allow_pickle=False))
        for k in range(int(config.get('num_clients', 5))):
            if data.get(f'reward_samples_{k}', np.empty(0)).shape != (states_per_client, actions, repeats):
                raise ValueError('Diagnostic truth cache reward schema mismatch')
            if data.get(f'session_seed_{k}', np.empty(0)).shape != (states_per_client,):
                raise ValueError('Diagnostic truth cache episode provenance mismatch')
    else:
        data = {}
        n = int(config.get('num_clients', 5))
        slate_size = int(config.get('slate_size', 5))
        first = env_factory(0, seed)
        catalog_size = len(first.catalog_features)
        action_rng = np.random.default_rng(np.random.SeedSequence([seed, 700]))
        slates = np.stack([action_rng.choice(catalog_size, slate_size, replace=False) for _ in range(actions)])
        reserved = np.unique(slates)
        data['slates'] = slates
        for k in range(n):
            env = env_factory(k, seed + 1009 * k)
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, 701]))
            histories, contexts, candidates, rewards, outcomes, reward_ses = ([], [], [], [], [], [])
            raw_rewards, session_seeds, episode_steps = ([], [], [])
            session_seed = int(rng.integers(2 ** 31))
            episode_step = 0
            obs = env.reset(session_seed=session_seed)
            for state_index in range(states_per_client):
                steps = int(rng.integers(0, min(8, int(config.get('horizon', 20)))))
                for _ in range(steps):
                    eligible = np.setdiff1d(obs.candidate_ids, reserved)
                    if len(eligible) < slate_size:
                        session_seed = int(rng.integers(2 ** 31))
                        episode_step = 0
                        obs = env.reset(session_seed=session_seed)
                        break
                    result = env.step(rng.choice(eligible, slate_size, replace=False))
                    episode_step += 1
                    obs = result.observation
                    if result.done:
                        session_seed = int(rng.integers(2 ** 31))
                        episode_step = 0
                        obs = env.reset(session_seed=session_seed)
                snapshot = env.diagnostic_state()
                if not np.isin(reserved, env.legal_items(snapshot)).all():
                    raise RuntimeError('Diagnostic pool lost common legal actions')
                histories.append(obs.history.copy())
                contexts.append(obs.context.copy())
                candidates.append(obs.candidate_ids.copy())
                session_seeds.append(session_seed)
                episode_steps.append(episode_step)
                state_rewards, state_outcomes, state_ses, state_raw = ([], [], [], [])
                for ai, slate in enumerate(slates):
                    seeds = np.random.SeedSequence([seed, k, state_index, ai, 702]).generate_state(repeats).tolist()
                    truth = env.diagnostic_rollout(snapshot, slate, seeds)
                    r = np.asarray(truth['reward_samples'], float)
                    state_raw.append(r)
                    state_rewards.append(r.mean())
                    state_ses.append(r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float('nan'))
                    state_outcomes.append(np.asarray(truth['next_state_samples'], np.float32))
                rewards.append(state_rewards)
                outcomes.append(state_outcomes)
                reward_ses.append(state_ses)
                raw_rewards.append(state_raw)
                session_seed = int(rng.integers(2 ** 31))
                episode_step = 0
                obs = env.reset(session_seed=session_seed)
            for key, value in [('history', histories), ('context', contexts), ('candidates', candidates), ('reward', rewards), ('next', outcomes), ('reward_se', reward_ses), ('reward_samples', raw_rewards), ('session_seed', session_seeds), ('episode_step', episode_steps)]:
                data[f'{key}_{k}'] = np.asarray(value)
        np.savez_compressed(npz, **data)
        metadata.write_text(json.dumps(dict(identity=identity, digest=digest, npz_sha256=fingerprint(npz), sampling='Independent held-out random histories; diagnostic slates excluded from historical exposure to ensure common legal actions.', normalization='Environment-defined physical ranges, including termination indicator; never fitted to tested representations.'), indent=2, default=str), encoding='utf-8')
    n = int(config.get('num_clients', 5))
    observations = [[Observation(h, c, ids) for h, c, ids in zip(data[f'history_{k}'], data[f'context_{k}'], data[f'candidates_{k}'])] for k in range(n)]
    return (observations, [data[f'reward_{k}'] for k in range(n)], [data[f'next_{k}'] for k in range(n)], data, digest)

def representation_diagnostics(latents, rewards):
    from scipy.spatial.distance import cdist
    joined = np.concatenate(latents)
    centered = joined - joined.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular ** 2
    ratio = eigen / eigen.sum() if eigen.sum() > 0 else np.zeros_like(eigen)
    positive = ratio[ratio > 0]
    local = []
    for z, r in zip(latents, rewards):
        if len(z) < 2:
            local.append(dict(neighbor_reward_discrepancy=None, all_pair_reward_discrepancy=None, tied_neighbor_fraction=None))
            continue
        distance = cdist(z, z)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.argmin(axis=1)
        discrepancy = np.abs(r - r[nearest]).mean()
        baseline = np.abs(r[:, None, :] - r[None, :, :]).mean(axis=2)
        local.append(dict(neighbor_reward_discrepancy=float(discrepancy), all_pair_reward_discrepancy=float(baseline[~np.eye(len(z), dtype=bool)].mean()), tied_neighbor_fraction=float((np.isclose(distance, distance.min(axis=1)[:, None]).sum(axis=1) > 1).mean())))
    return dict(effective_rank=float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0, pca_explained_variance_ratio=np.pad(ratio[:2], (0, max(0, 2 - len(ratio)))).tolist(), local_behavioral_distinguishability=local)
