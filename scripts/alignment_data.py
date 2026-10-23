from pathlib import Path
import json
import numpy as np
import torch
from fedlpd.analysis.diagnostics import _pool, evaluate_matching, representation_diagnostics
from fedlpd.artifacts import fingerprint, atomic_json
from fedlpd.training import load_agents_from_checkpoint
from fedlpd.envs import make_env

def alignment_evidence(runs, output, phase, device='cpu'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in Path(runs).glob('*/run.json'):
        m = json.loads(p.read_text(encoding='utf-8'))
        if m['environment'] == 'recsim' and m['method'] in ('full', 'alignment_off'):
            paths.append((p, m))
    if {(m['method'], m['seed']) for _, m in paths} != {(m, s) for m in ('full', 'alignment_off') for s in (2026, 2027, 2028)} or len(paths) != 6:
        raise ValueError('Need six fixed RecSim alignment checkpoints')
    smoke = paths[0][1]['smoke']
    states, actions, repeats = (4, 2, 2) if smoke else (256, 16, 32)
    cache = output / 'truth'
    random_reference = None
    pool_identity = None
    records = []
    for path, meta in sorted(paths, key=lambda pair: (pair[1]['method'], pair[1]['seed'])):
        config = meta['config']
        step = config['training']['requests_per_client'] // (2 if phase == 'half' else 1)
        checkpoint = path.parent / 'checkpoints' / f'request-{step}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('Missing fixed diagnostic checkpoint: ' + str(checkpoint))
        if meta['smoke'] != smoke or meta['status'] not in ('paused', 'completed'):
            raise ValueError('Incompatible diagnostic run')
        agents, loaded = load_agents_from_checkpoint(path.parent, step, device)
        if loaded != config:
            raise ValueError('Diagnostic checkpoint config differs')
        env_config = dict(config['env'], smoke=smoke, num_clients=len(agents))

        def factory(k, s):
            return make_env('recsim', k, s, env_config, split='test', device=device, evaluation=True)
        observations, rewards, outcomes, data, digest = _pool(env_config, factory, cache, states, actions, repeats, 918273)
        if pool_identity is not None and digest != pool_identity:
            raise ValueError('Diagnostic pools differ')
        pool_identity = digest
        with torch.no_grad():
            latent = [a.encode_states(pool).cpu().numpy() for a, pool in zip(agents, observations)]
        if not all((np.isfinite(z).all() for z in latent)):
            raise ValueError('Nonfinite diagnostic latent')
        matching = evaluate_matching(latent, rewards, outcomes)
        if random_reference is None:
            random_reference = [evaluate_matching(latent, rewards, outcomes, 918373 + i) for i in range(5)]
        representation = representation_diagnostics(latent, rewards)
        joined = np.concatenate(latent)
        centered = joined - joined.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
        folder = output / f"{meta['method']}-seed{meta['seed']}"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / 'latent_coordinates.npz', coordinates=coords, reward_color=np.concatenate([r.mean(1) for r in rewards]), client_id=np.concatenate([np.full(len(z), k) for k, z in enumerate(latent)]), explained_variance_ratio=representation['pca_explained_variance_ratio'])
        record = dict(method=meta['method'], seed=meta['seed'], environment='recsim', step=step, task_hash=meta['task_hash'], source_hash=meta['source_hash'], checkpoint_sha256=fingerprint(checkpoint), pool_digest=digest, matching=matching, representation=representation, smoke=smoke)
        atomic_json(folder / 'diagnostics.json', record)
        records.append(record)
        print(f"Alignment data: {meta['method']} seed {meta['seed']} at {step}", flush=True)
    atomic_json(output / 'alignment_evidence.json', dict(records=records, random_matching=random_reference, pool_digest=pool_identity, sampling=dict(states_per_client=states, actions=actions, repeats=repeats, pool_seed=918273), interpretation='Read-only common-action one-step consistency, not long-term equivalence; PCA is fitted separately per model.'))
    return output
