from hashlib import sha256
import math
import numpy as np
from scipy.spatial.distance import cdist
DEFAULTS = dict(mode='legacy_kernel', shadow=False, neighbors=8, tail_alpha=0.05, sample_limit=4096, min_reference=64, min_calibration=128, min_check=128, min_episodes=2, degenerate_radius=1e-08)

def settings(config):
    return dict(DEFAULTS, **config.get('federation', config).get('coverage', {}))

def validate_settings(config):
    patch = config['federation'].get('coverage', {})
    if not isinstance(patch, dict) or not set(patch) <= DEFAULTS.keys():
        raise ValueError('Invalid federation.coverage keys')
    c = settings(config)
    if c['mode'] not in ('legacy_kernel', 'replay_knn_rank'):
        raise ValueError('Invalid coverage mode')
    if type(c['shadow']) is not bool:
        raise ValueError('coverage.shadow must be boolean')
    for key in ('neighbors', 'sample_limit', 'min_reference', 'min_calibration', 'min_check', 'min_episodes'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Invalid coverage.' + key)
    for key in ('tail_alpha', 'degenerate_radius'):
        v = c[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (v <= 0):
            raise ValueError('Invalid coverage.' + key)
    if c['tail_alpha'] >= 1:
        raise ValueError('coverage.tail_alpha must be below one')
    limits = (c['sample_limit'] // 2, c['sample_limit'] // 4, c['sample_limit'] // 4)
    if any((cap < c[key] for cap, key in zip(limits, ('min_reference', 'min_calibration', 'min_check')))):
        raise ValueError('coverage.sample_limit cannot meet split minimums')
    if c['neighbors'] > c['min_reference']:
        raise ValueError('coverage.neighbors exceeds reference minimum')
    if 1 / (c['min_calibration'] + 1) >= c['tail_alpha']:
        raise ValueError('coverage calibration cannot resolve tail_alpha')
    protocol = config['federation'].get('query_protocol', 'legacy')
    if protocol not in ('legacy', 'indexed_v1'):
        raise ValueError('Invalid federation.query_protocol')
    if c['mode'] == 'replay_knn_rank' and protocol != 'indexed_v1':
        raise ValueError('Rank coverage requires indexed_v1 queries')

def stream(seed, environment, round_index, client_id, purpose):
    label = f'coverage-v1|{seed}|{environment}|{round_index}|{client_id}|{purpose}'
    words = np.frombuffer(sha256(label.encode()).digest(), dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence(words))

def query_items(config, round_index, count, catalog_size):
    rng = stream(config.get('seed', 0), config.get('environment', 'unknown'), round_index, 0, 'query_items')
    size = min(int(config['federation'].get('probe_items', 50)), catalog_size)
    return np.stack([rng.choice(catalog_size, size=size, replace=False) for _ in range(count)])

def split_records(records, rng, limits):
    episodes = np.array(sorted({int(r['episode_id']) for r in records}), dtype=np.int64)
    episodes = rng.permutation(episodes)
    n = len(episodes)
    partitions = (episodes[:n // 2], episodes[n // 2:n // 2 + n // 4], episodes[n // 2 + n // 4:])
    groups = []
    for partition, limit in zip(partitions, limits):
        chosen = set(partition.tolist())
        idx = [i for i, r in enumerate(records) if int(r['episode_id']) in chosen]
        if len(idx) > limit:
            idx = rng.choice(idx, size=limit, replace=False).tolist()
        groups.append(sorted(idx))
    return groups

def kth_distance(reference, queries, neighbors):
    values = []
    for start in range(0, len(queries), 128):
        distances = cdist(queries[start:start + 128], reference, metric='euclidean')
        values.extend(np.partition(distances, neighbors - 1, axis=1)[:, neighbors - 1])
    return np.asarray(values, dtype=np.float64)

def rank_support(reference, calibration, check, probes, *, neighbors=8, alpha=0.05, degenerate_radius=1e-08):
    from .experiment_checks import quantiles
    reference, calibration, check, probes = [np.asarray(x, dtype=np.float64) for x in (reference, calibration, check, probes)]
    if any((x.ndim != 2 or x.shape[1] != reference.shape[1] for x in (reference, calibration, check, probes))):
        raise ValueError('Coverage representations must be compatible matrices')
    if any((not np.isfinite(x).all() for x in (reference, calibration, check, probes))):
        raise FloatingPointError('Nonfinite coverage representation')
    unique = np.unique(reference, axis=0)
    report = dict(reference_count=len(reference), unique_reference_count=len(unique), duplicate_reference_fraction=1 - len(unique) / max(1, len(reference)), calibration_count=len(calibration), check_count=len(check))

    def unavailable(reason):
        return (np.zeros(len(probes)), dict(report, status='unavailable', reason=reason))
    if len(unique) < neighbors or not len(calibration) or (not len(check)):
        return unavailable('insufficient_unique_reference_or_calibration')
    dcal = kth_distance(unique, calibration, neighbors)
    if np.quantile(dcal, 0.95) <= degenerate_radius:
        return unavailable('degenerate_calibration_radius')
    ordered = np.sort(dcal)

    def scores(distances):
        return (1 + len(ordered) - np.searchsorted(ordered, distances, side='left')) / (len(ordered) + 1)
    dp = kth_distance(unique, probes, neighbors)
    dc = kth_distance(unique, check, neighbors)
    center = unique.mean(0)
    radius = np.linalg.norm(unique - center, axis=1).max()
    directions = np.eye(unique.shape[1])[:min(16, unique.shape[1])]
    far = center + directions * (radius + float(dcal.max()) + max(1.0, radius) * 10)
    far_scores = scores(kth_distance(unique, far, neighbors))
    near_scores = scores(kth_distance(unique, unique[:min(32, len(unique))], neighbors))
    report.update(status='available', reason=None, calibration_distances=quantiles(dcal), check_distances=quantiles(dc), probe_distances=quantiles(dp), check_acceptance=float((scores(dc) > alpha).mean()), near_acceptance=float((near_scores > alpha).mean()), far_acceptance=float((far_scores > alpha).mean()), far_queries=len(far), nominal_check_acceptance=1 - alpha, empirical_not_conformal=True)
    return (scores(dp), report)

def local_coverage(agent, probes, config, round_index, client_id):
    import torch
    cfg = settings(config)
    limit = cfg['sample_limit']
    records = agent.replay.data
    rng = stream(config.get('seed', 0), config.get('environment', 'unknown'), round_index, client_id, 'split')
    groups = split_records(records, rng, (limit // 2, limit // 4, limit // 4))
    counts = [len(g) for g in groups]
    episodes = [len({records[i]['episode_id'] for i in g}) for g in groups]
    meta = dict(sample_counts=counts, episode_counts=episodes, tail_alpha=cfg['tail_alpha'], neighbors=cfg['neighbors'])
    if any((n < cfg[k] for n, k in zip(counts, ('min_reference', 'min_calibration', 'min_check')))) or min(episodes) < cfg['min_episodes']:
        return (np.zeros(len(probes), dtype=bool), dict(meta, status='unavailable', reason='insufficient_episode_split', probe_scores=[0.0] * len(probes)))
    encoded = []
    with torch.no_grad():
        for indices in groups:
            chunks = [agent.encode_states([records[i]['state'] for i in indices[start:start + 128]]).cpu().numpy() for start in range(0, len(indices), 128)]
            encoded.append(np.concatenate(chunks))
    p, report = rank_support(*encoded, probes, neighbors=cfg['neighbors'], alpha=cfg['tail_alpha'], degenerate_radius=cfg['degenerate_radius'])
    return (p > cfg['tail_alpha'], dict(meta, **report, probe_scores=p.tolist()))
