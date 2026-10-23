from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
from .artifacts import atomic_json
from .config import validate_config

def quantiles(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(a, [0, 0.1, 0.5, 0.9, 1]))))

def policy_snapshot(agent):
    import torch
    records = agent.replay.data[:16]
    if not records:
        return None
    with torch.no_grad():
        ids = np.stack([r['candidate_ids'] for r in records])
        z = agent.encode_states([r['state'] for r in records])
        p = agent.policy_response(z, ids).cpu().numpy()
    return dict(z=z.cpu().numpy(), p=p, top=np.argsort(-p, axis=1, kind='stable')[:, :agent.slate_size])

def policy_change(before, after):
    if before is None or after is None:
        return None
    p, q = (before['p'], after['p'])
    return dict(states=len(p), latent_mean_l2=float(np.linalg.norm(before['z'] - after['z'], axis=1).mean()), latent_norm_before=quantiles(np.linalg.norm(before['z'], axis=1)), latent_norm_after=quantiles(np.linalg.norm(after['z'], axis=1)), mean_kl=float(np.mean(np.sum(p * np.log(np.maximum(p, 1e-30) / np.maximum(q, 1e-30)), axis=1))), top5_overlap=float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(before['top'], after['top'])])))

def support_audit(occupancy, squared, sigma, threshold, fraction):
    occupancy = np.asarray(occupancy, dtype=float)
    squared = np.asarray(squared, dtype=float)
    support = occupancy @ np.exp(-squared / (2 * sigma ** 2)).T
    required = math.ceil(fraction * len(occupancy))
    supporting = (support >= threshold).sum(0)
    ordered = np.sort(support, axis=0)[::-1]
    sensitivity = []
    for scale in (0.5, 1.0, 2.0):
        alt = occupancy @ np.exp(-squared / (2 * (sigma * scale) ** 2)).T
        for cut in sorted({threshold / 2, threshold, min(1.0, threshold * 2)}):
            sensitivity.append(dict(sigma_multiplier=scale, threshold=cut, eligible=int(((alt >= cut).sum(0) >= required).sum())))
    return dict(candidate_probes=len(squared), required_clients=required, supporting_clients=supporting.tolist(), eligible_probes=int((supporting >= required).sum()), support_quantiles=quantiles(support), distance_quantiles=quantiles(np.sqrt(squared)), boundary_support=ordered[max(0, required - 1)].tolist(), third_highest_support=ordered[2].tolist() if len(ordered) >= 3 else None, support_by_client_probe=support.tolist(), sigma_probe=float(sigma), support_threshold=float(threshold), coverage_fraction=float(fraction), sensitivity=sensitivity, interpretation='Descriptive fixed-grid sensitivity, not a parameter selection rule.')

def catalog_audit(frame, train, counts, config, source_files, removed):
    selected = frame.loc[train]
    ordered = sorted(counts.index, key=lambda x: (-int(counts[x]), int(x)))
    capacities = []
    for size in sorted({int(config['catalog_size']), 2000, 5000, 10000, 20000, 50000, 100000}):
        ids = ordered[:size]
        mask = selected.video_id.isin(ids)
        freq = counts.loc[ids].to_numpy()
        capacities.append(dict(requested_size=size, actual_size=len(ids), train_target_rows=int(mask.sum()), train_interaction_coverage=float(mask.mean()), train_users_with_targets=int(selected.loc[mask, 'user_id'].nunique()), item_train_count_quantiles=quantiles(freq)))
    mass = np.cumsum([int(counts[x]) for x in ordered]) / len(selected)
    targets = {str(target): int(np.searchsorted(mass, target) + 1) if len(mass) and mass[-1] >= target else None for target in (0.25, 0.5, 0.8)}
    return dict(schema_version=1, purpose='catalog_capacity_audit', decision_split='train', source_files=source_files, removed_exact_duplicates=removed, train_rows=len(selected), train_users=int(selected.user_id.nunique()), train_unique_items=int(selected.video_id.nunique()), eligible_items=len(ordered), min_item_count=int(config['min_item_count']), capacities=capacities, items_for_train_coverage=targets, interpretation='Coverage is descriptive. Larger catalogs need response-quality and runtime validation; no future split is used to choose a capacity.')

def diagnostic_matrix(base, profile='learning'):
    from .config import build_matrix
    if profile not in ('learning', 'quick'):
        raise ValueError('Unknown pilot profile')
    jobs = [j for j in build_matrix(base) if j['seed'] == 2026]
    if profile == 'learning':
        jobs = [j for j in jobs if j['environment'] == 'kuaisim' and j['method'] in ('full', 'local', 'fedavg', 'coverage_off')]
    for j in jobs:
        j.update(purpose='pilot', phase='learning_diagnostic' if profile == 'learning' else 'quick_paths', seed=base['cost']['pilot_seed'])
        if not j['smoke']:
            j['training'].update(requests_per_client=4000 if profile == 'learning' else 1200, warmup=2000 if profile == 'learning' else 800, communication_every=1000 if profile == 'learning' else 200, checkpoint_every=4000 if profile == 'learning' else 1200, resume_every=1000 if profile == 'learning' else 600)
        j['evaluation'].update(every=j['training']['requests_per_client'], curve_sessions=5, final_sessions=5, validation_sessions=100 if profile == 'learning' and (not j['smoke']) else 5)
        if j['smoke']:
            j['evaluation']['calibration_sessions'] = max(5, j['evaluation']['calibration_sessions'])
        j['diagnostics'] = {'enabled': profile == 'learning', 'paired_federation': profile == 'learning'}
        validate_config(j)
    return jobs

def paired_returns(before, after):

    def keyed(rows):
        result = {}
        for r in rows:
            k = (r['client_id'], r['session'])
            if k in result:
                raise ValueError('Duplicate paired session')
            result[k] = r
        return result
    a, b = (keyed(before), keyed(after))
    if not a or a.keys() != b.keys():
        raise ValueError('Paired sessions incomplete')
    diffs = {}
    for key in sorted(a):
        if a[key]['initialization_seed'] != b[key]['initialization_seed']:
            raise ValueError('Paired initialization mismatch')
        diffs.setdefault(key[0], []).append(b[key]['discounted_return'] - a[key]['discounted_return'])
    rng = np.random.default_rng(90327)
    boot = np.zeros(2000)
    clients = []
    for client, values in diffs.items():
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite paired returns')
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
        boot += means / len(diffs)
        clients.append(dict(client_id=client, n=len(values), difference=float(values.mean()), conditional_session_interval_95=np.quantile(means, [0.025, 0.975]).tolist()))
    return dict(macro_difference=float(np.mean([np.mean(v) for v in diffs.values()])), clients=clients, conditional_session_interval_95=np.quantile(boot, [0.025, 0.975]).tolist(), limitation='Paired session bootstrap conditional on this trained seed. Not uncertainty across training seeds and not proof of causality.')

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def diagnostic_report(root, output):
    records = {}
    result = {'runs': [], 'comparisons': [], 'publication_eligible': False}
    for path in sorted(Path(root).glob('*/run.json')):
        r = json.loads(path.read_text())
        folder = path.parent
        if r.get('phase') != 'learning_diagnostic' or r.get('status') != 'completed':
            continue
        sessions = [x for x in read_rows(folder / 'evaluation_sessions.jsonl') if x['split'] == 'validation']
        groups = {}
        for s in sessions:
            groups.setdefault((s['requests_per_client'], s['eval_kind']), []).append(s)
        metrics = [x for x in read_rows(folder / 'metrics.jsonl') if x['split'] == 'validation']
        rounds = read_rows(folder / 'rounds.jsonl') if (folder / 'rounds.jsonl').exists() else []
        effects = []
        for (step, kind), v in groups.items():
            if kind == 'pre_federation' and (step, 'post_federation') in groups:
                effects.append(dict(step=step, **paired_returns(v, groups[step, 'post_federation'])))
        records[r['environment'], r['method']] = (r, groups)
        result['runs'].append(dict(environment=r['environment'], method=r['method'], seed=r['seed'], minutes=r['wall_seconds_this_invocation'] / 60, validation=[dict(step=m['requests_per_client'], stage=m['eval_kind'], J=m['macro']['J'], clients=m['clients']) for m in metrics], immediate_federation_effects=effects, learning=read_rows(folder / 'learning.jsonl') if (folder / 'learning.jsonl').exists() else [], coverage=[q.get('coverage_audit') for q in rounds], stage_policy_changes=[q.get('policy_change_audit') for q in rounds]))
    if not records:
        raise ValueError('No completed learning diagnostic runs')
    for (env, method), (r, groups) in records.items():
        if method == 'local' or (env, 'local') not in records:
            continue
        local, lg = records[env, 'local']
        for key in ('task_hash', 'backbone_hash', 'evaluation_hash', 'seed', 'source_hash'):
            if r[key] != local[key]:
                raise ValueError('Incompatible comparison ' + key)
        step = r['requests_per_client']
        right = groups.get((step, 'post_federation'), groups.get((step, 'validation')))
        left = lg.get((step, 'validation'))
        result['comparisons'].append(dict(environment=env, method=method, reference='local', step=step, **paired_returns(left, right)))
    atomic_json(output, result)
    return Path(output)

def migration_check(config, output, data_dir=None, response_path=None, device='cpu'):
    import torch
    from .protocol import source_identity, dependency_manifest
    from .resources import resource_snapshot
    from .training import _assets
    from .envs import make_env
    if Path(output).exists():
        raise FileExistsError('Choose a new migration report')
    validate_config(config)
    if not config['smoke']:
        from .data import validate_data, FrozenResponse
        if not data_dir or not response_path:
            raise ValueError('Migration check needs prepared data and response assets')
        validate_data(data_dir)
        FrozenResponse(data_dir, response_path, device)
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
        x = torch.ones((64, 64), device=device, requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        ok = bool(torch.isfinite(loss) and torch.isfinite(x.grad).all())
    if not ok:
        raise ValueError('Device forward/backward produced nonfinite values')
    environments = []
    for name in ('kuaisim', 'recsim'):
        from .config import environment_config
        c = environment_config(config, name, concrete=True)
        for k in range(c['env']['num_clients']):
            env = make_env(name, k, 3026, c, data_dir=data_dir if name == 'kuaisim' else None, response_path=response_path if name == 'kuaisim' else None, device=device, split='validation')
            obs = env.reset(session_seed=918273 + k)
            result = env.step(obs.candidate_ids[:c['env']['slate_size']])
            if not np.isfinite(result.reward):
                raise ValueError('Nonfinite environment reward')
        environments.append(dict(environment=name, task_assets=_assets(c, data_dir if name == 'kuaisim' else None, response_path if name == 'kuaisim' else None)['assets']))
    atomic_json(output, dict(status='passed', backward_finite=True, source_hash=source_identity(), dependencies=dependency_manifest(), resources=resource_snapshot(device, config['training']['cpu_threads']), environments=environments, publication_eligible=False, next_step='Freeze final protocol on destination paths after independent target calibration; do not reuse absolute-path freeze from another host.'))
    return Path(output)
