from __future__ import annotations
import time
import copy
import numpy as np
import torch
from scipy.special import softmax
from .metric import transport_residual, InsufficientLocalData, DegenerateGeometry
from .costs import payload_descriptor as message_descriptor
ALIASES = {'fedlpd': 'full', 'no_metric': 'metric_off', 'no_prediction_error': 'prediction_error_off', 'no_alignment': 'alignment_off', 'no_coverage': 'coverage_off', 'support_only_teacher': 'support_teacher', 'no_distillation': 'distillation_off'}

def method_options(method):
    method = {'direct_lpd': 'full', 'v14_reference': 'full', 'local_fixed': 'local'}.get(method, ALIASES.get(method, method))
    flags = {'metric': True, 'prediction_error': True, 'alignment': True, 'global_probes': False, 'coverage': True, 'random_selection': False, 'uniform_teacher': False, 'support_only': False, 'distillation': True, 'uniform_confidence': False, 'donor_filter': True}
    changes = {'metric_off': ('metric', False), 'prediction_error_off': ('prediction_error', False), 'alignment_off': ('alignment', False), 'global_probes': ('global_probes', True), 'coverage_off': ('coverage', False), 'random_probes': ('random_selection', True), 'uniform_teacher': ('uniform_teacher', True), 'support_teacher': ('support_only', True), 'distillation_off': ('distillation', False), 'uniform_confidence': ('uniform_confidence', True), 'donor_filter_off': ('donor_filter', False)}
    if method in changes:
        key, value = changes[method]
        flags[key] = value
    elif method not in ('full', 'local', 'random', 'fedavg', 'transfer_guard_off'):
        raise ValueError('Unknown method ' + method)
    return flags

def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = (p + q) / 2
    return 0.5 * np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(m, 1e-300)), axis=-1) + 0.5 * np.sum(q * np.log(np.maximum(q, 1e-300) / np.maximum(m, 1e-300)), axis=-1)

def personalized_teachers(responses, support, options, profiles=None, temperature=0.2, *, donor_masks=None, audit=None):
    p = np.asarray(responses, dtype=np.float64)
    s = np.asarray(support, dtype=np.float64)
    n, m, _ = p.shape
    if n < 2:
        raise ValueError('Personalized transfer needs at least two clients')
    if profiles is None:
        profiles = p
    profiles = np.asarray(profiles, dtype=np.float64)
    raw = np.zeros((n, n, m))
    for k in range(n):
        for j in range(n):
            if j != k:
                raw[k, j] = s[j] * np.exp(-js_divergence(profiles[k], profiles[j]) / temperature)
    confidence = raw.sum(1) / (n - 1)
    if audit is not None:
        audit['legacy_confidence'] = confidence.tolist()
    selected = raw.copy()
    for k in range(n):
        if options['uniform_teacher']:
            selected[k] = 1.0
            selected[k, k] = 0
        elif options['support_only']:
            selected[k] = s
            selected[k, k] = 0
    if donor_masks is not None:
        from .donor import checked_donors
        selected *= checked_donors(donor_masks, (n, m))[None, :, :]
    denom = selected.sum(1, keepdims=True)
    weights = np.divide(selected, denom, out=np.zeros_like(selected), where=denom > 1e-300)
    unsupported = denom[:, 0] <= 1e-300
    confidence[unsupported] = 0
    teachers = np.einsum('kjp,jpi->kpi', weights, p)
    if options['uniform_confidence']:
        for k in range(n):
            positive = confidence[k] > 0
            if positive.any():
                confidence[k, positive] = confidence[k, positive].mean()
    return (teachers, weights, confidence)

def average_projectors(agents, messages=None):
    states = [a.projector_state() for a in agents]
    mean = {key: torch.stack([s[key] for s in states]).mean(0) for key in states[0]}
    descriptors = []
    for i, state in enumerate(states):
        descriptors.append(message_descriptor('upload', f'client:{i}', 'server', 'projector_upload', state))
    for i, a in enumerate(agents):
        a.load_projector(mean)
        descriptors.append(message_descriptor('download', 'server', f'client:{i}', 'projector_download', mean))
    if messages is not None:
        messages.extend(descriptors)
    return sum((m['payload_nbytes'] for m in descriptors))

def _solve_gw(summary, cost, nu, cfg, diagnostics):
    import ot
    maximum = int(cfg.get('gw_iterations', 50))
    mass = np.asarray(summary['mass'], dtype=np.float64)
    distance = np.asarray(summary['distances'], dtype=np.float64)
    if mass.ndim != 1 or np.any(mass <= 0) or (not np.isfinite(mass).all()) or (abs(mass.sum() - 1) > 1e-08):
        raise ValueError('Invalid positive prototype mass')
    if distance.shape != (len(mass), len(mass)) or not np.isfinite(distance).all():
        raise RuntimeError('Nonfinite or malformed GW distance matrix')
    if not np.isfinite(cost).all():
        raise RuntimeError('Nonfinite anchor geometry')
    coupling, report = ot.gromov.gromov_wasserstein(distance, cost, mass, nu, loss_fun='square_loss', G0=np.outer(mass, nu), max_iter=maximum, tol_rel=float(cfg.get('gw_tol_rel', 1e-07)), tol_abs=float(cfg.get('gw_tol_abs', 1e-09)), log=True)
    residual = transport_residual(coupling, mass, nu, float(cfg.get('ot_marginal_tolerance', 1e-08)))
    objective = float(report['gw_dist'])
    if not np.isfinite(objective):
        raise RuntimeError('Nonfinite GW objective')
    iterations = max(0, len(report.get('loss', [])) - 1)
    diagnostics.append({'solver': 'POT.gromov_wasserstein', 'objective': objective, 'marginal_residual': residual, 'iterations': iterations, 'max_iterations': maximum, 'reached_iteration_limit': iterations >= maximum, 'tol_rel': float(cfg.get('gw_tol_rel', 1e-07)), 'tol_abs': float(cfg.get('gw_tol_abs', 1e-09))})
    return coupling

def solve_anchors(summaries, latent_dim, cfg, rng, previous=None, diagnostics=None):
    if diagnostics is None:
        diagnostics = []
    m = int(cfg.get('anchors', 16))
    nu = np.full(m, 1.0 / m)
    if m < 2 or not summaries:
        raise ValueError('At least two anchors and one summary required')
    if previous is None:
        anchors = rng.normal(size=(m, latent_dim))
        distance = np.linalg.norm(anchors[:, None] - anchors[None, :], axis=-1)
        anchors /= distance[np.triu_indices(m, 1)].mean()
    else:
        anchors = np.asarray(previous, dtype=np.float64).copy()
    if anchors.shape != (m, latent_dim) or not np.isfinite(anchors).all():
        raise RuntimeError('Invalid previous anchors')
    anchor = torch.tensor(anchors, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([anchor], lr=float(cfg.get('anchor_lr', 0.03)))
    iterations = int(cfg.get('anchor_iterations', 5))
    couplings = []
    for _ in range(iterations):
        c = torch.cdist(anchor, anchor).detach().numpy()
        couplings = [_solve_gw(s, c, nu, cfg, diagnostics) for s in summaries]
        for _ in range(int(cfg.get('anchor_gradient_steps', 10))):
            ca = torch.cdist(anchor, anchor)
            loss = anchor.sum() * 0
            for s, coupling in zip(summaries, couplings):
                d = torch.as_tensor(s['distances'], dtype=torch.float64)
                g = torch.as_tensor(coupling, dtype=torch.float64)
                mu = torch.as_tensor(s['mass'], dtype=torch.float64)
                loss = loss + (d.square() * mu[:, None] * mu[None, :]).sum() + (ca.square() / m ** 2).sum() - 2 * (g.T @ d @ g * ca).sum()
            optimizer.zero_grad()
            (loss / len(summaries)).backward()
            optimizer.step()
            with torch.no_grad():
                anchor.sub_(anchor.mean(0, keepdim=True))
    anchors = anchor.detach().numpy()
    c = np.linalg.norm(anchors[:, None] - anchors[None, :], axis=-1)
    couplings = [_solve_gw(s, c, nu, cfg, diagnostics) for s in summaries]
    return (anchors, couplings)

def make_probes(anchors, cfg, options, rng):
    count = int(cfg.get('probe_candidates', 64))
    m = len(anchors)
    if count < m:
        raise ValueError('probe_candidates must include every anchor')
    if m < 2:
        raise ValueError('At least two anchors required')
    distances = np.linalg.norm(anchors[:, None] - anchors[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind='stable')[:, :min(m - 1, int(cfg.get('probe_neighbors', 3)))]
    probes = list(anchors.copy())
    for _ in range(count - m):
        source = int(rng.integers(m))
        candidates = np.delete(np.arange(m), source) if options['global_probes'] else neighbors[source]
        target = int(rng.choice(candidates))
        beta = float(rng.uniform())
        probes.append((1 - beta) * anchors[source] + beta * anchors[target])
    return np.asarray(probes)

def federated_round(agents, method, config, rng, round_index, state=None):
    from .methodology import PAPER_METHODS, paper_round
    if config.get('methodology') and method in PAPER_METHODS:
        return paper_round(agents, method, config, rng, round_index, state)
    if method in ('fedhpd', 'fedslate'):
        from .baselines.output import output_round
        return output_round(agents, method, config, rng, round_index, state)
    if method in ('fedrag', 'gpfedrec', 'gp_uniform'):
        from .baselines import baseline_round
        return baseline_round(agents, method, config, round_index)
    if config.get('federation', {}).get('replay_bridge') and method not in ('local', 'local_fixed', 'random', 'fedavg', 'direct_lpd', 'v14_reference'):
        from .grounded_transfer import ReplayBridge
        agents = [ReplayBridge(a, round_index, i) for i, a in enumerate(agents)]
    partial = {}
    started = time.perf_counter()
    try:
        result = _federated_round(agents, method, config, rng, round_index, state, partial)
        if agents and hasattr(agents[0], 'bridge_audit'):
            result['replay_bridge'] = [a.bridge_audit for a in agents]
        return result
    except Exception as exc:
        partial['communication_bytes'] = sum((m['payload_nbytes'] for m in partial.get('messages', [])))
        partial['elapsed_seconds'] = time.perf_counter() - started
        partial['failed'] = True
        partial['failure_reason'] = str(exc)
        exc.federation_log = partial
        raise

def _federated_round(agents, method, config, rng, round_index, state, logs):
    if state is None:
        raise ValueError('Persistent federation state dictionary is required')
    method = ALIASES.get(method, method)
    options = method_options(method)
    cfg = config.get('federation', config)
    start = time.perf_counter()
    messages = []
    logs.update(round=int(round_index), method=method, skipped=False, communication_bytes=0, messages=messages, completed_aggregation_count=0, distillation_update_count=0)

    def finish():
        logs['communication_bytes'] = sum((m['payload_nbytes'] for m in messages))
        logs['elapsed_seconds'] = time.perf_counter() - start
        return logs

    def upload(i, kind, tensors):
        messages.append(message_descriptor('upload', f'client:{i}', 'server', kind, tensors))

    def download(i, kind, tensors):
        messages.append(message_descriptor('download', 'server', f'client:{i}', kind, tensors))
    diagnostic = config.get('diagnostics', {}).get('enabled', False)
    previous = []
    if diagnostic:
        from .experiment_checks import policy_snapshot, policy_change
        previous = [policy_snapshot(a) for a in agents]
        logs['policy_change_audit'] = []

    def observe_stage(stage):
        if diagnostic:
            current = [policy_snapshot(a) for a in agents]
            logs['policy_change_audit'].append(dict(stage=stage, clients=[policy_change(b, a) for b, a in zip(previous, current)]))
            previous[:] = current
    if method in ('local', 'random'):
        return finish()
    if method == 'fedavg':
        average_projectors(agents, messages)
        logs['completed_aggregation_count'] = 1
        observe_stage('projector_aggregation')
        return finish()
    summaries = []
    for i, a in enumerate(agents):
        try:
            summary = a.metric_summary(rng, options, cfg)
        except (InsufficientLocalData, DegenerateGeometry) as exc:
            logs.update(skipped=True, skip_reason=str(exc), skip_category=type(exc).__name__, local_models=[s['diagnostics'] for s in summaries])
            state['skipped_rounds'] = state.get('skipped_rounds', 0) + 1
            return finish()
        summaries.append(summary)
        upload(i, 'behavior_summary', {'distances': summary['distances'], 'mass': summary['mass']})
    logs['local_models'] = [s['diagnostics'] for s in summaries]
    anchor_started = time.perf_counter()
    gw = []
    anchors, couplings = solve_anchors(summaries, agents[0].latent_dim, cfg, rng, state.get('anchors'), diagnostics=gw)
    logs['anchor_seconds'] = time.perf_counter() - anchor_started
    logs['gw_diagnostics'] = gw
    state['anchors'] = anchors
    state['round'] = int(round_index)
    targets = [coupling @ anchors / summary['mass'][:, None] for summary, coupling in zip(summaries, couplings)]
    schedule = cfg.get('geometry_schedule', 'align_then_aggregate')
    logs['geometry_schedule'] = schedule
    coordinate_diagnostic = diagnostic and config.get('diagnostics', {}).get('coordinate_audit', False)
    preview = None
    if coordinate_diagnostic:
        from .coordinate_audit import array_digest, local_geometry, coupling_summary
        preview_rng = copy.deepcopy(rng)
        for a in agents:
            a.replay.sample(min(len(a.replay), int(cfg.get('occupancy_sample_limit', 4096))), preview_rng)
        preview = make_probes(anchors, cfg, options, preview_rng)
        logs['coordinate_identity'] = dict(anchors_sha256=array_digest(anchors), probes_sha256=array_digest(preview), targets_sha256=[array_digest(t) for t in targets], couplings_sha256=[array_digest(t) for t in couplings])
        logs['coupling_audit'] = [coupling_summary(t, s['mass']) for t, s in zip(couplings, summaries)]
        logs.update(coordinate_audit=[], diagnostic_coordinate_seconds=0.0, diagnostic_coordinate_bytes=0)

    def observe_coordinates(stage):
        if not coordinate_diagnostic:
            return
        began = time.perf_counter()
        clients = []
        masks = []
        for i, (a, target) in enumerate(zip(agents, targets)):
            mask = a.coverage_mask(preview, config, round_index, i)
            masks.append(mask)
            clients.append(dict(client=i, geometry=local_geometry(a, target, anchors), coverage=a._coverage_report))
            logs['diagnostic_coordinate_bytes'] += message_descriptor('download', 'server', f'client:{i}', 'diagnostic_coordinate_query', {'probes': preview})['payload_nbytes']
            logs['diagnostic_coordinate_bytes'] += message_descriptor('upload', f'client:{i}', 'server', 'diagnostic_coordinate_mask', {'mask': mask})['payload_nbytes']
        masks = np.stack(masks)
        required = int(np.ceil(float(cfg.get('coverage_fraction', 0.6)) * len(agents)))
        logs['coordinate_audit'].append(dict(stage=stage, clients=clients, support_masks=masks.tolist(), supporting_clients=masks.sum(0).tolist(), eligible_probes=int((masks.sum(0) >= required).sum())))
        logs['diagnostic_coordinate_seconds'] += time.perf_counter() - began

    def aggregate():
        average_projectors(agents, messages)
        logs['completed_aggregation_count'] = 1
        observe_stage('projector_aggregation')
        observe_coordinates('after_aggregation')

    def align():
        if coordinate_diagnostic and config['diagnostics'].get('fitting_audit', False):
            from .fitting_audit import fitting_audit
            began = time.perf_counter()
            logs['fitting_audit'] = [fitting_audit(a, t, anchors, preview, options, config, round_index, i) for i, (a, t) in enumerate(zip(agents, targets))]
            logs['diagnostic_fitting_seconds'] = time.perf_counter() - began
        geometry = []
        for i, (agent, target) in enumerate(zip(agents, targets)):
            download(i, 'alignment_targets', {'targets': target})
            geometry.append(agent.fit_geometry(target, options, cfg))
        logs['geometry'] = geometry
        observe_stage('geometry')
        observe_coordinates('after_geometry')
    observe_coordinates('before_geometry_pipeline')
    if schedule == 'aggregate_then_align':
        aggregate()
        align()
    elif schedule == 'align_then_aggregate':
        align()
        aggregate()
    else:
        raise ValueError('Invalid geometry_schedule')
    occupancy = []
    for i, a in enumerate(agents):
        download(i, 'anchor_broadcast', {'anchors': anchors})
        mass = a.anchor_occupancy(anchors, rng, cfg)
        occupancy.append(mass)
        upload(i, 'anchor_occupancy', {'occupancy': mass})
    occupancy = np.stack(occupancy)
    if diagnostic:
        logs['occupancy_proximity_audit'] = [a._occupancy_audit for a in agents]
    probes = make_probes(anchors, cfg, options, rng)
    if preview is not None and (not np.array_equal(preview, probes)):
        raise RuntimeError('Diagnostic preview differs from actual probes')
    squared = np.sum((probes[:, None] - anchors[None, :]) ** 2, axis=-1)
    support = occupancy @ np.exp(-squared / (2 * float(cfg.get('sigma_probe', 1.0)) ** 2)).T
    coverage = (support >= float(cfg.get('support_threshold', 0.05))).mean(0)
    from .experiment_checks import support_audit
    logs['coverage_audit'] = support_audit(occupancy, squared, float(cfg.get('sigma_probe', 1.0)), float(cfg.get('support_threshold', 0.05)), float(cfg.get('coverage_fraction', 0.6)))
    from .coverage import settings, query_items, stream
    gate = settings(config)
    indexed = cfg.get('query_protocol', 'legacy') == 'indexed_v1'
    legacy_eligible = np.flatnonzero(coverage >= float(cfg.get('coverage_fraction', 0.6)))
    gate_active = gate['mode'] == 'replay_knn_rank' and options['coverage']
    donor_requested = cfg.get('donor_filter', 'legacy') == 'replay_knn_rank'
    donor_active = donor_requested and options['donor_filter']
    masks_active = gate_active or donor_active
    rank_eligible = np.array([], dtype=np.int64)
    masks = None
    if masks_active or gate['shadow']:
        gate_started = time.perf_counter()
        masks = []
        local_reports = []
        shadow_messages = []
        for i, a in enumerate(agents):
            query = message_descriptor('download', 'server', f'client:{i}', 'coverage_query', {'probes': probes})
            (messages if masks_active else shadow_messages).append(query)
            mask = a.coverage_mask(probes, config, round_index, i)
            if mask.dtype != np.bool_ or mask.shape != (len(probes),):
                raise ValueError('Malformed client coverage mask')
            reply = message_descriptor('upload', f'client:{i}', 'server', 'coverage_mask', {'mask': mask})
            (messages if masks_active else shadow_messages).append(reply)
            masks.append(mask)
            if diagnostic or gate['shadow']:
                local_reports.append(a._coverage_report)
        masks = np.stack(masks)
        required = int(np.ceil(float(cfg.get('coverage_fraction', 0.6)) * len(agents)))
        rank_eligible = np.flatnonzero(masks.sum(0) >= required)
        logs['rank_coverage_audit'] = dict(mode=gate['mode'], applied=gate_active, required_clients=required, tail_alpha=gate['tail_alpha'], support_masks=masks.tolist(), supporting_clients=masks.sum(0).tolist(), eligible_indices=rank_eligible.tolist(), eligible_probes=len(rank_eligible), legacy_eligible_indices=legacy_eligible.tolist(), local_diagnostics=local_reports, empirical_not_conformal=True)
        elapsed = time.perf_counter() - gate_started
        logs['coverage_seconds'] = elapsed if masks_active else 0.0
        logs['diagnostic_coverage_seconds'] = 0.0 if masks_active else elapsed
        logs['diagnostic_coverage_bytes'] = sum((m['payload_nbytes'] for m in shadow_messages))
        logs['diagnostic_coverage_messages'] = shadow_messages
    eligible = (rank_eligible if gate_active else legacy_eligible) if options['coverage'] else np.arange(len(probes))
    all_ids = query_items(config, round_index, len(probes), len(agents[0].item_features)) if indexed else None
    if indexed and (diagnostic or gate['shadow']):
        logs['candidate_item_pools'] = all_ids.tolist()
    logs.update(candidate_probes=len(probes), eligible_probes=len(eligible), eligible_indices=eligible.tolist(), coverage_mode=gate['mode'], query_protocol=cfg.get('query_protocol', 'legacy'), occupancy=occupancy.tolist())
    if len(eligible) == 0:
        logs.update(selected_probes=[], selected_indices=[], selected_original_indices=[], probe_candidates=[], teacher_weights=[], probe_confidence=[], distillation=[], distillation_skipped=True)
        if donor_requested:
            from .donor import donor_audit
            n = len(agents)
            logs['donor_filter_audit'] = donor_audit(np.zeros((n, 0), bool), np.zeros((n, n, 0)), np.zeros((n, 0)), np.zeros((n, 0)), applied=donor_active)
        return finish()
    probes = probes[eligible]
    support = support[:, eligible]
    squared = squared[eligible]
    count = min(int(cfg.get('probe_items', 50)), len(agents[0].item_features))
    ids = all_ids[eligible] if indexed else np.stack([rng.choice(len(agents[0].item_features), size=count, replace=False) for _ in probes])
    response_list = []
    for i, a in enumerate(agents):
        download(i, 'probe_query', {'probes': probes, 'candidate_ids': ids})
        with torch.no_grad():
            response = a.policy_response(probes, ids).cpu().numpy()
        upload(i, 'policy_response', {'probabilities': response})
        response_list.append(response)
    responses = np.stack(response_list)
    mean = responses.mean(0)
    disagreement = np.mean(np.sum(responses * np.log(np.maximum(responses, 1e-30) / np.maximum(mean, 1e-30)), axis=-1), axis=0)
    limit = min(int(cfg.get('probe_selected', 16)), len(probes))
    selection_rng = stream(config.get('seed', 0), config.get('environment', 'unknown'), round_index, 0, 'selection') if indexed else rng
    chosen = selection_rng.choice(len(probes), size=limit, replace=False) if options['random_selection'] else np.argsort(-disagreement, kind='stable')[:limit]
    chosen = np.asarray(chosen, dtype=np.int64)
    probes = probes[chosen]
    support = support[:, chosen]
    ids = ids[chosen]
    responses = responses[:, chosen]
    rho = softmax(-squared[chosen] / (2 * float(cfg.get('sigma_profile', 1.0)) ** 2), axis=-1)
    profiles = occupancy[:, None, :] * rho[None, :, :]
    profiles /= np.maximum(profiles.sum(-1, keepdims=True), 1e-300)
    teacher_audit = {}
    selected_masks = masks[:, eligible[chosen]] if masks is not None else None
    teacher, weights, confidence = personalized_teachers(responses, support, options, profiles, float(cfg.get('transfer_temperature', 0.2)), donor_masks=selected_masks if donor_active else None, audit=teacher_audit if donor_requested else None)
    if donor_requested:
        from .donor import donor_audit
        logs['donor_filter_audit'] = donor_audit(selected_masks, weights, confidence, teacher_audit['legacy_confidence'], applied=donor_active)
    distillation = []
    if options['distillation']:
        for i, a in enumerate(agents):
            download(i, 'personalized_teacher', {'selected_indices': chosen, 'teacher': teacher[i], 'confidence': confidence[i]})
            result = a.distill(probes, ids, teacher[i], confidence[i], cfg)
            distillation.append(result)
            logs['distillation_update_count'] += int(result['distillation_update_count'])
    observe_stage('distillation')
    logs.update(selected_probes=probes.tolist(), selected_indices=chosen.tolist(), eligible_indices=eligible.tolist(), selected_original_indices=eligible[chosen].tolist(), probe_candidates=ids.tolist(), teacher_weights=weights.tolist(), probe_confidence=confidence.tolist(), selected_disagreement=disagreement[chosen].tolist(), distillation=distillation, distillation_skipped=logs['distillation_update_count'] == 0)
    return finish()
