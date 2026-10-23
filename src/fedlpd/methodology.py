from copy import deepcopy
import time
import numpy as np
import torch
from .costs import payload_descriptor
from .metric import InsufficientLocalData
from .coverage import stream, query_items
PAPER_METHODS = ('full', 'alignment_off', 'metric_off', 'coverage_off', 'distillation_off', 'random_probes', 'uniform_teacher', 'signature_off')

def default_settings():
    return dict(version='action-grounded-v1', reference_actions=8, reference_seed=22049, signature_horizon=3, calibration_quantile=0.95, signature_threshold=1.0, gw_weight=1.0, coverage_reward=1.0, matching_steps=30, support_threshold=0.05, match_threshold=0.1, support_sigma=1.0, eta=0.25, local_neighbors=4, recalibration_steps=20, recalibration_alignment_steps=20)

def validate_settings(config):
    c = config.get('methodology')
    if c is None:
        return
    if not isinstance(c, dict) or set(c) != set(default_settings()) or c['version'] not in ('action-grounded-v1', 'action-grounded-v2'):
        raise ValueError('Invalid methodology settings')
    from .config import _number, _integer
    for k in ('reference_actions', 'reference_seed', 'signature_horizon', 'matching_steps', 'local_neighbors', 'recalibration_steps', 'recalibration_alignment_steps'):
        _integer(c[k], 'methodology.' + k)
    for k in ('signature_threshold', 'gw_weight', 'coverage_reward', 'support_sigma'):
        _number(c[k], 'methodology.' + k, positive=True)
    for k in ('calibration_quantile', 'support_threshold', 'match_threshold', 'eta'):
        _number(c[k], 'methodology.' + k, minimum=0, maximum=1)
    if c['calibration_quantile'] <= 0 or c['calibration_quantile'] >= 1:
        raise ValueError('Calibration quantile must be strictly between 0 and 1')
    if config['agent'].get('critic_action_representation') != 'slate_mean_st':
        raise ValueError('The current methodology adaptation requires the shared slate critic')
    if config['method'] in ('global_probes', 'prediction_error_off', 'support_teacher', 'uniform_confidence', 'donor_filter_off', 'transfer_guard_off'):
        raise ValueError('Historical ablation has no new-methodology definition; use its historical configuration')
    if config['federation']['probe_items'] < config['env']['slate_size']:
        raise ValueError('Paper query candidates must support a complete legal slate')
    if c['version'] == 'action-grounded-v2' and config['method'] in PAPER_METHODS:
        if config['federation']['probe_selected'] >= config['federation']['anchors']:
            raise ValueError('v23 requires a probe budget smaller than the anchor count')

def supported_disagreement(responses, support):
    p = np.asarray(responses)
    mask = np.asarray(support, bool)
    scores = np.full(p.shape[1], -np.inf)
    for m in range(p.shape[1]):
        group = p[mask[:, m], m]
        if len(group) >= 2:
            mean = group.mean(0)
            scores[m] = np.mean(np.sum(group * np.log(np.maximum(group, 1e-30) / np.maximum(mean, 1e-30)), axis=-1))
    return scores

def select_anchors(eligible, scores, budget, random_selection, rng):
    eligible = np.asarray(eligible, dtype=np.int64)
    limit = min(len(eligible), budget)
    selected = eligible[np.argsort(-np.asarray(scores)[eligible], kind='stable')[:limit]]
    if random_selection and limit:
        selected = rng.choice(eligible, size=limit, replace=False)
    return (selected, dict(probe_budget=int(budget), selected_probe_count=int(limit), selection_budget_binding=bool(len(eligible) > budget)))

def fit_geometry(agent, local, targets, mass, config, steps, alignment=True):
    d = torch.as_tensor(local['summary']['distances'], device=agent.device, dtype=torch.float32)
    reliability = torch.as_tensor(local['summary']['reliability'], device=agent.device, dtype=torch.float32)
    empirical = torch.as_tensor(local['summary']['mass'], device=agent.device, dtype=torch.float32)
    pair_mass = empirical[:, None] * empirical[None, :]
    goal = torch.as_tensor(targets, device=agent.device, dtype=torch.float32)
    weight = torch.as_tensor(mass, device=agent.device, dtype=torch.float32)
    f = config['federation']
    last = {}
    has_metric = config['method'] != 'metric_off' and f['lambda_metric'] > 0 and bool(reliability.any())
    has_alignment = alignment and f['lambda_align'] > 0 and bool(weight.any())
    if not has_metric and (not has_alignment):
        return dict(metric_loss=0.0, alignment_loss=0.0, geometry_skipped=True)
    optimizer = agent.geometry_optimizer if agent.geometry_optimizer is not None else agent.encoder_optimizer
    for _ in range(steps):
        z = agent.encode_states(local['states'])
        metric = (pair_mass * reliability * (torch.cdist(z, z) - d).square()).sum()
        align = (weight * (z - goal).square().sum(-1)).sum() / weight.sum().clamp_min(1e-12)
        loss = f['lambda_metric'] * metric if config['method'] != 'metric_off' else z.sum() * 0
        loss += f['lambda_align'] * align if alignment else z.sum() * 0
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite paper geometry')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.adapter.parameters()) + list(agent.projector.parameters()), f.get('geometry_grad_clip', 10.0))
        optimizer.step()
        last = dict(metric_loss=float(metric.detach()), alignment_loss=float(align.detach()))
    return last

def aggregate_projectors(agents, weights, messages):
    states = [a.projector_state() for a in agents]
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    mean = {k: sum((float(w) * s[k] for w, s in zip(weights, states))) for k in states[0]}
    for i, (a, s) in enumerate(zip(agents, states)):
        messages.append(payload_descriptor('upload', f'client:{i}', 'server', 'paper_projector_upload', s))
        a.load_projector(mean)
        messages.append(payload_descriptor('download', 'server', f'client:{i}', 'paper_projector_download', mean))

@torch.no_grad()
def absolute_support(agent, states, anchors, sigma):
    a = torch.as_tensor(anchors, device=agent.device, dtype=torch.float32)
    total = torch.zeros(len(a), device=agent.device)
    for i in range(0, len(states), 256):
        z = agent.encode_states(states[i:i + 256])
        total += torch.exp(-torch.cdist(z, a).square() / (2 * sigma * sigma)).sum(0)
    return (total / len(states)).cpu().numpy()

def paper_round(agents, method, config, rng, round_index, state):
    from .methodology_metric import estimate, reference_slates
    from .methodology_alignment import match
    from .methodology_transfer import bellman_values, recalibrate_critic, local_transfer
    if state is None:
        raise ValueError('Paper protocol needs persistent state')
    begin = time.perf_counter()
    f = config['federation']
    c = config['methodology']
    messages = []
    log = dict(round=int(round_index), method=method, methodology_version=c['version'], skipped=False, messages=messages, completed_aggregation_count=0, distillation_update_count=0, bounds_certified=False, action_semantics='ordered Top-K reference slates; candidate-completion advantage proxy')
    if c['version'] == 'action-grounded-v2':
        log.update(teacher_weight_audit_schema=1, ablation_method=method, signature_grounding_enabled=method != 'signature_off', metric_loss_enabled=method != 'metric_off', alignment_loss_enabled=method != 'alignment_off', replay_support_filter_enabled=method != 'coverage_off', distillation_enabled=method != 'distillation_off', teacher_rule='uniform_supported_peers' if method == 'uniform_teacher' else 'recipient_advantage', selection_rule='random' if method == 'random_probes' else 'disagreement')

    def msg(direction, i, name, values):
        a, b = (f'client:{i}', 'server') if direction == 'upload' else ('server', f'client:{i}')
        messages.append(payload_descriptor(direction, a, b, name, values))
    try:
        refs = reference_slates(config, len(agents[0].item_features))
        locals = []
        summaries = []
        for i, a in enumerate(agents):
            msg('download', i, 'paper_reference_actions', {'reference_slates': refs})
            try:
                local = estimate(a, config, refs, stream(config['seed'], config['environment'], round_index, i, 'paper_estimation'))
            except InsufficientLocalData as exc:
                log.update(skipped=True, skip_reason=str(exc))
                return log
            locals.append(local)
            summaries.append(local['summary'])
            msg('upload', i, 'paper_behavior_summary', local['summary'])
        log['local_models'] = [l['diagnostics'] for l in locals]
        bank, gs, reports = match(summaries, config, agents[0].latent_dim, state.get('paper_bank'))
        log['partial_matching'] = reports
        if bank is None:
            log.update(skipped=True, skip_reason='no_reliable_action_signatures', selected_probes=[])
            return log
        state['paper_bank'] = bank
        state['round'] = int(round_index)
        anchors = bank['anchors']
        nu = np.full(len(anchors), 1 / len(anchors))
        targets = []
        masses = []
        confidence = []
        frozen = []
        for i, (a, l, g) in enumerate(zip(agents, locals, gs)):
            mass = g.sum(1)
            target = np.divide(g @ anchors, mass[:, None], out=np.zeros((len(mass), a.latent_dim)), where=mass[:, None] > 0)
            targets.append(target)
            masses.append(mass)
            confidence.append(np.clip(g.sum(0) / nu, 0, 1))
            frozen.append(bellman_values(a, l['fit_records'])[-1])
            msg('download', i, 'paper_alignment_targets', {'target': target, 'matched_mass': mass})
        log['geometry'] = [fit_geometry(a, l, t, m, config, f['aux_steps'], method != 'alignment_off') for a, l, t, m in zip(agents, locals, targets, masses)]
        samples = [len(a.replay) for a in agents]
        for i, n in enumerate(samples):
            msg('upload', i, 'paper_sample_weight', {'sample_weight': np.array([n], dtype=np.int64)})
        aggregate_projectors(agents, samples, messages)
        log['completed_aggregation_count'] = 1
        recal = []
        for a, l, t, m, y in zip(agents, locals, targets, masses, frozen):
            params = list(a.projector.parameters())
            flags = [p.requires_grad for p in params]
            for p in params:
                p.requires_grad_(False)
            try:
                fit_geometry(a, l, t, m, config, c['recalibration_alignment_steps'], method != 'alignment_off')
            finally:
                for p, v in zip(params, flags):
                    p.requires_grad_(v)
            recal.append(recalibrate_critic(a, l['fit_records'], y, c['recalibration_steps']))
        log['recalibration'] = recal
        log['geometry_schedule'] = 'align_aggregate_recalibrate'
        support = []
        for i, (a, l) in enumerate(zip(agents, locals)):
            msg('download', i, 'paper_anchor_queries', {'anchors': anchors})
            support.append(absolute_support(a, l['bank'], anchors, c['support_sigma']))
            msg('upload', i, 'paper_support', {'support': support[-1], 'match_confidence': confidence[i]})
        support = np.stack(support)
        confidence = np.stack(confidence)
        masks = (support >= c['support_threshold']) & (confidence >= c['match_threshold'])
        if method == 'coverage_off':
            masks = confidence >= c['match_threshold']
        eligible = np.flatnonzero(masks.sum(0) >= 2)
        ids = query_items(config, round_index, len(anchors), len(agents[0].item_features))
        responses = np.zeros((len(agents), len(anchors), ids.shape[1]), dtype=np.float64)
        for i, a in enumerate(agents):
            supported = eligible[masks[i, eligible]]
            if not len(supported):
                continue
            msg('download', i, 'paper_policy_query', {'anchors': anchors[supported], 'candidate_ids': ids[supported]})
            with torch.no_grad():
                responses[i, supported] = a.policy_response(anchors[supported], ids[supported]).cpu().numpy()
            msg('upload', i, 'paper_policy_response', {'probabilities': responses[i, supported]})
        scores = supported_disagreement(responses, masks)
        selected, selection = select_anchors(eligible, scores, f['probe_selected'], method == 'random_probes', rng)
        if c['version'] == 'action-grounded-v2':
            log.update(selection)
        log.update(candidate_probes=len(anchors), eligible_probes=len(eligible), selected_probes=anchors[selected].tolist(), selected_original_indices=selected.tolist(), selected_disagreement=scores[selected].tolist(), supporting_clients=masks.sum(0).tolist(), matched_mass=[float(g.sum()) for g in gs], support_by_client=support.tolist(), matching_confidence_by_client=confidence.tolist(), distillation=[], distillation_skipped=True)
        for i, (a, l) in enumerate(zip(agents, locals)):
            accepted_support = selected[masks[i, selected]]
            if not len(accepted_support) or method == 'distillation_off':
                continue
            z = anchors[accepted_support]
            pool = ids[accepted_support]
            p = responses[:, accepted_support]
            b = confidence[:, accepted_support]
            s = masks[:, accepted_support]
            msg('download', i, 'paper_peer_relay', {'anchors': z, 'candidate_ids': pool, 'peer_responses': p, 'match_confidence': b, 'support_mask': s})
            result = local_transfer(a, i, z, pool, p, b, s, l, config)
            log['distillation'].append(dict(client=i, **result))
            log['distillation_update_count'] += result['distillation_update_count']
        log['distillation_skipped'] = log['distillation_update_count'] == 0
        return log
    except Exception as exc:
        log.update(failed=True, failure_reason=str(exc))
        exc.federation_log = log
        raise
    finally:
        log.update(communication_bytes=sum((m['payload_nbytes'] for m in messages)), elapsed_seconds=time.perf_counter() - begin)
