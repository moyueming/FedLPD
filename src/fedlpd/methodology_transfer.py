import numpy as np
import torch
from torch.nn import functional as F

def advantage_target(responses, client, q, confidence, support, epsilon_q, epsilon_pi, qmax, eta, *, uniform=False, record_weights=False):
    p = np.asarray(responses, float)
    q = np.asarray(q, float)
    confidence = np.asarray(confidence, float)
    support = np.asarray(support, bool)
    if p.ndim != 3 or q.shape != p.shape[1:] or confidence.shape != p.shape[:2] or (support.shape != p.shape[:2]):
        raise ValueError('Invalid advantage inputs')
    if not 0 <= eta <= 1 or qmax <= 0:
        raise ValueError('Invalid conservative target settings')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p < 0):
        raise ValueError('Nonfinite policy or Q')
    penalty = 2 * np.asarray(epsilon_q) + qmax * np.asarray(epsilon_pi)
    if np.any(penalty < 0) or not np.isfinite(penalty).all():
        raise ValueError('Invalid error estimates')
    advantage = np.einsum('nmi,mi->nm', p - p[client], q)
    delta = advantage - penalty[None]
    allowed = support & support[client][None]
    allowed[client] = False
    raw = confidence * np.maximum(delta, 0) * allowed
    if uniform:
        raw = (raw > 0).astype(float)
    den = raw.sum(0)
    accept = (den > 0) & (eta > 0)
    weights = np.divide(raw, den[None], out=np.zeros_like(raw), where=den[None] > 0)
    teacher = np.einsum('nm,nmi->mi', weights, p)
    target = p[client].copy()
    target[accept] = (1 - eta) * p[client, accept] + eta * teacher[accept]
    audit = dict(accepted_peer_counts=(raw > 0).sum(0).tolist(), accepted_anchors=int(accept.sum()), rejected_anchors=int((~accept).sum()), eta=float(eta), bounds_certified=False, eligible_peer_pairs=int(allowed.sum()), positive_before_penalty=int(((advantage > 0) & allowed).sum()), positive_after_penalty=int(((delta > 0) & allowed).sum()), mean_penalty=float(penalty.mean()), maximum_raw_advantage=float(advantage[allowed].max()) if allowed.any() else None, maximum_conservative_advantage=float(delta[allowed].max()) if allowed.any() else None, interpretation='empirical slate-completion Q advantage, not certified single-item improvement')
    if record_weights:
        audit.update(teacher_weight_sum=weights[:, accept].sum(1).tolist(), teacher_weight_count=int(accept.sum()))
    return (target, accept, audit)

@torch.no_grad()
def bellman_values(agent, records):
    states = [r['state'] for r in records]
    next_states = [r['next_state'] for r in records]
    z = agent.encode_states(states)
    nz = agent.encode_states(next_states)
    actions = agent.features[torch.as_tensor(np.stack([r['slate'] for r in records]), device=agent.device)].mean(1)
    q1, q2 = agent.critic(z, actions)
    target = torch.tensor([r['reward'] for r in records], device=agent.device, dtype=torch.float32)[:, None]
    active = [i for i, r in enumerate(records) if not r['done']]
    if active:
        slate = agent.slate_action(agent.actor(nz[active]), [records[i]['next_candidate_ids'] for i in active])
        v1, v2 = agent.critic(nz[active], slate)
        target[active] += agent.gamma * torch.minimum(v1, v2)
    return (z, actions, q1, q2, target)

def recalibrate_critic(agent, records, frozen_targets, steps):
    with torch.no_grad():
        z = agent.encode_states([r['state'] for r in records])
        action = agent.features[torch.as_tensor(np.stack([r['slate'] for r in records]), device=agent.device)].mean(1)
    target = frozen_targets.detach()
    loss = z.sum() * 0
    for _ in range(steps):
        q1, q2 = agent.critic(z, action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite recalibration')
        agent.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
        agent.critic_optimizer.step()
    agent.target_adapter.load_state_dict(agent.adapter.state_dict())
    agent.target_projector.load_state_dict(agent.projector.state_dict())
    agent.target_critic.load_state_dict(agent.critic.state_dict())
    return dict(critic_recalibration_steps=steps, critic_recalibration_loss=float(loss.detach()))

def candidate_completions(ids, probabilities, slate_size):
    ids = np.asarray(ids)
    p = np.asarray(probabilities)
    if len(ids) < slate_size or len(set(ids.tolist())) != len(ids):
        raise ValueError('Insufficient distinct query items')
    order = ids[np.argsort(-p, kind='stable')]
    return np.stack([np.r_[i, order[order != i][:slate_size - 1]] for i in ids])

@torch.no_grad()
def private_values(agent, anchors, ids, old, local, config):
    c = config['methodology']
    device = agent.device
    qmax = max(abs(config['federation'].get('reward_min', 0.0)), abs(config['federation'].get('reward_max', 1.0))) / (1 - agent.gamma)
    _, _, c1, c2, target = bellman_values(agent, local['calibration'])
    td = torch.maximum((c1 - target).abs(), (c2 - target).abs()).flatten()
    calibrated = float(torch.quantile(td, c['calibration_quantile']))
    states = local['bank']
    bank = torch.cat([agent.encode_states(states[i:i + 256]) for i in range(0, len(states), 256)])
    anchors = torch.as_tensor(anchors, device=device, dtype=torch.float32)
    near = torch.argsort(torch.cdist(anchors, bank), dim=1, stable=True)[:, :min(c['local_neighbors'], len(bank))]
    values = []
    eq = []
    ep = []
    for m, (z, pool, prob) in enumerate(zip(anchors, ids, old)):
        slates = candidate_completions(pool, prob, agent.slate_size)
        action = agent.features[torch.as_tensor(slates, device=device)].mean(1)
        q1, q2 = agent.critic(z[None].expand(len(action), -1), action)
        q = torch.minimum(q1, q2).flatten().clamp(-qmax, qmax)
        neighbors = bank[near[m]]
        k = len(neighbors)
        nq1, nq2 = agent.critic(neighbors.repeat_interleave(len(pool), 0), action.repeat(k, 1))
        nq = torch.minimum(nq1, nq2).reshape(k, -1).clamp(-qmax, qmax)
        representation_error = float((nq - q[None]).abs().max())
        epsq = calibrated + float((q1 - q2).abs().max()) / 2 + representation_error
        response = agent.policy_response(neighbors, pool)
        epspi = float((response - torch.as_tensor(prob, device=device)).abs().sum(-1).max())
        values.append(q.cpu().numpy())
        eq.append(epsq)
        ep.append(epspi)
    return (np.stack(values), np.array(eq), np.array(ep), qmax)

def distill_accepted(agent, anchors, ids, target, accept, config):
    selected = np.flatnonzero(accept)
    steps = config['federation']['distill_steps']
    if not len(selected) or not steps or config['federation']['lambda_distill'] == 0:
        return dict(distillation_update_count=0, distillation_loss=0.0)
    z = torch.as_tensor(anchors[selected], device=agent.device, dtype=torch.float32).detach()
    pool = torch.as_tensor(ids[selected], device=agent.device, dtype=torch.long)
    frozen = torch.as_tensor(target[selected], device=agent.device, dtype=torch.float32).detach()
    features = agent.scoring_features()[pool].detach()
    temperature = config['federation']['policy_temperature']
    value = 0.0
    for _ in range(steps):
        logits = torch.einsum('bd,bid->bi', agent.actor(z), features) / temperature
        loss = (frozen * (frozen.clamp_min(1e-12).log() - F.log_softmax(logits, -1))).sum(-1).mean()
        loss = loss * config['federation']['lambda_distill']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite accepted-anchor KL')
        agent.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
        agent.actor_optimizer.step()
        value = float(loss.detach())
    return dict(distillation_update_count=steps, distillation_loss=value, accepted_anchors=len(selected))

def local_transfer(agent, client, anchors, ids, responses, confidence, support, local, config):
    if config['method'] == 'uniform_teacher' and config['methodology']['version'] == 'action-grounded-v2':
        target, accept, audit = uniform_supported_target(responses, client, support, config['methodology']['eta'])
        result = distill_accepted(agent, anchors, ids, target, accept, config)
        result['local_advantage_audit'] = audit
        return result
    values, eq, ep, qmax = private_values(agent, anchors, ids, responses[client], local, config)
    target, accept, audit = advantage_target(responses, client, values, confidence, support, eq, ep, qmax, config['methodology']['eta'], uniform=config['method'] == 'uniform_teacher', record_weights=config['methodology']['version'] == 'action-grounded-v2')
    result = distill_accepted(agent, anchors, ids, target, accept, config)
    result['local_advantage_audit'] = dict(audit, epsilon_q_mean=float(eq.mean()), epsilon_pi_mean=float(ep.mean()), qmax=qmax)
    return result

def uniform_supported_target(responses, client, support, eta):
    p = np.asarray(responses, float)
    mask = np.asarray(support, bool)
    if p.ndim != 3 or mask.shape != p.shape[:2] or (not 0 <= eta <= 1):
        raise ValueError('Invalid uniform teacher inputs')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid policy responses')
    allowed = mask & mask[client][None]
    allowed[client] = False
    count = allowed.sum(0)
    accept = (count > 0) & (eta > 0)
    weights = np.divide(allowed, count[None], out=np.zeros_like(allowed, dtype=float), where=count[None] > 0)
    teacher = np.einsum('nm,nmi->mi', weights, p)
    target = p[client].copy()
    target[accept] = (1 - eta) * p[client, accept] + eta * teacher[accept]
    return (target, accept, dict(teacher_rule='uniform_supported_peers', advantage_evaluated=False, teacher_weight_sum=weights[:, accept].sum(1).tolist(), teacher_weight_count=int(accept.sum()), accepted_peer_counts=count.tolist(), accepted_anchors=int(accept.sum()), rejected_anchors=int((~accept).sum()), eligible_peer_pairs=int(allowed.sum()), eta=float(eta), bounds_certified=False, interpretation='whole recipient-advantage teacher removed; support and conservative mixture retained'))
