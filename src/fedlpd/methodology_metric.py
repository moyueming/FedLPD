from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import cdist
from .metric import FiniteDynamics, episode_split, farthest_points, prototype_mass, discrete_wasserstein, InsufficientLocalData

def reference_slates(config, catalog_size):
    c = config['methodology']
    rng = np.random.default_rng(np.random.SeedSequence([c['reference_seed'], config['env']['catalog_seed']]))
    return np.stack([rng.choice(catalog_size, size=config['env']['slate_size'], replace=False) for _ in range(c['reference_actions'])])

def action_fixed_point(rewards, transitions, gamma, tolerance, max_iterations, ot_tolerance=1e-08):
    r = np.asarray(rewards, dtype=np.float64)
    p = np.asarray(transitions, dtype=np.float64)
    if r.ndim != 2 or p.shape != (r.shape[0], r.shape[1], r.shape[0]):
        raise ValueError('Invalid action-conditioned finite model')
    if not 0 <= gamma < 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid fixed point settings')
    if not np.isfinite(r).all() or not np.isfinite(p).all() or np.any(p < 0) or (not np.allclose(p.sum(-1), 1)):
        raise ValueError('Invalid action probabilities/rewards')
    n, actions = r.shape
    d = np.zeros((n, n))
    for it in range(1, max_iterations + 1):
        new = np.zeros_like(d)
        for i in range(n):
            for j in range(i + 1, n):
                new[i, j] = new[j, i] = max((abs(r[i, a] - r[j, a]) + gamma * discrete_wasserstein(p[i, a], p[j, a], d, marginal_tolerance=ot_tolerance) for a in range(actions)))
        bound = float(np.abs(new - d).max()) / (1 - gamma)
        if bound <= tolerance:
            return (d, dict(iterations=it, residual_bound=bound, operator='max_shared_reference_actions'))
        d = new
    raise RuntimeError('Action-conditioned fixed point did not converge')

def finite_signatures(rewards, transitions, gamma, horizon):
    r = np.asarray(rewards)
    p = np.asarray(transitions)
    v = np.zeros(len(r))
    q = r.copy()
    for _ in range(horizon):
        q = r + gamma * np.einsum('san,n->sa', p, v)
        v = q.mean(1)
    return np.stack((r, q), axis=-1).reshape(len(r), -1)

def nearest_distance(reference, queries):
    return np.concatenate([cdist(queries[i:i + 128], reference).min(1) for i in range(0, len(queries), 128)])

def estimate(agent, config, refs, rng):
    c = config['methodology']
    f = config['federation']
    records = agent.replay.sample(f['model_sample_limit'], rng)
    b = int(f['prototypes'])
    L = len(refs)
    if len(records) < max(b, 6):
        raise InsufficientLocalData('Not enough local transitions for action signatures')
    fit, cal, check = episode_split(records, rng)
    states = [r['state'] for r in records]
    next_states = [r['next_state'] for r in records]
    with torch.no_grad():
        u = torch.cat([agent.adapted_states(states[i:i + 256]) for i in range(0, len(states), 256)])
        nxt = torch.cat([agent.adapted_states(next_states[i:i + 256]) for i in range(0, len(states), 256)])
    indices = farthest_points(u.cpu().numpy(), b, rng)
    basis = u[indices].detach()
    terminal = torch.tensor([r['done'] for r in records], device=agent.device, dtype=torch.bool)
    labels = torch.cdist(nxt, basis).argmin(1)
    labels[terminal] = b
    slate_ids = np.stack([r['slate'] for r in records])
    features = agent.features[torch.as_tensor(slate_ids, device=agent.device)].flatten(1)
    x = torch.cat((u, features), dim=1).detach()
    reward = torch.tensor([r['reward'] for r in records], device=agent.device, dtype=torch.float32)
    lo = float(f.get('reward_min', 0.0))
    hi = float(f.get('reward_max', 1.0))
    if float(reward.min()) < lo - 1e-06 or float(reward.max()) > hi + 1e-06:
        raise ValueError('Reward outside shared utility range')
    model = FiniteDynamics(x.shape[1], f['model_hidden'], b + 1, lo, hi).to(agent.device)
    opt = torch.optim.Adam(model.parameters(), lr=f['model_lr'])
    for _ in range(f['model_steps']):
        ix = rng.choice(fit, size=min(len(fit), f['model_batch']), replace=False)
        pred, logits = model(x[ix])
        loss = F.mse_loss(pred, reward[ix]) + F.cross_entropy(logits, labels[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    actions = agent.features[torch.as_tensor(refs, device=agent.device)].flatten(1)
    query = torch.cat((basis[:, None, :].expand(-1, L, -1), actions[None].expand(b, -1, -1)), dim=-1).reshape(b * L, -1)
    with torch.no_grad():
        pr, pl = model(query)
        rp = pr.reshape(b, L).cpu().numpy().astype(float)
        pp = torch.softmax(pl, -1).reshape(b, L, b + 1).cpu().numpy().astype(float)
        cr, cl = model(x[cal])
        hr, hl = model(x[check])
        reward_error = float(torch.quantile((cr - reward[cal]).abs(), c['calibration_quantile']))
        model_nll = float(F.cross_entropy(cl, labels[cal]))
        check_mse = float(F.mse_loss(hr, reward[check]))
        check_nll = float(F.cross_entropy(hl, labels[check]))
    raw = x.cpu().numpy()
    scale = np.maximum(raw[fit].std(0), 0.1)
    training = raw[fit] / scale
    calibration = nearest_distance(training, raw[cal] / scale)
    threshold = max(float(np.quantile(calibration, c['calibration_quantile'])), 1e-08)
    distance = nearest_distance(training, query.detach().cpu().numpy() / scale).reshape(b, L)
    covered = distance <= threshold
    valid = covered.all(1)
    confidence = np.exp(-distance.mean(1) / threshold - reward_error - model_nll) * valid
    pair_reliability = np.minimum(confidence[:, None], confidence[None, :])
    p = np.zeros((b + 1, L, b + 1))
    p[:b] = pp / pp.sum(-1, keepdims=True)
    p[b, :, b] = 1
    r = np.zeros((b + 1, L))
    r[:b] = rp
    distances, fp = action_fixed_point(r, p, agent.gamma, f['fixed_point_abs_bound'], f['fixed_point_max_iterations'], f['ot_marginal_tolerance'])
    signatures = finite_signatures(r, p, agent.gamma, c['signature_horizon'])[:b]
    mass = prototype_mass(u, indices)
    summary = dict(distances=distances[:b, :b], signatures=signatures, mass=mass, reliability=pair_reliability, valid=valid)
    diagnostics = dict(valid_prototypes=int(valid.sum()), prototypes=b, reference_actions=L, reference_action_coverage=covered.mean(0).tolist(), joint_support_threshold=threshold, calibration_reward_abs_quantile=reward_error, calibration_transition_nll=model_nll, check_reward_mse=check_mse, check_transition_nll=check_nll, fit_samples=len(fit), calibration_samples=len(cal), check_samples=len(check), uncertainty_role='reliability only, never added to behavioral distance', **fp)
    local = dict(states=[states[int(i)] for i in indices], bank=states, calibration=[records[int(i)] for i in cal], fit_records=[records[int(i)] for i in fit], diagnostics=diagnostics, summary=summary)
    agent._estimator_state = dict(model=deepcopy(model.state_dict()), optimizer=deepcopy(opt.state_dict()), reference_slates=refs.copy(), signatures=signatures, distances=distances, action_rewards=r, action_transitions=p, diagnostics=diagnostics)
    return local
