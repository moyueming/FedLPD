from __future__ import annotations
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class InsufficientLocalData(ValueError):
    pass

class DegenerateGeometry(ValueError):
    pass

def transport_residual(coupling, p, q, tolerance=1e-08):
    g = np.asarray(coupling, dtype=np.float64)
    if g.shape != (len(p), len(q)) or not np.isfinite(g).all():
        raise RuntimeError('Transport solver returned nonfinite or malformed coupling')
    residual = max(float(np.max(np.abs(g.sum(1) - p))), float(np.max(np.abs(g.sum(0) - q))), max(0.0, -float(g.min())))
    if residual > tolerance:
        raise RuntimeError(f'Transport marginal feasibility failure: {residual} > {tolerance}')
    return residual

def discrete_wasserstein(p, q, cost, marginal_tolerance=1e-08, max_iterations=100000, return_diagnostics=False):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    cfull = np.asarray(cost, dtype=np.float64)
    if any((v.ndim != 1 or not np.isfinite(v).all() or np.any(v < 0) or (abs(v.sum() - 1) > marginal_tolerance) for v in (p, q))):
        raise ValueError('Invalid transport probabilities')
    if cfull.shape != (len(p), len(q)) or not np.isfinite(cfull).all() or np.any(cfull < 0):
        raise ValueError('Invalid transport costs')
    ip = np.flatnonzero(p > 0)
    iq = np.flatnonzero(q > 0)
    a = np.ascontiguousarray(p[ip] / p[ip].sum())
    b = np.ascontiguousarray(q[iq] / q[iq].sum())
    c = np.ascontiguousarray(cfull[np.ix_(ip, iq)])
    if len(ip) == 1 or len(iq) == 1:
        coupling = np.outer(a, b)
        solver = 'analytic'
    else:
        try:
            import ot
            coupling, details = ot.emd(a, b, c, numItermax=max_iterations, check_marginals=True, log=True)
            if details.get('warning') or details.get('result_code', 1) != 1:
                raise RuntimeError('Exact finite OT did not solve: ' + str(details))
            solver = 'POT.emd'
        except ImportError:
            from scipy.optimize import linprog
            from scipy.sparse import kron, eye, vstack
            rows = kron(eye(len(a)), np.ones((1, len(b))))
            cols = kron(np.ones((1, len(a))), eye(len(b)))
            answer = linprog(c.ravel(), A_eq=vstack([rows, cols]).tocsc(), b_eq=np.r_[a, b], bounds=(0, None), method='highs', options={'maxiter': max_iterations, 'primal_feasibility_tolerance': max(1e-10, marginal_tolerance)})
            if not answer.success:
                raise RuntimeError('Exact finite OT failed: ' + answer.message)
            coupling = answer.x.reshape(len(a), len(b))
            solver = 'scipy.highs'
    residual = transport_residual(coupling, a, b, marginal_tolerance)
    value = float(np.sum(coupling * c))
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite OT objective')
    report = {'solver': solver, 'marginal_residual': residual, 'marginal_tolerance': marginal_tolerance, 'max_iterations': max_iterations}
    return (value, report) if return_diagnostics else value

def wasserstein_fixed_point(rewards, errors, transitions, gamma=0.95, alpha=0.1, tolerance=0.001, max_iterations=1000, relative_bound=0.01, ot_marginal_tolerance=1e-08, ot_max_iterations=100000):
    r = np.asarray(rewards, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    p = np.asarray(transitions, dtype=np.float64)
    n = len(r)
    if not 0 <= gamma < 1 or alpha < 0:
        raise ValueError('gamma must be in [0,1), alpha nonnegative')
    if tolerance <= 0 or relative_bound <= 0 or max_iterations < 1:
        raise ValueError('Invalid fixed point stopping parameters')
    if r.shape != (n,) or e.shape != (n,) or n == 0:
        raise ValueError('Invalid finite model shape')
    if p.shape != (n, n) or np.any(p < 0) or (not np.allclose(p.sum(1), 1.0, atol=ot_marginal_tolerance, rtol=0)):
        raise ValueError('Invalid transition probabilities')
    if not all((np.isfinite(v).all() for v in [r, e, p])):
        raise ValueError('Nonfinite finite model')
    base = np.abs(r[:, None] - r[None, :]) + alpha * np.abs(e[:, None] - e[None, :])
    cmax = float(base.max())
    cost_bound = cmax / (1 - gamma)
    threshold = min(tolerance, relative_bound * cost_bound)
    d = np.zeros((n, n), dtype=np.float64)
    max_marginal = 0.0
    solvers = set()
    for iteration in range(1, max_iterations + 1):
        new = base.copy()
        for i in range(n):
            for j in range(i + 1, n):
                value, report = discrete_wasserstein(p[i], p[j], d, marginal_tolerance=ot_marginal_tolerance, max_iterations=ot_max_iterations, return_diagnostics=True)
                new[i, j] += gamma * value
                new[j, i] = new[i, j]
                max_marginal = max(max_marginal, report['marginal_residual'])
                solvers.add(report['solver'])
        residual = float(np.max(np.abs(new - d)))
        bound = residual / (1 - gamma)
        if not np.isfinite(new).all():
            raise RuntimeError('Nonfinite finite metric iteration')
        if bound <= threshold:
            return (d, {'iterations': iteration, 'residual': residual, 'residual_bound': bound, 'relative_residual_bound': bound / cost_bound if cost_bound else 0.0, 'cost_bound': cost_bound, 'fixed_point_abs_bound': tolerance, 'fixed_point_relative_bound': relative_bound, 'ot_solvers': sorted(solvers), 'ot_marginal_residual': max_marginal, 'ot_marginal_tolerance': ot_marginal_tolerance, 'ot_max_iterations': ot_max_iterations, 'converged': True})
        d = new
    raise RuntimeError(f'Finite metric failed to converge in {max_iterations} iterations; residual_bound={bound}')

def normalize_distances(distances, epsilon=1e-08, residual_bound=0.0):
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or (not np.isfinite(d).all()) or np.any(d < -epsilon) or (not np.allclose(d, d.T, atol=epsilon, rtol=0)):
        raise ValueError('Invalid metric matrix')
    values = d[np.triu_indices(len(d), k=1)]
    values = values[values > 0]
    if not len(values):
        raise DegenerateGeometry('degenerate behavior distances: no positive distances')
    scale = float(values.mean())
    if scale <= max(epsilon, residual_bound):
        raise DegenerateGeometry('degenerate behavior distances: scale below residual bound')
    return (d / scale, scale)

def farthest_points(x, count, rng):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all() or count < 1:
        raise ValueError('Invalid prototype inputs')
    _, indices = np.unique(x, axis=0, return_index=True)
    indices = np.sort(indices)
    if len(indices) < count:
        raise InsufficientLocalData('Insufficient distinct reference states for behavior estimator')
    unique = x[indices]
    first = int(rng.integers(len(unique)))
    chosen = [first]
    distances = np.sum((unique - unique[first]) ** 2, axis=1)
    while len(chosen) < count:
        distances[chosen] = -1
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.sum((unique - unique[index]) ** 2, axis=1))
    return indices[np.asarray(chosen, dtype=int)]

class FiniteDynamics(nn.Module):

    def __init__(self, input_dim, hidden, classes, reward_min=0.0, reward_max=1.0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.reward = nn.Linear(hidden, 1)
        self.transition = nn.Linear(hidden, classes)
        self.reward_min = reward_min
        self.reward_max = reward_max

    def forward(self, x):
        h = self.body(x)
        return (self.reward_min + (self.reward_max - self.reward_min) * torch.sigmoid(self.reward(h)).squeeze(-1), self.transition(h))

def prototype_mass(embeddings, reference_indices):
    indices = torch.as_tensor(reference_indices, device=embeddings.device, dtype=torch.long)
    basis = embeddings[indices]
    assignments = torch.cdist(embeddings, basis, compute_mode='donot_use_mm_for_euclid_dist').argmin(1)
    assignments[indices] = torch.arange(len(indices), device=embeddings.device)
    mass = torch.bincount(assignments, minlength=len(indices)).double().cpu().numpy()
    return mass / mass.sum()

def episode_split(records, rng):
    episodes = np.unique([b['episode_id'] for b in records])
    rng.shuffle(episodes)
    if len(episodes) < 3:
        raise InsufficientLocalData('Insufficient local episodes for fit/calibration/check split')
    nfit = max(1, min(len(episodes) - 2, int(0.7 * len(episodes))))
    ncal = max(1, min(len(episodes) - nfit - 1, int(0.15 * len(episodes))))
    groups = [set(episodes[:nfit]), set(episodes[nfit:nfit + ncal]), set(episodes[nfit + ncal:])]
    return [np.asarray([i for i, b in enumerate(records) if b['episode_id'] in group], dtype=int) for group in groups]

def estimate_local_metric(agent, rng, options, cfg):
    estimate_started = time.perf_counter()
    records = agent.replay.sample(int(cfg.get('model_sample_limit', 4096)), rng)
    prototype_count = int(cfg.get('prototypes', 16))
    if len(records) < max(prototype_count, 6):
        raise InsufficientLocalData('Insufficient local replay for behavior estimator')
    fit, cal, check = episode_split(records, rng)
    states = [b['state'] for b in records]
    next_states = [b['next_state'] for b in records]
    with torch.no_grad():
        u = torch.cat([agent.adapted_states(states[i:i + 256]) for i in range(0, len(states), 256)])
        nxt = torch.cat([agent.adapted_states(next_states[i:i + 256]) for i in range(0, len(states), 256)])
    refs = farthest_points(u.cpu().numpy(), prototype_count, rng)
    basis = u[refs].detach()
    b = len(refs)
    done = torch.as_tensor([r['done'] for r in records], device=agent.device, dtype=torch.bool)
    nxt[done] = 0
    labels = torch.cdist(nxt, basis, compute_mode='donot_use_mm_for_euclid_dist').argmin(1)
    labels[done] = b
    mass = prototype_mass(u, refs)
    slates = np.stack([r['slate'] for r in records])
    action_features = agent.features[torch.as_tensor(slates, device=agent.device)].reshape(len(records), -1)
    x = torch.cat([u, action_features], 1).detach()
    rewards = torch.tensor([r['reward'] for r in records], device=agent.device, dtype=torch.float32)
    reward_min = float(cfg.get('reward_min', 0.0))
    reward_max = float(cfg.get('reward_max', 1.0))
    if rewards.min() < reward_min - 1e-06 or rewards.max() > reward_max + 1e-06:
        raise ValueError('Observed reward outside declared local-model range')
    model = FiniteDynamics(x.shape[1], int(cfg.get('model_hidden', 64)), b + 1, reward_min, reward_max).to(agent.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('model_lr', 0.001)))
    steps = int(cfg.get('model_steps', 100))
    batch = int(cfg.get('model_batch', 128))
    for _ in range(steps):
        ix = rng.choice(fit, size=min(batch, len(fit)), replace=False)
        rt, logits = model(x[ix])
        loss = F.mse_loss(rt, rewards[ix]) + F.cross_entropy(logits, labels[ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    support = torch.cat([basis, torch.zeros_like(basis[:1])])
    with torch.no_grad():
        predicted_r, logits = model(x)
        mean = torch.softmax(logits, -1) @ support
        residual = (mean - nxt).square().sum(1)
    error_model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()).to(agent.device)
    error_scale = 4.0 * agent.adapter_dim
    error_optimizer = torch.optim.Adam(error_model.parameters(), lr=float(cfg.get('model_lr', 0.001)))
    for _ in range(steps):
        ix = rng.choice(cal, size=min(batch, len(cal)), replace=False)
        prediction = error_model(x[ix]).squeeze(-1) * error_scale
        loss = F.mse_loss(prediction, residual[ix])
        error_optimizer.zero_grad()
        loss.backward()
        error_optimizer.step()
    averages = 1
    probe_actions = []
    with torch.no_grad():
        z = agent.projector(basis)
        actions = agent.actor(z).cpu().numpy()
        scoring = agent.scoring_features().detach().cpu().numpy()
    for reference, action in zip(refs, actions):
        candidates = records[int(reference)]['candidate_ids']
        rank = np.argsort(-(scoring[candidates] @ action), kind='stable')[:agent.slate_size]
        probe_actions.append([candidates[rank]])
    probe_actions = torch.as_tensor(np.asarray(probe_actions), device=agent.device)
    features = agent.features[probe_actions].reshape(b, averages, -1)
    policy_x = torch.cat([basis[:, None, :].expand(-1, averages, -1), features], -1).reshape(b * averages, -1)
    with torch.no_grad():
        policy_r, policy_logits = model(policy_x)
        rbar = policy_r.reshape(b, averages).mean(1).cpu().numpy()
        ebar = (error_model(policy_x).reshape(b, averages) * error_scale).mean(1).cpu().numpy()
        probabilities = torch.softmax(policy_logits, -1).reshape(b, averages, b + 1).mean(1).cpu().numpy().astype(np.float64)
        check_error = error_model(x[check]).squeeze(-1) * error_scale
        diagnostics = {'reward_mse': float(F.mse_loss(predicted_r[check], rewards[check])), 'transition_log_loss': float(F.cross_entropy(logits[check], labels[check])), 'error_mse': float(F.mse_loss(check_error, residual[check])), 'fit_samples': len(fit), 'calibration_samples': len(cal), 'check_samples': len(check), 'candidate_conditioning': 'observed_legal_pool'}
    probabilities /= probabilities.sum(1, keepdims=True)
    p = np.zeros((b + 1, b + 1))
    p[:b] = probabilities
    p[b, b] = 1
    fit_seconds = time.perf_counter() - estimate_started
    ot_started = time.perf_counter()
    raw, convergence = wasserstein_fixed_point(np.r_[rbar, 0.0], np.r_[ebar, 0.0], p, gamma=agent.gamma, alpha=float(cfg.get('metric_alpha', 0.1)) if options['prediction_error'] else 0.0, tolerance=float(cfg.get('fixed_point_abs_bound', cfg.get('ot_tolerance', 0.001))), relative_bound=float(cfg.get('fixed_point_relative_bound', 0.01)), max_iterations=int(cfg.get('fixed_point_max_iterations', cfg.get('ot_max_iterations', 1000))), ot_marginal_tolerance=float(cfg.get('ot_marginal_tolerance', 1e-08)), ot_max_iterations=int(cfg.get('ot_solver_max_iterations', 100000)))
    distances, scale = normalize_distances(raw[:b, :b], residual_bound=convergence['residual_bound'])
    diagnostics.update(convergence)
    diagnostics['raw_scale'] = scale
    diagnostics['model_and_reference_seconds'] = fit_seconds
    diagnostics['fixed_point_seconds'] = time.perf_counter() - ot_started
    agent._estimator_state = {'model': copy.deepcopy(model.state_dict()), 'error_model': copy.deepcopy(error_model.state_dict()), 'model_optimizer': copy.deepcopy(optimizer.state_dict()), 'error_optimizer': copy.deepcopy(error_optimizer.state_dict()), 'raw_distances': raw, 'basis': basis.cpu().numpy(), 'policy_transitions': p, 'diagnostics': copy.deepcopy(diagnostics), 'policy_rewards': np.r_[rbar, 0.0], 'policy_errors': np.r_[ebar, 0.0], 'reference_candidate_ids': [records[int(i)]['candidate_ids'].copy() for i in refs], 'policy_slates': probe_actions[:, 0, :].cpu().numpy(), 'candidate_conditioning': 'observed_legal_pool'}
    return {'states': [states[i] for i in refs], 'distances': distances, 'mass': mass, 'diagnostics': diagnostics}
