import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

def objective(g, D, C, cost, gw, coverage):
    row = g.sum(1)
    col = g.sum(0)
    relational = row @ (D * D) @ row + col @ (C * C) @ col - 2 * np.sum(g.T @ D @ g * C)
    return float(np.sum(g * cost) + gw * relational - coverage * g.sum())

def partial_coupling(D, C, cost, mass, nu, allowed, gw, coverage, iterations):
    D, C, cost = (np.asarray(D, float), np.asarray(C, float), np.asarray(cost, float))
    mass, nu = (np.asarray(mass, float), np.asarray(nu, float))
    allowed = np.asarray(allowed, bool)
    n, m = cost.shape
    if D.shape != (n, n) or C.shape != (m, m) or allowed.shape != cost.shape or (mass.shape != (n,)) or (nu.shape != (m,)):
        raise ValueError('Bad partial matching shape')
    if any((not np.isfinite(a).all() for a in (D, C, cost, mass, nu))) or min(mass.min(), nu.min()) < 0:
        raise ValueError('Invalid matching data')
    A = np.vstack((np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))))
    bounds = [(0, None) if v else (0, 0) for v in allowed.ravel()]
    g = np.zeros_like(cost)
    before = objective(g, D, C, cost, gw, coverage)
    gap = 0.0
    steps = 0
    for steps in range(1, iterations + 1):
        gradient = cost - coverage + 2 * gw * (D * D @ g.sum(1)[:, None] + (C * C @ g.sum(0))[None, :] - 2 * D @ g @ C)
        lp = linprog(gradient.ravel(), A_ub=A, b_ub=np.r_[mass, nu], bounds=bounds, method='highs')
        if not lp.success:
            raise RuntimeError('Partial GW linear oracle failed: ' + lp.message)
        direction = lp.x.reshape(n, m) - g
        gap = -float(np.sum(gradient * direction))
        if gap <= 1e-10:
            break
        f0 = objective(g, D, C, cost, gw, coverage)
        linear = float(np.sum(gradient * direction))
        quadratic = objective(g + direction, D, C, cost, gw, coverage) - f0 - linear
        alpha = float(np.clip(-linear / (2 * quadratic), 0, 1)) if quadratic > 0 else 1.0
        g += alpha * direction
    residual = max(0.0, float((g.sum(1) - mass).max()), float((g.sum(0) - nu).max()), -float(g.min()), float(np.abs(g[~allowed]).max(initial=0)))
    if residual > 1e-07:
        raise RuntimeError('Partial coupling violates marginal/edge constraints')
    after = objective(g, D, C, cost, gw, coverage)
    if after > before + 1e-07:
        raise RuntimeError('Partial matching objective increased')
    return (g, dict(matched_mass=float(g.sum()), feasibility_residual=residual, iterations=steps, frank_wolfe_gap=gap, objective_before=before, objective_after=after, global_optimum_claim=False))

def relational_coordinates(distance, dimension):
    n = len(distance)
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ distance ** 2 @ center
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(values)[::-1][:dimension]
    z = vectors[:, order] * np.sqrt(np.maximum(values[order], 0))
    for j in range(z.shape[1]):
        if z[np.abs(z[:, j]).argmax(), j] < 0:
            z[:, j] *= -1
    return np.pad(z, ((0, 0), (0, max(0, dimension - z.shape[1]))))

def initialize(summaries, count, dimension):
    eligible = [s for s in summaries if np.any(s['valid'])]
    if not eligible:
        return None
    source = max(eligible, key=lambda s: int(np.sum(s['valid'])))
    ids = np.flatnonzero(source['valid'])
    chosen = list(ids[:1])
    while len(chosen) < min(count, len(ids)):
        score = source['distances'][ids][:, chosen].min(1)
        score[np.isin(ids, chosen)] = -1
        chosen.append(int(ids[score.argmax()]))
    chosen = np.resize(np.asarray(chosen), count)
    u = source['signatures'][chosen].copy()
    z = relational_coordinates(source['distances'][np.ix_(chosen, chosen)], dimension)
    return dict(signatures=u, anchors=z)

def match(summaries, config, dimension, previous=None):
    c = config['methodology']
    f = config['federation']
    m = f['anchors']
    nu = np.full(m, 1 / m)
    grounded = config['method'] != 'signature_off'
    bank = previous if previous is not None else initialize(summaries, m, dimension)
    if bank is None:
        return (None, [np.zeros((len(s['mass']), m)) for s in summaries], [])
    u = np.array(bank['signatures'], copy=True)
    anchors = np.array(bank['anchors'], copy=True)
    reports = []
    weights = np.full(len(summaries), 1 / len(summaries))

    def couplings():
        out = []
        for s in summaries:
            cost = cdist(s['signatures'], u, 'sqeuclidean')
            allowed = np.broadcast_to(s['valid'][:, None], cost.shape).copy()
            if grounded:
                allowed &= cost <= c['signature_threshold'] ** 2
            else:
                cost = np.zeros_like(cost)
            g, r = partial_coupling(s['distances'], cdist(anchors, anchors), cost, s['mass'], nu, allowed, c['gw_weight'], c['coverage_reward'], c['matching_steps'])
            out.append(g)
            reports.append(r)
        return out
    for _ in range(f['anchor_iterations']):
        gs = couplings()
        den = sum((w * g.sum(0) for w, g in zip(weights, gs)))
        numerator = sum((w * g.T @ s['signatures'] for w, g, s in zip(weights, gs, summaries)))
        candidate = np.divide(numerator, den[:, None], out=u.copy(), where=den[:, None] > 0)
        for j in range(m):
            valid = all((np.all(np.linalg.norm(s['signatures'][g[:, j] > 1e-12] - candidate[j], axis=1) <= c['signature_threshold'] + 1e-10) for s, g in zip(summaries, gs)))
            if grounded and valid:
                u[j] = candidate[j]
        z = torch.tensor(anchors, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=f['anchor_lr'])
        for _ in range(f['anchor_gradient_steps']):
            C = torch.cdist(z, z)
            loss = z.sum() * 0
            for w, g, s in zip(weights, gs, summaries):
                G = torch.tensor(g, dtype=torch.float64)
                D = torch.tensor(s['distances'], dtype=torch.float64)
                row = G.sum(1)
                col = G.sum(0)
                loss += w * c['gw_weight'] * (row @ (D * D) @ row + col @ (C * C) @ col - 2 * (G.T @ D @ G * C).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.sub_(z.mean(0, keepdim=True))
        anchors = z.detach().numpy()
    gs = couplings()
    return (dict(signatures=u, anchors=anchors), gs, reports)
