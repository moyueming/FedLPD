from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from .coordinate_audit import array_digest, local_geometry
from .coverage import local_coverage

def _losses(z, target, distances):
    return (F.mse_loss(torch.cdist(z, z), distances), (z - target).square().sum(-1).mean())

def _gradient_relation(left, right):
    left = torch.cat([g.detach().reshape(-1).double() for g in left])
    right = torch.cat([g.detach().reshape(-1).double() for g in right])
    a = float(left.norm())
    b = float(right.norm())
    dot = float(left @ right)
    return dict(metric_norm=a, alignment_norm=b, dot=dot, cosine=max(-1.0, min(1.0, dot / (a * b))) if a > 0 and b > 0 else None)

def objective_audit(agent, target, cfg):
    cache = agent._metric_cache
    goal = torch.as_tensor(target, device=agent.device, dtype=torch.float32)
    distances = torch.as_tensor(cache['distances'], device=agent.device, dtype=torch.float32)
    z = agent.encode_states(cache['states'])
    metric, align = _losses(z, goal, distances)
    parameters = list(agent.adapter.parameters()) + list(agent.projector.parameters())
    gm = torch.autograd.grad(metric, parameters, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(align, parameters, retain_graph=True, allow_unused=True)
    gm = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gm)]
    ga = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, ga)]
    zm = torch.autograd.grad(metric, z, retain_graph=True)
    za = torch.autograd.grad(align, z)
    values = dict(metric_loss=float(metric.detach()), alignment_loss=float(align.detach()), target_metric_loss=float(F.mse_loss(torch.cdist(goal, goal), distances)), weighted_joint_loss=float(cfg.get('lambda_metric', 1.0)) * float(metric.detach()) + float(cfg.get('lambda_align', 1.0)) * float(align.detach()))
    if not all((np.isfinite(v) for v in values.values())):
        raise FloatingPointError('Nonfinite fitting objective')
    return dict(values, parameter_gradients=_gradient_relation(gm, ga), latent_gradients=_gradient_relation(zm, za))

def fitting_audit(agent, target, anchors, probes, options, config, round_index, client_id):
    steps = config['diagnostics']['fitting_steps']
    cfg = config['federation']
    traces = {}
    for name in ('joint', 'alignment_only'):
        memo = {id(agent.replay): agent.replay, id(agent._metric_cache): agent._metric_cache}
        if agent._estimator_state is not None:
            memo[id(agent._estimator_state)] = agent._estimator_state
        private = deepcopy(agent, memo)
        flags = dict(options, metric=name == 'joint', alignment=True)
        trace = []
        previous = 0
        for step in [0, *steps]:
            if step:
                private.fit_geometry(target, flags, dict(cfg, aux_steps=step - previous))
            mask, coverage = local_coverage(private, probes, config, round_index, client_id)
            trace.append(dict(step=step, **objective_audit(private, target, cfg), geometry=local_geometry(private, target, anchors), support_mask=mask.tolist(), accepted_probes=int(mask.sum()), coverage=coverage))
            previous = step
        traces[name] = trace
        del private
    with torch.no_grad():
        z0 = agent.encode_states(agent._metric_cache['states'])
    z = z0.detach().clone().requires_grad_(True)
    goal = torch.as_tensor(target, device=agent.device, dtype=torch.float32)
    distances = torch.as_tensor(agent._metric_cache['distances'], device=agent.device, dtype=torch.float32)
    optimizer = torch.optim.Adam([z], lr=0.03)
    free = []
    for step in range(steps[-1] + 1):
        metric, align = _losses(z, goal, distances)
        loss = float(cfg.get('lambda_metric', 1.0)) * metric + float(cfg.get('lambda_align', 1.0)) * align
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite free-coordinate diagnostic')
        if step == 0 or step in steps:
            free.append(dict(step=step, metric_loss=float(metric.detach()), alignment_loss=float(align.detach()), weighted_joint_loss=float(loss.detach())))
        if step < steps[-1]:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return dict(client=client_id, publication_eligible=False, identity=dict(initial_latents=array_digest(z0.cpu().numpy()), target=array_digest(target), distances=array_digest(agent._metric_cache['distances']), probes=array_digest(probes)), network=traces, free_coordinates=free, free_coordinate_lr=0.03, interpretation='Fixed inputs and finite optimization; no automatic adoption, global-optimum or policy-benefit claim.')
