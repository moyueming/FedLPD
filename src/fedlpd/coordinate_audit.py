from hashlib import sha256
import numpy as np
import torch
from .experiment_checks import quantiles

def array_digest(value):
    a = np.ascontiguousarray(value)
    return sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

def pair_mean(points):
    if len(points) < 2:
        return 0.0
    d = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return float(d[np.triu_indices(len(points), 1)].mean())

def local_geometry(agent, target, anchors):
    with torch.no_grad():
        z = agent.encode_states(agent._metric_cache['states']).cpu().numpy().astype(np.float64)
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite diagnostic prototype representation')
    residual = np.linalg.norm(z - target, axis=1)
    return dict(prototype_count=len(z), alignment_rms=float(np.sqrt(np.mean(residual ** 2))), alignment_distances=quantiles(residual), encoded_pair_mean=pair_mean(z), target_pair_mean=pair_mean(target), anchor_pair_mean=pair_mean(anchors), encoded_norm=quantiles(np.linalg.norm(z, axis=1)), target_norm=quantiles(np.linalg.norm(target, axis=1)), prototype_nearest_anchor=quantiles(np.linalg.norm(z[:, None] - anchors[None, :], axis=-1).min(1)), target_nearest_anchor=quantiles(np.linalg.norm(target[:, None] - anchors[None, :], axis=-1).min(1)))

def coupling_summary(coupling, mass):
    weights = coupling / np.asarray(mass)[:, None]
    entropy = -(weights * np.log(np.maximum(weights, 1e-300))).sum(1) / np.log(weights.shape[1])
    return dict(normalized_row_entropy=quantiles(entropy), row_maximum_weight=quantiles(weights.max(1)))
