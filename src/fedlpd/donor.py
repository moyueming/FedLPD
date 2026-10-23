import numpy as np

def checked_donors(masks, shape):
    masks = np.asarray(masks)
    if masks.dtype != np.bool_ or masks.shape != shape:
        raise ValueError('donor_masks must be Boolean [client, probe]')
    return masks

def donor_audit(masks, weights, confidence, legacy_confidence, *, applied):
    n, _, p = weights.shape
    masks = checked_donors(masks, (n, p))
    allowed = np.broadcast_to(masks[None, :, :], (n, n, p)) & (~np.eye(n, dtype=bool))[:, :, None]
    active = confidence > 0
    return dict(applied=bool(applied), mode='replay_knn_rank', selected_support_masks=masks.tolist(), selected_probe_count=p, legacy_confidence=np.asarray(legacy_confidence).tolist(), eligible_donor_counts=allowed.sum(1).tolist(), active_probe_counts=active.sum(1).tolist(), unsupported_donor_weight_mass=float((weights * ~masks[None, :, :]).sum()), unsupported_recipient_active_pairs=int((active & ~masks).sum()), no_donor_pairs=int((allowed.sum(1) == 0).sum()), confidence_sum=float(confidence.sum()), legacy_confidence_sum=float(np.asarray(legacy_confidence).sum()), interpretation='Rank filters donors only. Legacy confidence retained wherever a valid teacher exists; no extra updates.')
