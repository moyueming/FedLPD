from hashlib import sha256
import numpy as np

def content_groups(features, environment, topics=6):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 5 or (not np.isfinite(x).all()):
        raise ValueError('Content grouping needs at least five finite public item vectors')
    if environment == 'recsim':
        groups = np.argmax(x[:, :topics], axis=1) % 5
    else:
        raise ValueError('KuaiSim requires groups from training feedback, not random ID features')
    if set(groups.tolist()) != set(range(5)):
        raise ValueError('All five public content groups must exist')
    return np.asarray(groups, dtype=np.int64)

def candidate_probabilities(groups, available, client_id, strength):
    p = np.full(len(available), (1 - strength) / len(available), dtype=np.float64)
    selected = np.asarray(groups)[available] == client_id % 5
    if selected.any():
        p[selected] += strength / selected.sum()
    else:
        p.fill(1 / len(available))
    return p / p.sum()

def group_digest(groups):
    return sha256(np.asarray(groups, dtype='<i8').tobytes()).hexdigest()

def training_feedback_groups(items, splits, labels, catalog_size):
    counts = np.zeros(catalog_size, dtype=np.int64)
    reward_sums = np.zeros(catalog_size)
    for start in range(0, len(items), 1000000):
        stop = min(start + 1000000, len(items))
        ids = np.asarray(items[start:stop])
        valid = (np.asarray(splits[start:stop]) == 0) & (ids >= 0) & (ids < catalog_size)
        chosen = ids[valid]
        feedback = np.asarray(labels[start:stop])[valid]
        rewards = 0.8 * feedback[:, 1] + 0.2 * feedback[:, 2]
        if not np.isfinite(rewards).all():
            raise ValueError('Nonfinite training feedback')
        counts += np.bincount(chosen, minlength=catalog_size)
        reward_sums += np.bincount(chosen, weights=rewards, minlength=catalog_size)
    if (counts == 0).any():
        raise ValueError('Every grouped item must have training evidence')
    prior = float(reward_sums.sum() / counts.sum())
    scores = (reward_sums + 20 * prior) / (counts + 20)
    order = np.argsort(scores, kind='stable')
    groups = np.empty(catalog_size, dtype=np.int64)
    groups[order] = np.arange(catalog_size) * 5 // catalog_size
    info = dict(protocol='train-feedback-quintiles-v1', decision_split='train', prior_count=20, training_rows=int(counts.sum()), group_reward_means=[float(scores[groups == k].mean()) for k in range(5)], policy_receives_groups=False, policy_receives_feedback_statistics=False)
    return (groups, info)
