import numpy as np

def client_metrics(sessions, gamma=0.95):
    if not sessions or any((len(s) == 0 for s in sessions)):
        raise ValueError('At least one nonempty session is required')
    arrays = [np.asarray(s, dtype=np.float64) for s in sessions]
    if not all((np.isfinite(a).all() for a in arrays)):
        raise ValueError('Nonfinite reward')
    return {'J': float(np.mean([np.dot(gamma ** np.arange(len(a)), a) for a in arrays])), 'A': float(sum((a.sum() for a in arrays)) / sum((len(a) for a in arrays))), 'L': float(np.mean([len(a) for a in arrays])), 'sessions': len(arrays), 'requests': sum((len(a) for a in arrays))}

def macro_metrics(clients):
    if not clients:
        raise ValueError('No clients')
    return {name: float(np.mean([c[name] for c in clients])) for name in ('J', 'A', 'L')}

def seed_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or (not np.isfinite(a).all()):
        raise ValueError('Need >=2 independent finite seed values')
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)), 'n': int(a.size)}
