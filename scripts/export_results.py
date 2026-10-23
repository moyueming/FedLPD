from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from fedlpd.artifacts import atomic_json, fingerprint
from fedlpd.config import V23_ABLATIONS, V23_BASELINES
MAIN = ('full', *V23_BASELINES, 'random')
ENVS = ('kuaisim', 'recsim')
METRICS = ('J', 'A', 'L')

def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()] if Path(path).is_file() else []

def teacher_by_seed(rounds, config, endpoint):
    n = config['env']['num_clients']
    t = config['training']
    window = set(list(range(t['warmup'] + t['communication_every'], endpoint + 1, t['communication_every']))[-5:])
    total = np.zeros((n, n))
    active = np.zeros(n, int)
    events = np.zeros(n, int)
    missing = False
    relevant = [r for r in rounds if r['requests_per_client'] in window]
    for r in relevant:
        if 'teacher_weight_audit_schema' not in r and (not r.get('distillation')):
            missing = True
        for d in r.get('distillation', []):
            audit = d.get('local_advantage_audit', {})
            if 'teacher_weight_sum' not in audit or 'teacher_weight_count' not in audit:
                missing = True
                continue
            i = d['client']
            count = audit['teacher_weight_count']
            weights = np.asarray(audit['teacher_weight_sum'], float)
            if weights.shape != (n,) or count < 0 or (not np.isfinite(weights).all()) or (weights < 0).any() or (weights[i] != 0) or (not np.isclose(weights.sum(), count)):
                raise ValueError('Invalid recorded teacher weights')
            if count:
                total[i] += weights / count
                active[i] += 1
                events[i] += count
    matrix = np.divide(total, active[:, None], out=np.full((n, n), np.nan), where=active[:, None] > 0)
    np.fill_diagonal(matrix, np.nan)
    return (matrix, events, 'missing_weight_audit' if missing else 'recorded' if relevant else 'no_scheduled_rounds')

def client_gains(frame, environment, step):
    chosen = frame[(frame.environment == environment) & (frame.step == step) & frame.method.isin(('full', 'uniform_teacher', 'local'))]
    if chosen.duplicated(['method', 'seed', 'client']).any():
        raise ValueError('Duplicate paired client data')
    keys = set(zip(chosen[chosen.method == 'local'].seed, chosen[chosen.method == 'local'].client))
    n = chosen.client.nunique()
    if keys != {(s, i) for s in (2026, 2027, 2028) for i in range(n)}:
        raise ValueError('Missing Local client/seed')
    out = []
    reference = chosen[chosen.method == 'local'].set_index(['seed', 'client']).J
    for method in ('full', 'uniform_teacher'):
        group = chosen[chosen.method == method].set_index(['seed', 'client']).J
        if set(group.index) != keys:
            raise ValueError('Missing paired client/seed')
        for (seed, client), gain in (group - reference).items():
            out.append(dict(environment=environment, method=method, reference='local', step=step, seed=seed, client=client, gain=float(gain)))
    return pd.DataFrame(out)

def read_evidence(runs, phase):
    metadata = []
    rows = []
    provenance = {}
    caps = {}
    for path in sorted(Path(runs).glob('*/run.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        cfg = m['config']
        method = m['method']
        env = m['environment']
        if cfg.get('trial', {}).get('version') != 'v23':
            raise ValueError('Result exporter requires the configured comparison matrix')
        endpoint = cfg['training']['requests_per_client'] // (2 if phase == 'half' else 1)
        if env in caps and caps[env] != endpoint:
            raise ValueError('Unequal stage budgets')
        caps[env] = endpoint
        m['_path'] = path.parent
        metadata.append(m)
        for file in ('run.json', 'metrics.jsonl', 'rounds.jsonl'):
            if (path.parent / file).exists():
                provenance[path.parent.name + '/' + file] = fingerprint(path.parent / file)
        selected = {}
        priority = {'curve': 0, 'comparison': 1, 'final': 2}
        for record in jsonl(path.parent / 'metrics.jsonl'):
            step = 0 if method == 'random' else record['requests_per_client']
            if record['split'] != 'test' or step > endpoint:
                continue
            if step not in selected or priority.get(record['eval_kind'], 0) > priority.get(selected[step]['eval_kind'], 0):
                selected[step] = record
        if not selected or (not m['smoke'] and method != 'random' and (endpoint not in selected)):
            raise ValueError('Missing fixed-stage test evidence')
        for step, r in selected.items():
            if len(r['clients']) != cfg['env']['num_clients']:
                raise ValueError('Missing test clients')
            for client in r['clients']:
                rows.append(dict(environment=env, method=method, seed=m['seed'], step=step, client=client['client_id'], recorded_step=r['requests_per_client'], eval_sessions=r['eval_sessions'], **{key: client[key] for key in METRICS}))
    required = {(e, m, s) for e in ENVS for m in (*MAIN, *V23_ABLATIONS) for s in (2026, 2027, 2028)}
    if len(metadata) != 90 or {(r['environment'], r['method'], r['seed']) for r in metadata} != required:
        raise ValueError('Incomplete 90-condition comparison matrix')
    for field in ('source_hash', 'smoke'):
        if len({r[field] for r in metadata}) != 1:
            raise ValueError('Mixed result evidence ' + field)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[list(METRICS)]).all().all() or frame.duplicated(['environment', 'method', 'seed', 'step', 'client']).any():
        raise ValueError('Invalid test rows')
    macro = frame.groupby(['environment', 'method', 'seed', 'step'], as_index=False)[list(METRICS)].mean()
    for _, g in macro.groupby(['environment', 'method', 'step']):
        if set(g.seed) != {2026, 2027, 2028}:
            raise ValueError('Incomplete curve seeds')
    return (metadata, frame, macro, caps, provenance)

def personalized_data(metadata, frame, caps, output, status, env='kuaisim'):
    endpoint = caps[env]
    n = next((m['config']['env']['num_clients'] for m in metadata if m['environment'] == env))
    common = set.intersection(*[set(frame[(frame.environment == env) & (frame.method == method)].step) for method in ('full', 'uniform_teacher', 'local')])
    gain_step = max((s for s in common if s <= endpoint))
    gains = client_gains(frame, env, gain_step)
    gains.to_csv(output / f'{env}_client_gains_by_seed.csv', index=False)
    mats = []
    details = []
    audit_rows = []
    for method in ('full', 'uniform_teacher'):
        seed_mats = []
        for meta in metadata:
            if meta['method'] != method or meta['environment'] != env:
                continue
            matrix, events, observed = teacher_by_seed(jsonl(meta['_path'] / 'rounds.jsonl'), meta['config'], endpoint)
            seed_mats.append(matrix)
            for i in range(n):
                for j in range(n):
                    audit_rows.append(dict(environment=env, method=method, seed=meta['seed'], target=i, source=j, weight=matrix[i, j], accepted_anchor_events=events[i], audit_status=observed))
        stack = np.stack(seed_mats)
        counts = np.isfinite(stack).sum(0)
        mean = np.divide(np.nansum(stack, axis=0), counts, out=np.full((n, n), np.nan), where=counts > 0)
        mats.append(mean)
        details.append(counts)
    data = pd.DataFrame(audit_rows)
    data.to_csv(output / f'{env}_teacher_weights_by_seed.csv', index=False)
    return dict(teacher_endpoint=endpoint, gain_endpoint=gain_step, weight_window='last five scheduled rounds; mean within round, then rounds, then seeds', available_seed_counts=[c.tolist() for c in details], audit_status=sorted(set(data.audit_status)))

def alignment_summary(alignment, output, status):
    root = Path(alignment)
    e = json.loads((root / 'alignment_evidence.json').read_text(encoding='utf-8'))
    rows = []
    for r in e['records']:
        rows.append(dict(method=r['method'], seed=r['seed'], **{k: r['matching'][k] for k in ('reward_discrepancy', 'transition_discrepancy')}))
    for i, r in enumerate(e['random_matching']):
        rows.append(dict(method='random_matching', seed=i, **{k: r[k] for k in ('reward_discrepancy', 'transition_discrepancy')}))
    df = pd.DataFrame(rows)
    df.to_csv(output / 'alignment_by_seed.csv', index=False)

def generate(runs, output, phase, device='cpu'):
    import torch
    from alignment_data import alignment_evidence
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata, frame, macro, caps, provenance = read_evidence(runs, phase)
    frame.to_csv(output / 'test_client_metrics.csv', index=False)
    macro.to_csv(output / 'test_macro_by_seed.csv', index=False)
    teachers = {env: personalized_data(metadata, frame, caps, output, None, env) for env in ENVS}
    alignment = alignment_evidence(runs, output / 'alignment_diagnostics', phase, device)
    alignment_summary(alignment, output, None)
    atomic_json(output / 'manifest.json', dict(phase=phase, smoke=metadata[0]['smoke'], split='test', source_hash=metadata[0]['source_hash'], endpoints=caps, teachers=teachers, provenance=provenance, exporter_sha256={p.name: fingerprint(p) for p in (Path(__file__), Path(__file__).with_name('alignment_data.py'))}, files={p.relative_to(output).as_posix(): fingerprint(p) for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('Result data: ' + str(output.resolve()), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export numerical experiment results without plotting.')
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('half', 'finish'), required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    generate(args.runs, args.output, args.phase, args.device)
