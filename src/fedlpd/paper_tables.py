from pathlib import Path
import pandas as pd
from .config import V23_BASELINES, V23_ABLATIONS
from .ablation_reporting import LABELS
from .artifacts import atomic_json, fingerprint

def paper_tables(report, jobs, phase):
    report = Path(report)
    out = report / 'paper_tables'
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(report / 'data/test_macro_by_seed.csv')
    caps = {c['environment']: c['training']['requests_per_client'] // (2 if phase == 'half' else 1) for c in jobs}
    smoke = jobs[0]['smoke']

    def save(name, table):
        table = table.copy()
        if 'method' in table:
            table.insert(1, 'method_label', table.method.map({**LABELS, 'random_matching': 'Random matching'}))
        table.to_csv(out / (name + '.csv'), index=False)

    def metric_table(methods, comparison=False):
        pieces = []
        for env, cap in caps.items():
            wanted = min(40000, cap) if comparison else cap
            for method in methods:
                g = frame[(frame.environment == env) & (frame.method == method)]
                step = 0 if method == 'random' else wanted
                if smoke and step not in set(g.step):
                    step = int(g[g.step <= step].step.max())
                g = g[g.step == step]
                if len(g) != 3 or set(g.seed) != {2026, 2027, 2028}:
                    raise ValueError('Missing paper table test endpoint')
                row = dict(environment=env, method=method, step=step, n_seeds=3)
                for metric in ('J', 'A', 'L'):
                    row[metric + '_mean'] = g[metric].mean()
                    row[metric + '_sd'] = g[metric].std(ddof=1)
                pieces.append(row)
        return pd.DataFrame(pieces)
    save('main_results_test', metric_table(('full', *V23_BASELINES, 'random')))
    save('ablation_final_test', metric_table(('full', *V23_ABLATIONS)))
    save('ablation_40k_test', metric_table(('full', *V23_ABLATIONS), True))
    probe = pd.read_csv(report / 'probe_selection_by_round.csv')
    rows = []
    for (env, method), g in probe.groupby(['environment', 'method']):
        through = min(40000, caps[env])
        g = g[g.step <= through]
        rows.append(dict(environment=env, method=method, through_requests=through, eligible=int(g.eligible_probes.sum()), selected=int(g.selected_probe_count.sum()), zero_probe_rounds=int((g.selected_probe_count == 0).sum()), total_rounds=len(g), n_seeds=g.seed.nunique()))
    save('probe_activity_40k', pd.DataFrame(rows))
    alignment = pd.read_csv(report / 'data/alignment_by_seed.csv')
    stats = alignment.groupby('method').agg(reward_mean=('reward_discrepancy', 'mean'), reward_sd=('reward_discrepancy', 'std'), transition_W1_mean=('transition_discrepancy', 'mean'), transition_W1_sd=('transition_discrepancy', 'std'), n=('seed', 'count')).reset_index()
    save('alignment_discrepancies', stats)
    if (report / 'cost_summary.csv').exists():
        save('training_costs', pd.read_csv(report / 'cost_summary.csv', keep_default_na=False))
    atomic_json(out / 'manifest.json', dict(phase=phase, smoke=smoke, split='test (cost targets: validation)', files={p.name: fingerprint(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))
