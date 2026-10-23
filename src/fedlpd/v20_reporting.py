import json
import numpy as np
import pandas as pd
CONTRASTS = (('local', 'local_fixed', 'local item capacity + regularization'), ('gpfedrec', 'local', 'GP federation beyond matched local capacity'), ('gpfedrec', 'gp_uniform', 'graph and personalized target beyond uniform aggregation'), ('direct_lpd', 'v14_reference', 'LPD with learned item capacity'), ('full', 'direct_lpd', 'isolated geometry + replay grounding package'), ('full', 'v14_reference', 'total change from v14'), ('full', 'alignment_off', 'alignment within replay bridge'))

def causal_report(frame, runs, output, phase, version='v20'):
    selected = frame[frame.step == (frame.budget // 2 if phase == 'half' else frame.budget)]
    contrasts = CONTRASTS
    if version == 'v23':
        from .ablation_reporting import INTERPRETATIONS
        contrasts = tuple((('full', name, meaning) for name, meaning in INTERPRETATIONS.items()))
    rows = []
    for env in ('kuaisim', 'recsim'):
        for left, right, interpretation in contrasts:
            if version == 'v22' and left == 'full':
                interpretation = {'direct_lpd': 'whole new-methodology package versus historical direct LPD; not a single-component ablation', 'v14_reference': 'whole new-methodology package plus learned items versus v14', 'alignment_off': 'matched-mass alignment loss within new methodology; all other new mechanisms retained'}[right]
            a = selected[(selected.environment == env) & (selected.method == left)].set_index('seed').J
            b = selected[(selected.environment == env) & (selected.method == right)].set_index('seed').J
            if set(a.index) != {2026, 2027, 2028} or set(a.index) != set(b.index):
                raise ValueError('Missing causal pair')
            d = a - b
            se = d.std(ddof=1) / np.sqrt(3)
            rows.append(dict(environment=env, left=left, right=right, left_mean=a.mean(), right_mean=b.mean(), delta=d.mean(), relative_pct=100 * d.mean() / b.mean(), wins=int((d > 0).sum()), paired_95_low=d.mean() - 4.30265273 * se, paired_95_high=d.mean() + 4.30265273 * se, interpretation=interpretation))
    pd.DataFrame(rows).to_csv(output / 'causal_contrasts.csv', index=False)
    clients = []
    bridge = []
    graph = []
    parameters = []
    outputs = []
    paper = []
    selection = []
    for run in sorted(runs.glob('*/run.json')):
        m = json.loads(run.read_text(encoding='utf-8'))
        key = dict(environment=m['environment'], method=m['method'], seed=m['seed'])
        for i, p in enumerate(m['parameter_counts_by_client']):
            parameters.append(dict(key, client=i, **p))
        for line in (run.parent / 'metrics.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            for c in r['clients']:
                clients.append(dict(key, split=r['split'], step=r['requests_per_client'], eval_sessions=r['eval_sessions'], **c))
        path = run.parent / 'rounds.jsonl'
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                if r.get('methodology_version'):
                    if version == 'v23':
                        selection.append(dict(key, step=r['requests_per_client'], skipped=r.get('skipped', False), skip_reason=r.get('skip_reason'), probe_budget=m['config']['federation']['probe_selected'], eligible_probes=r.get('eligible_probes', 0), selected_probe_count=r.get('selected_probe_count', 0), selection_budget_binding=r.get('selection_budget_binding', False), distillation_updates=r.get('distillation_update_count', 0), accepted_anchors=sum((d.get('local_advantage_audit', {}).get('accepted_anchors', 0) for d in r.get('distillation', [])))))
                    common = dict(key, step=r['requests_per_client'], skipped=r.get('skipped', False), skip_reason=r.get('skip_reason'), eligible_probes=r.get('eligible_probes', 0), distillation_updates=r.get('distillation_update_count', 0), bounds_certified=False, matched_mass=json.dumps(r.get('matched_mass', [])), supporting_clients=json.dumps(r.get('supporting_clients', [])))
                    for i, local in enumerate(r.get('local_models', [])):
                        transfer = next((d for d in r.get('distillation', []) if d.get('client') == i), {})
                        paper.append(dict(common, client=i, **local, **transfer.get('local_advantage_audit', {})))
                    if not r.get('local_models'):
                        paper.append(common)
                if m['method'] in ('fedhpd', 'fedslate'):
                    fields = ('public_query_count', 'q_query_count', 'q_response_evaluations', 'server_update_count', 'server_parameter_count', 'server_loss_before', 'server_loss_last', 'critic_transfer_update_count', 'distillation_update_count', 'communication_bytes', 'mean_fusion_correction', 'information_protocol')
                    outputs.append(dict(key, step=r['requests_per_client'], **{k: r[k] for k in fields if k in r}))
                for i, b in enumerate(r.get('replay_bridge', [])):
                    bridge.append(dict(key, step=r['requests_per_client'], client=i, **b))
                if 'graph_neighbor_counts' in r:
                    graph.append(dict(key, step=r['requests_per_client'], neighbors=json.dumps(r['graph_neighbor_counts']), weights=json.dumps(r['graph_weights']), personalized_target_mse=r['personalized_target_mse']))
    for name, data in [('client_metrics', clients), ('bridge_diagnostics', bridge), ('graph_diagnostics', graph), ('parameter_counts', parameters), ('output_baseline_diagnostics', outputs)]:
        pd.DataFrame(data).to_csv(output / (name + '.csv'), index=False)
    if version in ('v22', 'v23'):
        pd.DataFrame(paper).to_csv(output / 'methodology_diagnostics.csv', index=False)
    if version == 'v23':
        fields = ['environment', 'method', 'seed', 'step', 'skipped', 'skip_reason', 'probe_budget', 'eligible_probes', 'selected_probe_count', 'selection_budget_binding', 'distillation_updates', 'accepted_anchors']
        df = pd.DataFrame(selection, columns=fields)
        df.to_csv(output / 'probe_selection_by_round.csv', index=False)
        execution = df.groupby(['environment', 'method'], as_index=False).agg(rounds=('step', 'count'), skipped_rounds=('skipped', 'sum'), binding_selection_rounds=('selection_budget_binding', 'sum'), distillation_updates=('distillation_updates', 'sum'), accepted_anchors=('accepted_anchors', 'sum'))
        execution.to_csv(output / 'mechanism_execution_summary.csv', index=False)
        from .ablation_reporting import ablation_report
        ablation_report(frame, output, phase)
