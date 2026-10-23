from .config import V23_ABLATIONS, METHOD_LABELS
LABELS = {**METHOD_LABELS, 'signature_off': 'w/o Signature Grounding', 'alignment_off': 'w/o Alignment Loss', 'fedhpd': 'FedHPD-TD3', 'fedslate': 'FedSlate-TD3', 'coverage_off': 'w/o Replay-Support Filter', 'random_probes': 'Random Anchor Selection', 'uniform_teacher': 'Uniform Supported-Peer Teacher'}
INTERPRETATIONS = {'metric_off': 'local metric-loss supervision; behavioral estimates retained', 'alignment_off': 'explicit matched-mass alignment loss; other new mechanisms retained', 'signature_off': 'signature cost and signature-discrepancy mask; action coverage and relational matching retained', 'coverage_off': 'Gaussian replay-support filter only; matching validity and confidence retained', 'random_probes': 'disagreement ranking versus equal-size random selection; identifiable only when eligible count exceeds budget', 'uniform_teacher': 'whole recipient-advantage teacher versus equal supported peers; not an isolated gate or error-penalty effect', 'distillation_off': 'policy distillation beyond retained representation learning, aggregation and critic recalibration'}

def ablation_report(frame, output, phase):
    from .optimization_trial_reporting import paired_comparisons
    import pandas as pd
    selected = frame[(frame.step == (frame.budget // 2 if phase == 'half' else frame.budget)) & frame.method.isin(('full', *V23_ABLATIONS))]
    summary = selected.groupby(['environment', 'method', 'step']).agg(J_mean=('J', 'mean'), J_sd=('J', 'std'), A_mean=('A', 'mean'), A_sd=('A', 'std'), L_mean=('L', 'mean'), L_sd=('L', 'std'), n_seeds=('seed', 'count')).reset_index()
    if len(summary) != 16 or not (summary.n_seeds == 3).all():
        raise ValueError('Incomplete seven-ablation endpoints')
    summary.to_csv(output / 'ablation_results.csv', index=False)
    pairs = pd.DataFrame(paired_comparisons(frame[frame.method.isin(('full', *V23_ABLATIONS))], phase))
    if len(pairs) != 14:
        raise ValueError('Incomplete seven-ablation comparisons')
    pairs['interpretation'] = pairs.method.map(INTERPRETATIONS)
    pairs.to_csv(output / 'ablation_comparisons.csv', index=False)
    lines = ['# Seven-ablation study', '', 'All comparisons use the same request budget within each environment. Mean +/- sample SD over three seeds.', 'Half results and smoke runs do not establish final performance. Intervals are exploratory, n=3, without multiplicity correction.', '', '|Environment|Variant|Requests/client|J mean|J SD|', '|---|---|---:|---:|---:|']
    for r in summary.to_dict('records'):
        lines.append(f"|{r['environment']}|{LABELS[r['method']]}|{r['step']}|{r['J_mean']:.6f}|{r['J_sd']:.6f}|")
    lines += ['', 'Uniform Supported-Peer Teacher removes the whole recipient-advantage teacher, while retaining support and eta=.25.', 'Selection requires eligible anchors > the selection budget. Check probe_selection_by_round.csv and mechanism_execution_summary.csv.', 'No accepted transfer or no binding selection rounds means the corresponding mechanism was not exercised; equal returns alone do not establish no contribution.', 'A zero-active-round cell can reflect insufficient local data or matching support; retain it and report the reason.']
    (output / 'ABLATIONS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
