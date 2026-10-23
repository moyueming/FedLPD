from pathlib import Path
import json
from .config import stable_hash, task_hash, method_config_hash

def verify_pilot_source_snapshot(directory, expected):
    from hashlib import sha256
    from zipfile import ZipFile
    digest = sha256()
    with ZipFile(Path(directory) / 'source/source.zip') as z:
        names = sorted((n for n in z.namelist() if n.startswith('fedlpd/') and n.endswith('.py')))
        if not names or len(names) != len(set(names)):
            raise ValueError('Invalid pilot source snapshot')
        for name in names:
            digest.update(name[len('fedlpd/'):].encode())
            digest.update(z.read(name))
    if digest.hexdigest() != expected:
        raise ValueError('Pilot source snapshot differs from declared identity')

def compatible_source(source):
    from .protocol import source_identity
    current = source_identity()
    if source == current:
        return True
    path = Path(__file__).resolve().parents[2] / 'configs/pilot-source-review.json'
    if not path.is_file():
        return False
    review = json.loads(path.read_text(encoding='utf-8'))
    return review.get('current_source_hash') == current and source in review.get('reviewed_sources', []) and (review.get('scope') == 'execution_evidence_only')

def validate_pilot_configuration(meta, wanted, base):
    if meta.get('status') != 'completed' or meta.get('smoke') or meta.get('purpose') != 'pilot' or (meta.get('seed') != base['cost']['pilot_seed']):
        raise ValueError('Incomplete pilot identity')
    if not compatible_source(meta.get('source_hash')):
        raise ValueError('Pilot source is not current or explicitly reviewed')
    if meta.get('task_hash') != task_hash(wanted) or meta.get('method_config_hash') != method_config_hash(wanted):
        raise ValueError('Pilot task or method differs')
    c = meta['config']
    if meta.get('configuration_hash') != stable_hash(c):
        raise ValueError('Pilot configuration checksum mismatch')
    if c.get('method') != meta['method'] or c.get('environment') != meta['environment'] or c.get('seed') != meta['seed']:
        raise ValueError('Pilot recorded identity differs')
    if task_hash(c) != task_hash(wanted) or method_config_hash(c) != method_config_hash(wanted):
        raise ValueError('Pilot configuration task differs')
    phase = meta['phase']
    profiles = {'quick_paths': (1200, 800, 200), 'learning_diagnostic': (4000, 2000, 1000)}
    if phase not in profiles:
        raise ValueError('Unknown short pilot phase')
    budget, warmup, every = profiles[phase]
    t = c['training']
    if (t['requests_per_client'], t['warmup'], t['communication_every']) != (budget, warmup, every):
        raise ValueError('Short pilot schedule differs')
    if meta.get('requests_per_client') != budget or meta.get('budget_per_client') != budget:
        raise ValueError('Short pilot budget incomplete')
    exclude = {'requests_per_client', 'warmup', 'communication_every', 'checkpoint_every', 'resume_every', 'cpu_threads'}
    if any((t.get(k) != v for k, v in wanted['training'].items() if k not in exclude)):
        raise ValueError('Pilot learning schedule differs')
    if c['agent'] != wanted['agent'] or c['federation'] != wanted['federation']:
        raise ValueError('Pilot algorithm configuration differs')
    for key in ('seed', 'validation_seed', 'diagnostic_seed', 'comparison_requests'):
        if c['evaluation'][key] != wanted['evaluation'][key]:
            raise ValueError('Pilot evaluation identity differs')

def _validate_rank_abstention(row, config):
    from math import ceil
    from .coverage import settings
    from .federation import method_options
    gate = settings(config)
    cfg = config['federation']
    n = config['env']['num_clients']
    audit = row.get('rank_coverage_audit', {})
    if gate['mode'] != 'replay_knn_rank' or not method_options(config['method'])['coverage'] or row.get('coverage_mode') != gate['mode'] or (audit.get('mode') != gate['mode']) or (audit.get('applied') is not True):
        raise ValueError('Pilot empty probes lack an active rank coverage audit')
    count = row.get('candidate_probes')
    required = ceil(cfg['coverage_fraction'] * n)
    if type(count) is not int or count != cfg['probe_candidates'] or count <= 0 or (audit.get('required_clients') != required) or (audit.get('tail_alpha') != gate['tail_alpha']):
        raise ValueError('Pilot rank coverage configuration differs')
    masks = audit.get('support_masks')
    if not isinstance(masks, list) or len(masks) != n or any((not isinstance(mask, list) or len(mask) != count or any((type(v) is not bool for v in mask)) for mask in masks)):
        raise ValueError('Pilot rank coverage masks missing or malformed')
    supporting = [sum((mask[i] for mask in masks)) for i in range(count)]
    if audit.get('supporting_clients') != supporting or any((v >= required for v in supporting)) or audit.get('eligible_probes') != 0 or (audit.get('eligible_indices') != []) or (row.get('eligible_probes') != 0):
        raise ValueError('Pilot rank abstention contradicts support evidence')
    empty = ('selected_probes', 'eligible_indices', 'selected_indices', 'selected_original_indices', 'probe_candidates', 'teacher_weights', 'probe_confidence', 'distillation')
    if any((row.get(key) != [] for key in empty)) or row.get('distillation_update_count') != 0 or row.get('distillation_skipped') is not True:
        raise ValueError('Pilot rank abstention has missing or nonempty downstream execution')

def validate_pilot_rounds(directory, meta):
    method = meta['method']
    c = meta['config']
    n = c['env']['num_clients']
    path = Path(directory) / 'rounds.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.is_file() else []
    summary = dict(rounds=len(rows), positive_probe_rounds=0, rank_abstention_rounds=0, distillation_updates=0)
    if method == 'local':
        if rows:
            raise ValueError('Local pilot has federation rounds')
        return summary
    t = c['training']
    expected = list(range(t['warmup'] + t['communication_every'], t['requests_per_client'] + 1, t['communication_every']))
    if len(expected) < 2 or [q.get('requests_per_client') for q in rows] != expected:
        raise ValueError('Pilot round schedule incomplete')
    for q in rows:
        if q.get('skipped') or q.get('failed') or q.get('completed_aggregation_count') != 1:
            raise ValueError('Pilot skipped an aggregation round')
        if method == 'fedavg':
            if q.get('distillation_update_count', 0) != 0:
                raise ValueError('FedAvg has distillation')
            continue
        if len(q.get('geometry', [])) != n:
            raise ValueError('Pilot missing client geometry: ' + str(directory))
        if not q.get('selected_probes'):
            _validate_rank_abstention(q, c)
            summary['rank_abstention_rounds'] += 1
            continue
        summary['positive_probe_rounds'] += 1
        updates = q.get('distillation', [])
        expected_updates = c['federation']['distill_steps']
        if method == 'distillation_off':
            if updates or q.get('distillation_update_count') != 0:
                raise ValueError('Disabled distillation executed')
        elif len(updates) != n or any((d.get('distillation_update_count') != expected_updates for d in updates)) or q.get('distillation_update_count') != n * expected_updates:
            raise ValueError('Pilot client distillation incomplete')
        summary['distillation_updates'] += q['distillation_update_count']
    if method == 'full' and (not summary['positive_probe_rounds']):
        raise ValueError('Full pilot never exercised effective probes: ' + str(directory))
    return summary
