from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
DESIGN_VERSION = '2.0'
SEEDS = (2026, 2027, 2028)
ABLATION_GROUPS = {'behavioral_metric': ['metric_off'], 'cross_client_alignment': ['alignment_off'], 'latent_probes': ['global_probes', 'coverage_off', 'random_probes'], 'personalized_teacher': ['uniform_teacher'], 'policy_distillation': ['distillation_off']}
ABLATIONS = tuple((m for group in ABLATION_GROUPS.values() for m in group))
HISTORICAL_METHODS = ('prediction_error_off', 'support_teacher', 'uniform_confidence')
METHODS = ('random', 'local', 'fedavg', 'full') + ABLATIONS
METHOD_LABELS = dict(zip(METHODS, ('Random', 'Local RL', 'FedAvg', 'FedLPD', 'w/o Metric Loss', 'w/o Alignment', 'w/ Global Interpolation', 'w/o Coverage Filtering', 'w/ Random Selection', 'w/ Uniform Teacher', 'w/o Distillation')))
METHOD_LABELS['gpfedrec'] = 'GPFedRec-TD3'
METHOD_LABELS['fedrag'] = 'FedRAG-TD3'
METHODS += ('fedrag', 'gpfedrec')
METHODS += ('donor_filter_off',)
METHODS += ('central_shared',)
METHODS += ('transfer_guard_off',)
METHOD_LABELS['transfer_guard_off'] = 'v14 transfer (guard off)'
METHOD_LABELS['central_shared'] = 'Central shared-head diagnostic'
METHOD_LABELS['donor_filter_off'] = 'w/o Donor Filtering (v14 transfer)'
V20_METHODS = ('full', 'local', 'fedavg', 'fedrag', 'gpfedrec', 'direct_lpd', 'v14_reference', 'local_fixed', 'gp_uniform', 'alignment_off')
METHODS += ('direct_lpd', 'v14_reference', 'local_fixed', 'gp_uniform')
METHODS += ('fedhpd', 'fedslate')
V21_METHODS = V20_METHODS + ('fedhpd', 'fedslate')
V23_ABLATIONS = ('metric_off', 'alignment_off', 'signature_off', 'coverage_off', 'random_probes', 'uniform_teacher', 'distillation_off')
V23_BASELINES = ('local', 'fedavg', 'fedrag', 'gpfedrec', 'fedhpd', 'fedslate')
V23_METHODS = ('full',) + V23_BASELINES + V23_ABLATIONS
METHODS += ('signature_off',)
METHOD_LABELS['signature_off'] = 'w/o Signature Grounding'
METHOD_LABELS.update(fedhpd='FedHPD-TD3 (synthetic queries)', fedslate='FedSlate-TD3 (stage-conditioned)')
METHOD_LABELS.update(direct_lpd='LPD direct probes + learned items', v14_reference='v14 Full (fixed items)', local_fixed='v14 Local (fixed items)', gp_uniform='GP uniform averaging')

def default_config():
    return {'design_version': DESIGN_VERSION, 'purpose': 'formal', 'environment': 'recsim', 'method': 'full', 'seed': 2026, 'smoke': False, 'env': {'num_clients': 5, 'catalog_size': 2000, 'feature_dim': 16, 'candidate_size': 100, 'slate_size': 5, 'horizon': 20, 'initial_temper': 20, 'catalog_seed': 1729, 'partition_seed': 2718}, 'environment_overrides': {'kuaisim': {'env': {'catalog_size': 20000}, 'data': {'min_item_count': 25}}, 'recsim': {'env': {'catalog_size': 2000}, 'data': {'min_item_count': 5}}}, 'agent': {'adapter_dim': 64, 'latent_dim': 32, 'hidden_dim': 64, 'sequence_hidden': 32, 'statistic_hidden': 64, 'batch_size': 128, 'replay_capacity': 100000, 'learning_starts': 256, 'lr': 0.0003, 'gamma': 0.95, 'tau': 0.005, 'policy_delay': 2, 'exploration_noise': 0.1, 'actor_updates_encoder': False}, 'federation': {'prototypes': 16, 'anchors': 16, 'model_steps': 100, 'aux_steps': 300, 'probe_candidates': 64, 'probe_selected': 16, 'probe_items': 50, 'model_sample_limit': 4096, 'fixed_point_abs_bound': 0.001, 'fixed_point_relative_bound': 0.01, 'fixed_point_max_iterations': 1000, 'ot_solver_max_iterations': 100000, 'ot_marginal_tolerance': 1e-08, 'metric_alpha': 0.1, 'lambda_metric': 1.0, 'lambda_align': 1.0, 'lambda_distill': 1.0, 'geometry_optimizer': 'shared', 'geometry_lr': 0.0003, 'geometry_grad_clip': 10.0, 'geometry_schedule': 'aggregate_then_align', 'query_protocol': 'indexed_v1', 'coverage': {'mode': 'replay_knn_rank', 'shadow': False, 'neighbors': 8, 'tail_alpha': 0.05, 'sample_limit': 4096, 'min_reference': 64, 'min_calibration': 128, 'min_check': 128, 'min_episodes': 2, 'degenerate_radius': 1e-08}, 'distill_steps': 20, 'sigma_anchor': 1.0, 'sigma_probe': 1.0, 'sigma_profile': 1.0, 'model_hidden': 64, 'model_lr': 0.001, 'model_batch': 128, 'occupancy_sample_limit': 4096, 'anchor_lr': 0.03, 'anchor_iterations': 5, 'anchor_gradient_steps': 10, 'gw_iterations': 50, 'gw_tol_rel': 1e-07, 'gw_tol_abs': 1e-09, 'probe_neighbors': 3, 'support_threshold': 0.05, 'coverage_fraction': 0.6, 'transfer_temperature': 0.2, 'policy_temperature': 1.0, 'probability_epsilon': 1e-12}, 'training': {'requests_per_client': 40000, 'warmup': 8000, 'communication_every': 2000, 'updates_per_request': 1, 'update_every': 1, 'checkpoint_every': 2000, 'resume_every': 10000, 'cpu_threads': 4, 'deterministic': True}, 'evaluation': {'every': 4000, 'curve_sessions': 100, 'final_sessions': 500, 'comparison_requests': 40000, 'seed': 314159, 'validation_seed': 271828, 'validation_sessions': 100, 'calibration_sessions': 500, 'diagnostic_seed': 918273}, 'data': {'train_fraction': 0.7, 'val_fraction': 0.1, 'min_item_count': 5, 'static_feature_columns': [], 'duplicate_policy': 'drop_exact'}, 'response': {'epochs': 20, 'batch_size': 1024, 'num_workers': 4, 'hidden': 32, 'history_length': 20, 'seed': 1234, 'max_train_rows': 2000000, 'sample_window': 128, 'learning_rate': 0.001}, 'storage': {'mode': 'compact'}, 'cost': {'protocol_version': 2, 'controlled_load': False, 'target_fraction': 0.9, 'pilot_seed': 3026, 'targets_path': None, 'freeze_path': None, 'jobs_per_device': 1}}

def v14_config():
    c = default_config()
    c['agent'].update(critic_action_representation='slate_mean_st', slate_surrogate_temperature=1.0)
    c['evaluation'].update(validation_sessions=500, matched_comparison_all_methods=True)
    c['storage']['analysis_checkpoints'] = True
    c['preflight'] = {'mode': 'environment_audit_only'}
    return validate_config(c)

def v16_config():
    c = v14_config()
    c['federation'].update(donor_filter='replay_knn_rank', distillation_audit=True)
    return validate_config(c)

def v18_config():
    from .baselines import default_settings
    c = v14_config()
    c['baselines'] = default_settings()
    c['comparison'] = {'baselines': ['fedrag', 'gpfedrec']}
    return validate_config(c)

def v19_config():
    from .replay_guard import default_settings
    c = v18_config()
    c['purpose'] = 'optimization_trial'
    c['trial'] = {'version': 'v19', 'freeze_path': None}
    c['federation'].update(transfer_guard=default_settings(), distillation_audit=True)
    return validate_config(c)

def v20_config():
    c = v18_config()
    c['purpose'] = 'optimization_trial'
    c['trial'] = {'version': 'v20', 'freeze_path': None}
    c['agent'].update(learned_items=True, item_regularization=0.1)
    c['federation']['replay_bridge'] = {'neighbors': 4, 'sample_limit': 4096}
    c['federation']['distillation_audit'] = True
    return validate_config(c)

def v21_config():
    from .baselines.output import default_settings
    c = v20_config()
    c['trial']['version'] = 'v21'
    c['output_baselines'] = default_settings()
    return validate_config(c)

def v22_config():
    from .methodology import default_settings
    c = v21_config()
    c['trial']['version'] = 'v22'
    c['methodology'] = default_settings()
    return validate_config(c)

def v23_config():
    c = v22_config()
    c['trial']['version'] = 'v23'
    c['methodology']['version'] = 'action-grounded-v2'
    c['federation']['probe_selected'] = 8
    return validate_config(c)

def deep_merge(base, extra):
    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out

def environment_config(base, environment, *, concrete=False):
    c = deepcopy(base)
    overrides = c.get('environment_overrides', {})
    if c.get('smoke', False):
        overrides = {}
    for name, patch in overrides.items():
        if name not in ('kuaisim', 'recsim') or not isinstance(patch, dict):
            raise ValueError('Invalid environment_overrides')
        allowed = {'env': {'catalog_size'}, 'data': {'min_item_count'}}
        for section, values in patch.items():
            if section not in allowed or not isinstance(values, dict) or (not set(values) <= allowed[section]):
                raise ValueError('environment_overrides may only change catalog_size/min_item_count')
    c = deep_merge(c, overrides.get(environment, {}))
    c['environment'] = environment
    if concrete:
        c.pop('environment_overrides', None)
    return validate_config(c)

def load_config(path=None):
    if path is None:
        return default_config()
    p = Path(path)
    if p.suffix.lower() == '.json':
        content = json.loads(p.read_text(encoding='utf-8-sig'))
    else:
        import yaml
        content = yaml.safe_load(p.read_text(encoding='utf-8-sig'))
    federation = content.get('federation', {})
    for legacy, current in (('ot_tolerance', 'fixed_point_abs_bound'), ('ot_max_iterations', 'fixed_point_max_iterations')):
        if legacy in federation:
            if current in federation and federation[current] != federation[legacy]:
                raise ValueError('Conflicting ' + legacy + ' and ' + current)
            federation[current] = federation.pop(legacy)
    federation.pop('policy_samples', None)
    c = deep_merge(default_config(), content)
    if 'environment_overrides' not in content:
        c.pop('environment_overrides', None)
    else:
        c['environment_overrides'] = deepcopy(content['environment_overrides'])
    return environment_config(c, c['environment'])

def smoke_config(base=None):
    c = deepcopy(base or default_config())
    c['smoke'] = True
    c['purpose'] = 'smoke'
    c.pop('environment_overrides', None)
    c['storage'] = {'mode': 'full'}
    c['env'].update(catalog_size=64, feature_dim=8, candidate_size=16, horizon=4, initial_temper=4)
    c['agent'].update(adapter_dim=16, latent_dim=8, hidden_dim=16, sequence_hidden=16, statistic_hidden=16, batch_size=8, replay_capacity=256, learning_starts=8)
    c['federation'].update(prototypes=4, anchors=4, model_steps=2, aux_steps=1, probe_candidates=8, probe_selected=4, probe_items=8, model_sample_limit=64, distill_steps=1)
    c['training'].update(requests_per_client=24, warmup=8, communication_every=8, checkpoint_every=8, resume_every=8, cpu_threads=1)
    c['evaluation'].update(every=8, curve_sessions=2, final_sessions=3, comparison_requests=16, validation_sessions=2, calibration_sessions=3)
    if 'output_baselines' in c:
        c['output_baselines']['fedhpd'].update(queries=8, items=8)
        c['output_baselines']['fedslate'].update(batch_size=4, contexts_per_bin=2, sample_limit=32, hidden=16, server_steps=2)
    if 'methodology' in c:
        c['methodology'].update(reference_actions=2, signature_horizon=2, matching_steps=3, recalibration_steps=1, recalibration_alignment_steps=1)
        if c['methodology']['version'] == 'action-grounded-v2' and c['method'] in ('full', *V23_ABLATIONS):
            c['federation']['probe_selected'] = 2
    return validate_config(c)

def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')

def _number(value, name, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise ValueError(name + ' must be finite numeric')
    if positive and value <= 0:
        raise ValueError(name + ' must be positive')
    if minimum is not None and value < minimum:
        raise ValueError(name + ' below valid range')
    if maximum is not None and value > maximum:
        raise ValueError(name + ' above valid range')

def _validate_numeric(c):
    integer_groups = {'env': ('num_clients', 'catalog_size', 'feature_dim', 'candidate_size', 'slate_size', 'horizon'), 'agent': ('adapter_dim', 'latent_dim', 'hidden_dim', 'sequence_hidden', 'statistic_hidden', 'batch_size', 'replay_capacity', 'policy_delay'), 'training': ('requests_per_client', 'communication_every', 'updates_per_request', 'update_every', 'checkpoint_every', 'resume_every', 'cpu_threads'), 'evaluation': ('every', 'curve_sessions', 'final_sessions', 'validation_sessions', 'calibration_sessions', 'comparison_requests'), 'federation': ('prototypes', 'anchors', 'model_steps', 'probe_candidates', 'probe_selected', 'probe_items', 'model_sample_limit', 'model_hidden', 'model_batch', 'occupancy_sample_limit', 'anchor_iterations', 'anchor_gradient_steps', 'gw_iterations', 'probe_neighbors', 'fixed_point_max_iterations', 'ot_solver_max_iterations'), 'response': ('epochs', 'batch_size', 'hidden', 'history_length', 'sample_window'), 'data': ('min_item_count',), 'cost': ('protocol_version', 'jobs_per_device')}
    for section, keys in integer_groups.items():
        for key in keys:
            _integer(c[section][key], section + '.' + key)
    for section, key in (('training', 'warmup'), ('agent', 'learning_starts'), ('federation', 'aux_steps'), ('federation', 'distill_steps'), ('response', 'num_workers')):
        _integer(c[section][key], section + '.' + key, 0)
    for section, key in (('env', 'catalog_seed'), ('env', 'partition_seed'), ('evaluation', 'seed'), ('evaluation', 'validation_seed'), ('evaluation', 'diagnostic_seed'), ('response', 'seed'), ('cost', 'pilot_seed')):
        _integer(c[section][key], section + '.' + key, 0)
    _integer(c['seed'], 'seed', 0)
    if c['response'].get('max_train_rows') is not None:
        _integer(c['response']['max_train_rows'], 'response.max_train_rows')
    positive = {'agent': ('lr',), 'env': ('initial_temper',), 'response': ('learning_rate',), 'federation': ('fixed_point_abs_bound', 'fixed_point_relative_bound', 'ot_marginal_tolerance', 'sigma_anchor', 'sigma_probe', 'sigma_profile', 'model_lr', 'anchor_lr', 'transfer_temperature', 'policy_temperature', 'probability_epsilon', 'gw_tol_abs', 'gw_tol_rel')}
    for section, keys in positive.items():
        for key in keys:
            _number(c[section][key], section + '.' + key, positive=True)
    for key in ('metric_alpha', 'lambda_metric', 'lambda_align', 'lambda_distill'):
        _number(c['federation'][key], 'federation.' + key, minimum=0)
    _number(c['agent']['exploration_noise'], 'agent.exploration_noise', minimum=0)
    for section, key in (('agent', 'gamma'), ('agent', 'tau'), ('federation', 'support_threshold'), ('federation', 'coverage_fraction')):
        _number(c[section][key], section + '.' + key, minimum=0, maximum=1)
    if c['agent']['gamma'] >= 1:
        raise ValueError('agent.gamma must be less than one')
    _number(c['cost']['target_fraction'], 'cost.target_fraction', maximum=1, positive=True)
    if c['cost']['pilot_seed'] in SEEDS:
        raise ValueError('cost.pilot_seed must be independent of the formal training seeds')
    for key in ('train_fraction', 'val_fraction'):
        _number(c['data'][key], 'data.' + key, maximum=1, positive=True)
    if c['data']['train_fraction'] + c['data']['val_fraction'] >= 1:
        raise ValueError('data.train_fraction + val_fraction must leave a test split')
    f = c['federation']
    if f['anchors'] < 2 or f['prototypes'] < 2:
        raise ValueError('federation.anchors and prototypes must both be at least two')
    if f['probe_candidates'] < f['anchors']:
        raise ValueError('probe_candidates must include every anchor')
    if f['probe_selected'] > f['probe_candidates']:
        raise ValueError('probe_selected exceeds probe_candidates')
    if f['model_sample_limit'] < f['prototypes']:
        raise ValueError('model_sample_limit is smaller than prototypes')
    if c['agent']['replay_capacity'] < max(c['agent']['batch_size'], c['agent']['learning_starts']):
        raise ValueError('replay_capacity cannot reach the configured learning threshold')
    if f['probability_epsilon'] >= 1:
        raise ValueError('probability_epsilon must be less than one')
    for key in ('reward_min', 'reward_max'):
        if key in f:
            _number(f[key], 'federation.' + key)
    if f.get('reward_min', 0.0) >= f.get('reward_max', 1.0):
        raise ValueError('reward_min must be smaller than reward_max')
    if c['evaluation']['calibration_sessions'] < c['evaluation']['validation_sessions']:
        raise ValueError('calibration_sessions must contain the validation session prefix')
    for section, key in (('training', 'deterministic'), ('cost', 'controlled_load')):
        if type(c[section][key]) is not bool:
            raise ValueError(section + '.' + key + ' must be boolean')
    if type(c['smoke']) is not bool:
        raise ValueError('smoke must be boolean')

def validate_config(c):
    _validate_numeric(c)
    for section, key in (('cost', 'require_calibrated_target'), ('evaluation', 'validation_every_round')):
        if type(c[section].get(key, False)) is not bool:
            raise ValueError(section + '.' + key + ' must be boolean')
    if c['cost'].get('require_calibrated_target'):
        if c.get('trial', {}).get('version') != 'v23' or not c['evaluation'].get('validation_every_round'):
            raise ValueError('Calibrated trial target requires v23 and validation every round')
    if c['method'] == 'signature_off' and c.get('methodology', {}).get('version') != 'action-grounded-v2':
        raise ValueError('signature_off requires the v23 methodology definition')
    from .methodology import validate_settings as validate_methodology
    validate_methodology(c)
    from .replay_guard import validate_settings as validate_guard
    validate_guard(c)
    if c['method'] == 'transfer_guard_off' and (not c['federation'].get('transfer_guard')):
        raise ValueError('transfer_guard_off requires an explicit transfer_guard')
    if c.get('trial'):
        if c['trial'].get('version') not in ('v19', 'v20', 'v21', 'v22', 'v23') or set(c['trial']) != {'version', 'freeze_path'}:
            raise ValueError('Unknown optimization trial')
        if not c['smoke'] and c.get('purpose') != 'optimization_trial':
            raise ValueError('Trial must be labeled optimization_trial')
        if c.get('study') or c['env'].get('training_candidate_bias'):
            raise ValueError('v19 trial uses the original v14 task')
    if type(c['agent'].get('learned_items', False)) is not bool:
        raise ValueError('learned_items must be boolean')
    _number(c['agent'].get('item_regularization', 0.1), 'item_regularization', minimum=0)
    bridge = c['federation'].get('replay_bridge')
    if bridge is not None:
        if not isinstance(bridge, dict) or set(bridge) != {'neighbors', 'sample_limit'}:
            raise ValueError('Invalid replay_bridge')
        for k, vv in bridge.items():
            _integer(vv, 'replay_bridge.' + k)
        if bridge['neighbors'] > bridge['sample_limit']:
            raise ValueError('Invalid bridge sample size')
        if c['federation'].get('transfer_guard'):
            raise ValueError('Replay bridge cannot use synthetic-probe transfer guard')
    donor = c['federation'].get('donor_filter', 'legacy')
    if donor not in ('legacy', 'replay_knn_rank'):
        raise ValueError('Invalid federation.donor_filter')
    if type(c['federation'].get('distillation_audit', False)) is not bool:
        raise ValueError('distillation_audit must be boolean')
    if donor == 'replay_knn_rank' and c['federation'].get('coverage', {}).get('mode') != 'replay_knn_rank':
        raise ValueError('donor_filter requires replay_knn_rank coverage')
    if c['method'] == 'donor_filter_off' and donor != 'replay_knn_rank':
        raise ValueError('donor_filter_off requires donor_filter config')
    if c['agent'].get('critic_action_representation', 'score_vector') not in ('score_vector', 'slate_mean_st'):
        raise ValueError('Invalid agent.critic_action_representation')
    _number(c['agent'].get('slate_surrogate_temperature', 1.0), 'agent.slate_surrogate_temperature', positive=True)
    if c.get('preflight', {}).get('mode', 'all_paths') not in ('all_paths', 'environment_audit_only'):
        raise ValueError('Invalid preflight.mode')
    for section, key in (('evaluation', 'matched_comparison_all_methods'), ('storage', 'analysis_checkpoints')):
        if type(c.get(section, {}).get(key, False)) is not bool:
            raise ValueError(section + '.' + key + ' must be boolean')
    from .coverage import validate_settings
    validate_settings(c)
    if type(c['agent'].get('actor_updates_encoder', True)) is not bool:
        raise ValueError('agent.actor_updates_encoder must be boolean')
    if c['federation'].get('geometry_optimizer', 'shared') not in ('shared', 'separate'):
        raise ValueError('Invalid federation.geometry_optimizer')
    if c['federation'].get('geometry_schedule', 'align_then_aggregate') not in ('align_then_aggregate', 'aggregate_then_align'):
        raise ValueError('Invalid federation.geometry_schedule')
    for key in ('geometry_lr', 'geometry_grad_clip'):
        if key in c['federation']:
            _number(c['federation'][key], 'federation.' + key, positive=True)
    diagnostic = c.get('diagnostics', {})
    if diagnostic:
        if type(diagnostic.get('enabled', False)) is not bool:
            raise ValueError('diagnostics.enabled must be boolean')
        if type(diagnostic.get('coordinate_audit', False)) is not bool:
            raise ValueError('diagnostics.coordinate_audit must be boolean')
        if diagnostic.get('coordinate_audit') and (not diagnostic.get('enabled') or c['federation'].get('query_protocol') != 'indexed_v1'):
            raise ValueError('coordinate_audit requires enabled diagnostics and indexed_v1 queries')
        if type(diagnostic.get('fitting_audit', False)) is not bool:
            raise ValueError('diagnostics.fitting_audit must be boolean')
        if diagnostic.get('fitting_audit'):
            if not diagnostic.get('coordinate_audit') or c.get('cost', {}).get('controlled_load'):
                raise ValueError('fitting_audit requires coordinate_audit and uncontrolled diagnostic cost')
            steps = diagnostic.get('fitting_steps')
            if not isinstance(steps, list) or not steps or any((type(v) is not int or not 1 <= v <= 1000 for v in steps)) or (steps != sorted(set(steps))):
                raise ValueError('fitting_steps must be increasing positive integers, at most 1000')
        if diagnostic.get('enabled') and c.get('purpose') != 'pilot' and (not c['smoke']):
            raise ValueError('Learning diagnostics require pilot purpose; formal timing must remain unchanged')
    if c['data'].get('duplicate_policy', 'reject') not in ('reject', 'drop_exact'):
        raise ValueError('Invalid data.duplicate_policy')
    if c['environment'] not in ('kuaisim', 'recsim'):
        raise ValueError('Unknown environment')
    from .baselines import validate_settings as validate_baselines, BASELINES
    validate_baselines(c)
    from .baselines.output import validate_settings as validate_outputs
    validate_outputs(c)
    extra = c.get('comparison', {}).get('baselines', [])
    if not isinstance(extra, list) or len(extra) != len(set(extra)) or any((m not in BASELINES for m in extra)):
        raise ValueError('Invalid comparison baselines')
    if extra and (not c.get('baselines')):
        raise ValueError('Comparison requires explicit baseline settings')
    if c['method'] in HISTORICAL_METHODS:
        raise ValueError('This historical method is outside design v2.0')
    if c['method'] not in METHODS:
        raise ValueError('Unknown method')
    if c['method'] == 'central_shared' and (not c.get('study')):
        raise ValueError('Central shared heads require an explicit task study')
    bias = c['env'].get('training_candidate_bias')
    if bias is not None:
        if not isinstance(bias, dict) or set(bias) != {'strength', 'groups', 'version'}:
            raise ValueError('Invalid training candidate bias')
        _number(bias['strength'], 'candidate bias strength', minimum=0, maximum=0.95)
        if bias['groups'] != 5 or bias['version'] != 1:
            raise ValueError('Unsupported candidate grouping protocol')
    if c.get('study'):
        s = c['study']
        if s.get('version') != 'v17' or s.get('scenario') not in ('original', 'complementary'):
            raise ValueError('Unknown study version/scenario')
        if c['method'] not in ('local', 'fedavg', 'full', 'central_shared', *BASELINES):
            raise ValueError('Method outside task study')
        if not c['smoke'] and c.get('purpose') != 'task_study':
            raise ValueError('Task study must be separately labeled')
        if (s['scenario'] == 'complementary') != bool(bias):
            raise ValueError('Scenario and training sampling disagree')
    if c.get('storage', {}).get('mode', 'full') not in ('full', 'compact'):
        raise ValueError('Unknown storage.mode')
    e, t, v = (c['env'], c['training'], c['evaluation'])
    if not c.get('smoke', False) and (e['num_clients'] != 5 or e['slate_size'] != 5):
        raise ValueError('Formal protocol requires five clients and Top-5')
    for key in ('num_clients', 'catalog_size', 'feature_dim', 'candidate_size', 'slate_size', 'horizon'):
        if e[key] < 1:
            raise ValueError(f'env.{key} must be positive')
    if not e['slate_size'] <= e['candidate_size'] <= e['catalog_size']:
        raise ValueError('Invalid candidate/slate/catalog sizes')
    if e['catalog_size'] < e['candidate_size'] + (e['horizon'] - 1) * e['slate_size']:
        raise ValueError('catalog_size cannot support legal candidates through the full horizon')
    for key in ('requests_per_client', 'communication_every', 'checkpoint_every', 'resume_every', 'update_every'):
        if t[key] < 1:
            raise ValueError(f'training.{key} must be positive')
    if t['warmup'] < 0 or t['updates_per_request'] < 1:
        raise ValueError('Invalid training budget')
    if t['warmup'] % t['communication_every']:
        raise ValueError('warmup must be a multiple of communication_every')
    for key in ('every', 'curve_sessions', 'final_sessions', 'validation_sessions', 'calibration_sessions'):
        if v[key] < 1:
            raise ValueError(f'evaluation.{key} must be positive')
    if v['final_sessions'] < v['curve_sessions']:
        raise ValueError('Final evaluation too small')
    return c

def stable_hash(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def task_hash(c):
    return stable_hash({'design_version': c.get('design_version', DESIGN_VERSION), 'environment': c['environment'], 'env': c['env'], 'gamma': c['agent']['gamma'], 'data': c.get('data', {}), 'assets': {k: v for k, v in c.get('assets', {}).items() if k != 'implementation_sha256'}, 'smoke': c.get('smoke', False)})

def evaluation_hash(c):
    return stable_hash({'evaluation': c['evaluation'], 'gamma': c['agent']['gamma'], 'task_hash': task_hash(c)})

def protocol_hash(c):
    shared = {k: v for k, v in c['training'].items() if k not in ('requests_per_client', 'cpu_threads', 'resume_every', 'checkpoint_every')}
    return stable_hash({'task_hash': task_hash(c), 'evaluation_hash': evaluation_hash(c), 'training': shared})

def build_matrix(base=None, include_random=False):
    base = deepcopy(base or default_config())
    jobs = []
    if base.get('trial', {}).get('version') in ('v20', 'v21', 'v22', 'v23'):
        for env in ('kuaisim', 'recsim'):
            methods = V23_METHODS if base['trial']['version'] == 'v23' else V21_METHODS if base['trial']['version'] in ('v21', 'v22') else V20_METHODS
            for method in methods + (('random',) if include_random else ()):
                for seed in SEEDS:
                    c = environment_config(base, env, concrete=True)
                    c.update(method=method, seed=seed)
                    if base['trial']['version'] == 'v23' and method in (*V23_BASELINES, 'random'):
                        c['federation']['probe_selected'] = 16
                    c['training']['requests_per_client'] = 80000 if env == 'kuaisim' else 40000
                    jobs.append(validate_config(smoke_config(c) if c.get('smoke') else c))
        return jobs
    for env in ('kuaisim', 'recsim'):
        extra = tuple(base.get('comparison', {}).get('baselines', []))
        methods = ('full', 'local', 'fedavg') + extra + (ABLATIONS if env == 'kuaisim' else ('alignment_off',))
        if base['federation'].get('donor_filter') == 'replay_knn_rank':
            methods += ('donor_filter_off',)
        if base['federation'].get('transfer_guard'):
            methods += ('transfer_guard_off',)
        if include_random:
            methods += ('random',)
        for method in methods:
            for seed in (2026, 2027, 2028):
                c = environment_config(base, env, concrete=True)
                c.update(method=method, seed=seed)
                c['training']['requests_per_client'] = 80000 if env == 'kuaisim' and method in ('full', 'local', 'fedavg', 'random', 'donor_filter_off', 'transfer_guard_off', *extra) else 40000
                jobs.append(validate_config(smoke_config(c) if c.get('smoke') else c))
    return jobs

def run_id(c):
    prefix = 'smoke-' if c.get('smoke') else ''
    if c.get('study'):
        prefix += f"study-{c['study']['version']}-{c['study']['scenario']}-"
    return f"{prefix}{c['environment']}-{c['method']}-seed{c['seed']}"

def scheduled_round(config, requests):
    t = config['training']
    return max(0, (int(requests) - t['warmup']) // t['communication_every'])

def validation_steps(config):
    t = config['training']
    budget = t['requests_per_client']
    warmup = t['warmup']
    if config['evaluation'].get('validation_every_round'):
        return sorted({0, budget, budget // 2} | ({warmup} if warmup <= budget else set()) | set(range(config['evaluation']['every'], budget + 1, config['evaluation']['every'])) | set(range(warmup + t['communication_every'], budget + 1, t['communication_every'])))
    if config.get('trial'):
        return sorted({0, budget, budget // 2} | ({warmup} if warmup <= budget else set()) | set(range(config['evaluation']['every'], budget + 1, config['evaluation']['every'])))
    if config.get('study'):
        return sorted({0, budget} | ({warmup} if warmup <= budget else set()) | set(range(config['evaluation']['every'], budget + 1, config['evaluation']['every'])))
    return sorted({0} | ({warmup} if warmup <= budget else set()) | set(range(warmup + t['communication_every'], budget + 1, t['communication_every'])))

def backbone_hash(c):
    agent = deepcopy(c['agent'])
    if c.get('trial', {}).get('version') in ('v20', 'v21', 'v22', 'v23'):
        agent['learned_items'] = c['method'] not in ('v14_reference', 'local_fixed', 'random')
    return stable_hash(agent)

def method_config_hash(c):
    value = {'method': c['method'], 'agent': c['agent'], 'federation': c['federation']}
    from .methodology import PAPER_METHODS
    if c.get('methodology') and c['method'] in PAPER_METHODS:
        value['methodology'] = c['methodology']
    if c['method'] in ('fedrag', 'gpfedrec'):
        value['baseline'] = c['baselines'][c['method']]
    if c['method'] in ('fedhpd', 'fedslate'):
        value['baseline'] = c['output_baselines'][c['method']]
    return stable_hash(value)
