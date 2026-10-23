from copy import deepcopy
import math
import time
import torch
from ..agents import LocalAgent
from ..costs import payload_descriptor
BASELINES = ('fedrag', 'gpfedrec')

def default_settings():
    return {'fedrag': {'metric_weight': 1.0, 'proximal_weight': 0.01, 'angle_weight': 0.5, 'variance_min': 0.0001, 'variance_max': 1.0, 'model_lr': 0.001}, 'gpfedrec': {'regularization': 0.1, 'threshold_scale': 1.0, 'graph_layers': 1}}

def validate_settings(c):
    settings = c.get('baselines')
    if settings is None:
        if c['method'] in BASELINES:
            raise ValueError('Adapted baseline requires explicit baselines configuration')
        return
    defaults = default_settings()
    if not isinstance(settings, dict) or set(settings) != set(defaults):
        raise ValueError('Invalid baselines configuration')
    for method, values in settings.items():
        if not isinstance(values, dict) or set(values) != set(defaults[method]):
            raise ValueError('Incomplete ' + method + ' configuration')
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0):
                raise ValueError('Invalid baseline parameter ' + method + '.' + key)
    if settings['fedrag']['variance_min'] >= settings['fedrag']['variance_max']:
        raise ValueError('Invalid Gaussian variance bounds')
    if type(settings['gpfedrec']['graph_layers']) is not int:
        raise ValueError('graph_layers must be an integer')
    if c['method'] in BASELINES and c['agent'].get('critic_action_representation') != 'slate_mean_st':
        raise ValueError('Baseline adaptations require the public slate action protocol')

def make_agent(history_shape, context_dim, item_features, config, device='cpu'):
    method = config.get('method')
    if method == 'fedrag':
        from .rag import RAGAgent
        cls = RAGAgent
    elif method in ('gpfedrec', 'gp_uniform'):
        from .graph import GraphAgent
        cls = GraphAgent
    else:
        cls = LocalAgent
    return cls(history_shape, context_dim, item_features, config, device)

def baseline_round(agents, method, config, round_index):
    start = time.perf_counter()
    messages = []
    log = dict(round=int(round_index), method=method, skipped=False, messages=messages, completed_aggregation_count=0, distillation_update_count=0)
    try:
        if method == 'fedrag':
            from ..federation import average_projectors
            average_projectors(agents, messages)
        elif method in ('gpfedrec', 'gp_uniform'):
            from .graph import graph_aggregate
            tables = []
            for i, a in enumerate(agents):
                table = a.item_embedding.weight.detach().cpu().clone()
                tables.append(table)
                messages.append(payload_descriptor('upload', f'client:{i}', 'server', 'item_embedding_upload', {'items': table}))
            cfg = config['baselines']['gpfedrec']
            global_items, personal, weights, similarity = graph_aggregate(torch.stack(tables), cfg['threshold_scale'], cfg['graph_layers'])
            if method == 'gp_uniform':
                weights = torch.full_like(weights, 1 / len(tables))
                global_items = torch.stack(tables).mean(0)
                personal = global_items[None].expand_as(personal).clone()
            for i, a in enumerate(agents):
                a.load_item_round(global_items, personal[i])
                messages.append(payload_descriptor('download', 'server', f'client:{i}', 'item_embedding_download', {'global_items': global_items, 'personalized_items': personal[i]}))
            log.update(graph_weights=weights.tolist(), cosine_similarity=similarity.tolist(), graph_neighbor_counts=(weights > 0).sum(1).tolist(), personalized_target_mse=float((personal - global_items).square().mean()))
        else:
            raise ValueError('Unknown adapted baseline')
        log.update(completed_aggregation_count=1, client_learning=[deepcopy(a.baseline_statistics) for a in agents])
    except Exception as exc:
        log.update(failed=True, failure_reason=str(exc), communication_bytes=sum((m['payload_nbytes'] for m in messages)), elapsed_seconds=time.perf_counter() - start)
        exc.federation_log = log
        raise
    log.update(communication_bytes=sum((m['payload_nbytes'] for m in messages)), elapsed_seconds=time.perf_counter() - start)
    return log
