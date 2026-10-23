from copy import deepcopy
import hashlib
import math
import time
import numpy as np
import torch
from ..costs import payload_descriptor
OUTPUT_METHODS = ('fedhpd', 'fedslate')

def default_settings():
    return {'fedhpd': {'queries': 64, 'items': 50, 'query_seed': 49171, 'weight': 1.0}, 'fedslate': {'batch_size': 32, 'stage_bins': 4, 'contexts_per_bin': 4, 'sample_limit': 4096, 'hidden': 64, 'server_steps': 20, 'weight': 1.0}}

def validate_settings(c):
    settings = c.get('output_baselines')
    if settings is None:
        if c['method'] in OUTPUT_METHODS:
            raise ValueError('Output baselines require explicit settings')
        return
    defaults = default_settings()
    if not isinstance(settings, dict) or set(settings) != set(defaults):
        raise ValueError('Invalid output baseline settings')
    for method, values in settings.items():
        if not isinstance(values, dict) or set(values) != set(defaults[method]):
            raise ValueError('Incomplete output baseline settings')
        for key, value in values.items():
            if key == 'weight':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value < 0):
                    raise ValueError('Invalid output transfer weight')
            elif type(value) is not int or value < (0 if key == 'query_seed' else 1):
                raise ValueError('Invalid output baseline parameter ' + key)
    if settings['fedhpd']['items'] < c['env']['slate_size'] or settings['fedhpd']['items'] + 20 > c['env']['catalog_size']:
        raise ValueError('Public queries need disjoint history and candidate items')
    if c['method'] in OUTPUT_METHODS and c['agent'].get('critic_action_representation') != 'slate_mean_st':
        raise ValueError('Output baseline adaptations require public slate actions')

def public_queries(config, item_features, round_index):
    c = config['output_baselines']['fedhpd']
    e = config['env']
    rng = np.random.default_rng(np.random.SeedSequence([c['query_seed'], config['seed'], int(round_index)]))
    n = c['queries']
    steps = rng.integers(0, e['horizon'], size=n)
    lengths = np.minimum(20, steps * e['slate_size'])
    histories = np.full((n, 20), -1, dtype=np.int64)
    candidates = np.empty((n, c['items']), dtype=np.int64)
    rates = rng.uniform(0.1, 0.9, size=(n, 1, 3))
    feedback = (rng.random((n, 20, 3)) < rates).astype(np.float32)
    feedback[:, :, 1] *= feedback[:, :, 0]
    feedback[:, :, 2] *= feedback[:, :, 1]
    for i, length in enumerate(lengths):
        ids = rng.choice(len(item_features), size=int(length) + c['items'], replace=False)
        histories[i, :length] = ids[:length]
        candidates[i] = ids[length:]
        feedback[i, length:] = 0
    return dict(history_ids=histories, feedback=feedback, lengths=lengths, stages=(steps / e['horizon']).astype(np.float32), candidates=candidates)

def render_queries(query, item_features, client_id, config):
    f = item_features.shape[1]
    kind = client_id % 5
    length = 10 if kind in (0, 3) else 0 if kind == 2 else 20
    result = []
    for i, n in enumerate(query['lengths']):
        n = int(n)
        ids = query['history_ids'][i, :n]
        labels = query['feedback'][i, :n]
        time = np.array([query['stages'][i], 1 - query['stages'][i]], dtype=np.float32)
        history = np.zeros((length, f + 3), dtype=np.float32)
        if length:
            start = max(0, n - length)
            visible = labels[:, [0, 1]] if kind in (0, 3) else np.stack((labels[:, 2], 1 - labels[:, 1]), axis=-1)
            count = n - start
            history[:count, :f] = item_features[ids[start:]]
            history[:count, f:f + 2] = visible[start:]
            history[:count, -1] = 1
            context = time
        else:
            summary = np.zeros(f + 2, dtype=np.float32)
            if n:
                summary = np.concatenate((item_features[ids].mean(0), labels[:, :2].mean(0)))
            context = np.concatenate((summary, time))
        result.append(dict(history=history, context=context))
    return result

def send(messages, direction, client, kind, values):
    sender, receiver = (f'client:{client}', 'server') if direction == 'upload' else ('server', f'client:{client}')
    messages.append(payload_descriptor(direction, sender, receiver, kind, values))

def hpd_round(agents, config, round_index, state, log):
    query = public_queries(config, agents[0].item_features, round_index)
    responses = []
    latents = []
    messages = log['messages']
    for i, a in enumerate(agents):
        send(messages, 'download', i, 'hpd_public_queries', query)
        with torch.no_grad():
            z = a.encode_states(render_queries(query, a.item_features, i, config))
            latents.append(z)
            p = a.policy_response(z, query['candidates']).cpu().numpy()
            responses.append(p)
        send(messages, 'upload', i, 'hpd_probabilities', {'probabilities': p})
    teacher = np.mean(responses, axis=0, dtype=np.float64).astype(np.float32)
    fed = deepcopy(config['federation'])
    fed['lambda_distill'] = config['output_baselines']['fedhpd']['weight']
    fed.pop('transfer_guard', None)
    details = []
    for i, a in enumerate(agents):
        send(messages, 'download', i, 'hpd_teacher', {'teacher': teacher})
        details.append(a._distill_proposal(latents[i], query['candidates'], teacher, np.ones(len(teacher)), fed))
    digest = hashlib.sha256(b''.join((x.tobytes() for x in query.values()))).hexdigest()
    state['fedhpd'] = dict(round=int(round_index), query_sha256=digest, teacher=teacher.copy())
    log.update(completed_aggregation_count=1, distillation=details, distillation_update_count=sum((d['distillation_update_count'] for d in details)), public_query_count=len(teacher), query_sha256=digest, server_parameter_count=0, information_protocol='synthetic public histories + declared schema renderer; no response-model calls')

def output_round(agents, method, config, rng, round_index, state):
    if state is None:
        raise ValueError('Persistent output federation state is required')
    started = time.perf_counter()
    log = dict(round=int(round_index), method=method, skipped=False, messages=[], completed_aggregation_count=0, distillation_update_count=0)
    try:
        if method == 'fedhpd':
            hpd_round(agents, config, round_index, state, log)
        elif method == 'fedslate':
            from .slate import slate_round
            slate_round(agents, config, rng, round_index, state, log)
        else:
            raise ValueError('Unknown output baseline')
    except Exception as exc:
        log.update(failed=True, failure_reason=str(exc))
        exc.federation_log = log
        raise
    finally:
        log.update(communication_bytes=sum((m['payload_nbytes'] for m in log['messages'])), elapsed_seconds=time.perf_counter() - started)
    return log
