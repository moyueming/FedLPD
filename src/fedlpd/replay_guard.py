from copy import deepcopy
import time
import numpy as np
import torch

def default_settings():
    return dict(mode='replay_twin_q', sample_size=256, max_kl=0.01, scales=[1.0, 0.5, 0.25, 0.125, 0.0625], q_tolerance=1e-08)

def validate_settings(config):
    s = config.get('federation', {}).get('transfer_guard')
    if s is None:
        return
    from .config import _number, _integer
    if not isinstance(s, dict) or set(s) != set(default_settings()) or s['mode'] != 'replay_twin_q':
        raise ValueError('Invalid transfer_guard settings')
    _integer(s['sample_size'], 'guard sample_size', minimum=1)
    _number(s['max_kl'], 'guard max_kl', positive=True)
    _number(s['q_tolerance'], 'guard q_tolerance', minimum=0)
    scales = s['scales']
    if not isinstance(scales, list) or not scales or scales[0] != 1.0:
        raise ValueError('Guard scales must start with 1')
    for x in scales:
        _number(x, 'guard scale', positive=True, maximum=1)
    if scales != sorted(set(scales), reverse=True):
        raise ValueError('Guard scales must strictly decrease')
    if config['agent'].get('critic_action_representation') != 'slate_mean_st':
        raise ValueError('Replay guard requires actual slate critic actions')

@torch.no_grad()
def _evaluate(agent, z, pools):
    scores = agent.actor(z)
    action = agent.slate_action(scores, pools)
    q1, q2 = agent.critic(z, action)
    temperature = float(agent.config.get('federation', {}).get('policy_temperature', 1.0))
    logp = []
    slates = []
    for score, ids in zip(scores, pools):
        ids = torch.as_tensor(ids, device=agent.device, dtype=torch.long)
        logits = agent.features[ids] @ score / temperature
        logp.append(torch.log_softmax(logits, dim=0))
        chosen = torch.argsort(logits, descending=True, stable=True)[:agent.slate_size]
        slates.append(ids[chosen])
    return (q1.flatten(), q2.flatten(), logp, torch.stack(slates))

def guarded_proposal(agent, propose, settings):
    started = time.perf_counter()
    guard = dict(mode=settings['mode'], accepted_scale=0.0, attempts=[], sample_size=0, interpretation='Local replay twin-Q and policy-KL proxy; not observed evaluation gain.')
    if not len(agent.replay):
        guard.update(reason='empty_replay', seconds=time.perf_counter() - started)
        return dict(distillation_loss=0.0, distillation_update_count=0, transfer_guard=guard)
    rng = np.random.default_rng(np.random.SeedSequence([int(agent.config.get('seed', 0)), 6567, int(agent.updates), len(agent.replay), int(agent.replay.position)]))
    batch = agent.replay.sample(int(settings['sample_size']), rng)
    pools = [b['candidate_ids'] for b in batch]
    with torch.no_grad():
        z = agent.encode_states([b['state'] for b in batch]).detach()
    reference = _evaluate(agent, z, pools)
    before = deepcopy(agent.actor.state_dict())
    optimizer = deepcopy(agent.actor_optimizer.state_dict())
    guard['sample_size'] = len(batch)
    try:
        result = propose()
        if 'audit' in result:
            result['proposal_audit'] = result.pop('audit')
        candidate = deepcopy(agent.actor.state_dict())
        for scale in settings['scales']:
            agent.actor.load_state_dict(candidate if scale == 1.0 else {k: before[k] + float(scale) * (v - before[k]) for k, v in candidate.items()})
            q1, q2, logp, slates = _evaluate(agent, z, pools)
            kl = torch.stack([(old.exp() * (old - new)).sum() for old, new in zip(reference[2], logp)]).mean()
            d1, d2 = ((q1 - reference[0]).mean(), (q2 - reference[1]).mean())
            finite = all((bool(torch.isfinite(x)) for x in (d1, d2, kl)))
            accepted = finite and float(d1) >= -settings['q_tolerance'] and (float(d2) >= -settings['q_tolerance']) and (float(kl) <= settings['max_kl'])
            guard['attempts'].append(dict(scale=float(scale), q1_delta=float(d1) if torch.isfinite(d1) else None, q2_delta=float(d2) if torch.isfinite(d2) else None, kl=float(kl) if torch.isfinite(kl) else None, changed_slate_fraction=float((slates != reference[3]).any(1).float().mean()), accepted=accepted))
            if accepted:
                guard['accepted_scale'] = float(scale)
                break
        if guard['accepted_scale'] < 1.0:
            agent.actor_optimizer.load_state_dict(optimizer)
        if guard['accepted_scale'] == 0.0:
            agent.actor.load_state_dict(before)
        agent.actor_optimizer.zero_grad(set_to_none=True)
    except BaseException:
        agent.actor.load_state_dict(before)
        agent.actor_optimizer.load_state_dict(optimizer)
        agent.actor_optimizer.zero_grad(set_to_none=True)
        raise
    guard.update(reason='accepted' if guard['accepted_scale'] else 'no_acceptable_step', seconds=time.perf_counter() - started, retained_replay_kl=guard['attempts'][-1]['kl'] if guard['accepted_scale'] else 0.0)
    result['transfer_guard'] = guard
    return result
