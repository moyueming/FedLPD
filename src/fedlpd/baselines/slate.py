from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .output import send

class QFusion(nn.Module):

    def __init__(self, clients, hidden):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(clients + 1, hidden), nn.ReLU(), nn.Linear(hidden, clients))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, q, stage, owner):
        row = torch.arange(len(q), device=q.device)
        return q[row, owner] + self.residual(torch.cat((q, stage[:, None]), dim=-1))[row, owner]

class LocalQEndpoint:

    def __init__(self, agent, settings, rng):
        self.agent = agent
        self.settings = settings
        self.batch = agent.replay.sample(settings['batch_size'], rng)
        self.count = len(self.batch)
        self.states = [r['state'] for r in self.batch]
        self.next_states = [r['next_state'] for r in self.batch]
        self.slates = np.stack([r['slate'] for r in self.batch])
        self.stages = np.array([s['context'][-2] for s in self.states], dtype=np.float32)
        self.next_stages = np.array([s['context'][-2] for s in self.next_states], dtype=np.float32)
        with torch.no_grad():
            z = agent.target_projector(agent.target_adapter(*agent._batch(self.next_states)))
            noise = torch.as_tensor(rng.normal(0, 0.2, size=(self.count, agent.action_dim)).clip(-0.5, 0.5), device=agent.device, dtype=torch.float32)
            scores = (agent.target_actor(z) + noise).clamp(-1, 1)
            next_slates = []
            for i, r in enumerate(self.batch):
                if r['done']:
                    next_slates.append(r['slate'].copy())
                    continue
                ids = np.asarray(r['next_candidate_ids'])
                logits = agent.scoring_features(True)[torch.as_tensor(ids, device=agent.device)] @ scores[i]
                chosen = torch.argsort(logits, descending=True, stable=True)[:agent.slate_size].cpu().numpy()
                next_slates.append(ids[chosen])
            self.next_slates = np.stack(next_slates)
        pool = agent.replay.sample(settings['sample_limit'], rng)
        bins = settings['stage_bins']
        self.contexts = {}
        self.fallback_bins = []
        for b in range(bins):
            candidates = [r['state'] for r in pool if self.bin(r['state']['context'][-2]) == b]
            if not candidates:
                continue
            ix = rng.choice(len(candidates), size=min(len(candidates), settings['contexts_per_bin']), replace=False)
            self.contexts[b] = [candidates[int(i)] for i in ix]

    def bin(self, stage):
        return min(self.settings['stage_bins'] - 1, max(0, int(float(stage) * self.settings['stage_bins'])))

    def request(self):
        return dict(slates=np.concatenate((self.slates, self.next_slates)), stages=np.concatenate((self.stages, self.next_stages)))

    def _q(self, states, slates, target=False):
        a = self.agent
        z = a.target_projector(a.target_adapter(*a._batch(states))) if target else a.encode_states(states)
        actions = a.features[torch.as_tensor(slates, device=a.device)].mean(1)
        q = (a.target_critic if target else a.critic)(z, actions)
        return torch.minimum(*q).flatten()

    @torch.no_grad()
    def respond(self, requests, offset):
        a = self.agent
        result = []
        for req in requests:
            size = len(req['stages']) // 2
            blocks = []
            for target in (False, True):
                ids = req['slates'][size if target else 0:None if target else size]
                stages = req['stages'][size if target else 0:None if target else size]
                values = np.empty(len(ids), dtype=np.float32)
                for b in sorted(set((self.bin(t) for t in stages))):
                    rows = np.array([i for i, t in enumerate(stages) if self.bin(t) == b])
                    chosen = min(self.contexts, key=lambda v: (abs(v - b), v))
                    if chosen != b:
                        self.fallback_bins.append((b, chosen))
                    states = self.contexts[chosen]
                    k = len(states)
                    z = a.target_projector(a.target_adapter(*a._batch(states))) if target else a.encode_states(states)
                    for start in range(0, len(rows), 128):
                        part = rows[start:start + 128]
                        actions = a.features[torch.as_tensor(ids[part], device=a.device)].mean(1)
                        q = (a.target_critic if target else a.critic)(z.repeat(len(part), 1), actions.repeat_interleave(k, dim=0))
                        values[part] = torch.minimum(*q).reshape(len(part), k).mean(1).cpu().numpy()
                blocks.append(values)
            result.append(np.concatenate(blocks))
        response = np.concatenate(result)
        response[offset:offset + self.count] = self._q(self.states, self.slates).cpu().numpy()
        response[offset + self.count:offset + 2 * self.count] = self._q(self.next_states, self.next_slates, True).cpu().numpy()
        return response

    def bellman_target(self, next_value):
        rewards = np.array([r['reward'] for r in self.batch], dtype=np.float32)
        discount = self.agent.gamma * np.array([not r['done'] for r in self.batch], dtype=np.float32)
        return rewards + discount * next_value

    def learn(self, teacher, steps, weight):
        a = self.agent
        if weight == 0:
            return dict(loss=0.0, updates=0)
        target = torch.as_tensor(teacher, device=a.device, dtype=torch.float32)[:, None]
        with torch.no_grad():
            z = a.encode_states(self.states)
        action = a.features[torch.as_tensor(self.slates, device=a.device)].mean(1)
        value = 0.0
        for _ in range(steps):
            q1, q2 = a.critic(z, action)
            loss = weight * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite FedSlate critic transfer')
            a.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(a.critic.parameters(), 10.0)
            a.critic_optimizer.step()
            value = float(loss.detach())
        if steps:
            with torch.no_grad():
                for p, t in zip(a.critic.parameters(), a.target_critic.parameters()):
                    t.lerp_(p, a.tau)
        return dict(loss=value, updates=steps)

def slate_round(agents, config, rng, round_index, state, log):
    settings = config['output_baselines']['fedslate']
    messages = log['messages']
    n = len(agents)
    if any((len(a.replay) == 0 for a in agents)):
        log.update(skipped=True, skip_reason='empty local replay')
        return
    local = [LocalQEndpoint(a, settings, rng) for a in agents]
    requests = [p.request() for p in local]
    offsets = np.cumsum([0] + [len(r['stages']) for r in requests])
    for i, r in enumerate(requests):
        send(messages, 'upload', i, 'slate_action_queries', r)
    queries = {key: np.concatenate([r[key] for r in requests]) for key in ('slates', 'stages')}
    responses = []
    for i, p in enumerate(local):
        send(messages, 'download', i, 'slate_action_queries', queries)
        q = p.respond(requests, int(offsets[i]))
        responses.append(q)
        send(messages, 'upload', i, 'slate_q_outputs', {'q': q})
    device = agents[0].device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(config['seed']) + 81021)
        model = QFusion(n, settings['hidden']).to(device)
    target = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['agent']['lr'])
    prior = state.get('fedslate')
    if prior:
        model.load_state_dict(prior['model'])
        target.load_state_dict(prior['target'])
        optimizer.load_state_dict(prior['optimizer'])
    q = torch.as_tensor(np.stack(responses, axis=-1), device=device)
    stages = torch.as_tensor(queries['stages'], device=device)
    owners = torch.cat([torch.full((len(r['stages']),), i, device=device, dtype=torch.long) for i, r in enumerate(requests)])
    current = np.concatenate([np.arange(offsets[i], offsets[i] + p.count) for i, p in enumerate(local)])
    nxt = np.concatenate([np.arange(offsets[i] + p.count, offsets[i + 1]) for i, p in enumerate(local)])
    with torch.no_grad():
        next_values = target(q[nxt], stages[nxt], owners[nxt]).cpu().numpy()
    targets = []
    start = 0
    for i, p in enumerate(local):
        value = next_values[start:start + p.count]
        start += p.count
        send(messages, 'download', i, 'slate_bootstrap_value', {'next_value': value})
        td = p.bellman_target(value)
        targets.append(td)
        send(messages, 'upload', i, 'slate_bellman_target', {'td_target': td})
    y = torch.as_tensor(np.concatenate(targets), device=device)
    losses = []
    for _ in range(settings['server_steps']):
        prediction = model(q[current], stages[current], owners[current])
        loss = F.mse_loss(prediction, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite FedSlate server loss')
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        fused = model(q[current], stages[current], owners[current]).cpu().numpy()
        for p, t in zip(model.parameters(), target.parameters()):
            t.lerp_(p, config['agent']['tau'])
    details = []
    start = 0
    for i, p in enumerate(local):
        value = fused[start:start + p.count]
        start += p.count
        send(messages, 'download', i, 'slate_fused_q', {'fused_q': value})
        details.append(p.learn(value, int(config['federation']['distill_steps']), settings['weight']))
    state['fedslate'] = dict(model=deepcopy(model.state_dict()), target=deepcopy(target.state_dict()), optimizer=deepcopy(optimizer.state_dict()), round=int(round_index))
    log.update(completed_aggregation_count=1, server_update_count=len(losses), server_parameter_count=sum((p.numel() for p in model.parameters())), server_loss_before=losses[0], server_loss_last=losses[-1], critic_transfer=details, critic_transfer_update_count=sum((d['updates'] for d in details)), q_query_count=len(q), q_response_evaluations=len(q) * n, stage_fallback_counts=[len(p.fallback_bins) for p in local], mean_fusion_correction=float(np.mean(np.abs(fused - q[current, owners[current]].cpu().numpy()))), information_protocol='public slate IDs + session stage + Q outputs + local Bellman scalars; independent private contexts')
