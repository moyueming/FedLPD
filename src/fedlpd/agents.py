from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .replay import ReplayBuffer, state_of

class Adapter(nn.Module):

    def __init__(self, history_shape, context_dim, adapter_dim, hidden_dim, sequence_hidden=32, has_valid_bit=True):
        super().__init__()
        self.has_history = history_shape[0] > 0
        self.has_valid_bit = has_valid_bit
        if self.has_history:
            self.gru = nn.GRU(history_shape[1], sequence_hidden, batch_first=True)
            self.output = nn.Sequential(nn.Linear(context_dim + sequence_hidden, adapter_dim), nn.Tanh())
        else:
            self.output = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, adapter_dim), nn.Tanh())

    def forward(self, history, context):
        if self.has_history:
            if self.has_valid_bit:
                valid = history[:, :, -1] > 0
                lengths = valid.sum(1)
                hidden = history.new_zeros((len(history), self.gru.hidden_size))
                nonempty = torch.nonzero(lengths > 0, as_tuple=True)[0]
                if len(nonempty):
                    sequences = [history[i][valid[i]] for i in nonempty]
                    packed = nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
                    _, encoded = self.gru(packed)
                    hidden = hidden.index_copy(0, nonempty, encoded[-1])
            else:
                _, encoded = self.gru(history)
                hidden = encoded[-1]
            context = torch.cat([hidden, context], dim=-1)
        return self.output(context)

def mlp(i, h, o, final=None):
    layers = [nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o)]
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)

class TwinCritic(nn.Module):

    def __init__(self, z, a, h):
        super().__init__()
        self.q1 = mlp(z + a, h, 1)
        self.q2 = mlp(z + a, h, 1)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return (self.q1(x), self.q2(x))

class LocalAgent:

    def __init__(self, history_shape, context_dim, item_features, config, device='cpu'):
        self.config = copy.deepcopy(config)
        cfg = config.get('agent', config)
        self.device = torch.device(device)
        self.history_shape = tuple(history_shape)
        self.context_dim = int(context_dim)
        self.item_features = np.asarray(item_features, dtype=np.float32).copy()
        self.features = torch.as_tensor(self.item_features, device=self.device)
        self.slate_size = int(config.get('env', config.get('environment', {}) if isinstance(config.get('environment'), dict) else {}).get('slate_size', 5))
        self.horizon = int(config.get('env', config.get('environment', {}) if isinstance(config.get('environment'), dict) else {}).get('horizon', 20))
        self.action_dim = self.features.shape[1]
        self.adapter_dim = int(cfg.get('adapter_dim', 64))
        self.latent_dim = int(cfg.get('latent_dim', 32))
        h = int(cfg.get('hidden_dim', 64))
        self.gamma = float(cfg.get('gamma', 0.95))
        self.tau = float(cfg.get('tau', 0.005))
        self.batch_size = int(cfg.get('batch_size', 128))
        self.learning_starts = int(cfg.get('learning_starts', 256))
        self.policy_delay = int(cfg.get('policy_delay', 2))
        self.exploration_noise = float(cfg.get('exploration_noise', 0.1))
        self.critic_action_representation = cfg.get('critic_action_representation', 'score_vector')
        self.slate_surrogate_temperature = float(cfg.get('slate_surrogate_temperature', 1.0))
        if self.critic_action_representation not in ('score_vector', 'slate_mean_st'):
            raise ValueError('Unknown critic action representation')
        if not np.isfinite(self.slate_surrogate_temperature) or self.slate_surrogate_temperature <= 0:
            raise ValueError('Invalid slate surrogate temperature')
        self.adapter = Adapter(self.history_shape, self.context_dim, self.adapter_dim, int(cfg.get('statistic_hidden', 64)), sequence_hidden=int(cfg.get('sequence_hidden', 32)), has_valid_bit=self.history_shape[1] == self.action_dim + 3).to(self.device)
        self.projector = nn.Sequential(nn.Linear(self.adapter_dim, self.adapter_dim), nn.ReLU(), nn.Linear(self.adapter_dim, self.latent_dim)).to(self.device)
        self.actor = mlp(self.latent_dim, h, self.action_dim, nn.Tanh()).to(self.device)
        self.critic = TwinCritic(self.latent_dim, self.action_dim, h).to(self.device)
        self.target_adapter = copy.deepcopy(self.adapter)
        self.target_projector = copy.deepcopy(self.projector)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        lr = float(cfg.get('lr', 0.0003))
        self.encoder_optimizer = torch.optim.Adam(list(self.adapter.parameters()) + list(self.projector.parameters()), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_updates_encoder = bool(cfg.get('actor_updates_encoder', True))
        fed = config.get('federation', {})
        self.geometry_optimizer = torch.optim.Adam(list(self.adapter.parameters()) + list(self.projector.parameters()), lr=float(fed.get('geometry_lr', lr))) if fed.get('geometry_optimizer', 'shared') == 'separate' else None
        self.replay = ReplayBuffer(cfg.get('replay_capacity', 100000))
        self.updates = 0
        self._metric_cache = None
        self._estimator_state = None
        self._learning_sums = {}
        self._learning_counts = {}
        if cfg.get('learned_items', False) and config.get('method') not in ('v14_reference', 'local_fixed', 'random', 'gpfedrec', 'gp_uniform'):
            self.item_embedding = nn.Embedding.from_pretrained(self.features.detach().clone(), freeze=False)
            self.target_items = self.features.detach().clone()
            self.item_optimizer = torch.optim.Adam(self.item_embedding.parameters(), lr=lr)

    def learning_summary(self):
        return dict(critic_updates=self._learning_counts.get('critic_loss', 0), actor_updates=self._learning_counts.get('actor_loss', 0), replay_size=len(self.replay), means={k: v / self._learning_counts[k] for k, v in self._learning_sums.items()})

    def _batch(self, states):
        values = [state_of(s) for s in states]
        return (torch.as_tensor(np.stack([v['history'] for v in values]), device=self.device, dtype=torch.float32), torch.as_tensor(np.stack([v['context'] for v in values]), device=self.device, dtype=torch.float32))

    def encode_states(self, states):
        history, context = self._batch(states)
        return self.projector(self.adapter(history, context))

    def adapted_states(self, states):
        return self.adapter(*self._batch(states))

    def act(self, obs, rng, explore=False):
        with torch.no_grad():
            action = self.actor(self.encode_states([obs]))[0].cpu().numpy()
        if explore:
            action = np.clip(action + rng.normal(0, self.exploration_noise, size=action.shape), -1, 1).astype(np.float32)
        ids = np.asarray(obs.candidate_ids, dtype=np.int64)
        if len(ids) < self.slate_size:
            raise ValueError('Candidate set smaller than slate')
        if hasattr(self, 'item_embedding'):
            with torch.no_grad():
                features = self.scoring_features()[torch.as_tensor(ids, device=self.device)].cpu().numpy()
        else:
            features = self.item_features[ids]
        scores = features @ action
        ranking = np.argsort(-scores, kind='stable')[:self.slate_size]
        return (ids[ranking].copy(), action.copy())

    def observe(self, obs, action_vector, slate_ids, reward, next_obs, done, episode_id):
        step = int(round(float(obs.context[-2]) * self.horizon))
        self.replay.add(obs, action_vector, slate_ids, reward, next_obs, done, episode_id, step_in_episode=step)

    def scoring_features(self, target=False):
        if hasattr(self, 'item_embedding'):
            return self.target_items if target else self.item_embedding.weight
        return self.features

    def auxiliary_update(self, batch):
        return {}

    def actor_regularization(self):
        if hasattr(self, 'item_embedding'):
            return self.config['agent'].get('item_regularization', 0.1) * F.mse_loss(self.item_embedding.weight, self.features)
        return None

    def slate_action(self, scores, candidate_pools, straight_through=False, target=False):
        pools = [np.asarray(ids, dtype=np.int64) for ids in candidate_pools]
        if len(pools) != len(scores) or not pools or any((ids.ndim != 1 or len(ids) < self.slate_size for ids in pools)):
            raise ValueError('A nonterminal action requires enough legal candidates')
        lengths = np.array([len(ids) for ids in pools])
        width = int(lengths.max())
        ids = np.stack(pools) if np.all(lengths == width) else np.stack([np.pad(p, (0, width - len(p))) for p in pools])
        if ids.min() < 0 or ids.max() >= len(self.item_features):
            raise ValueError('Invalid candidate item ID')
        features = self.features[torch.as_tensor(ids, device=self.device)]
        valid = torch.arange(width, device=self.device)[None, :] < torch.as_tensor(lengths, device=self.device)[:, None]
        scoring = self.scoring_features(target)[torch.as_tensor(ids, device=self.device)]
        logits = torch.einsum('bd,bnd->bn', scores, scoring).masked_fill(~valid, -torch.inf)
        chosen = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :self.slate_size]
        hard = features.gather(1, chosen[:, :, None].expand(-1, -1, self.action_dim)).mean(1)
        if not straight_through:
            return hard
        weights = torch.softmax(logits / self.slate_surrogate_temperature, dim=-1)
        soft = torch.einsum('bn,bnd->bd', weights, features)
        return hard + (soft - soft.detach())

    def update(self, rng):
        if len(self.replay) < max(self.batch_size, self.learning_starts):
            return {}
        batch = self.replay.sample(self.batch_size, rng)
        states = [b['state'] for b in batch]
        next_states = [b['next_state'] for b in batch]
        action = torch.as_tensor(np.stack([b['action'] for b in batch]), device=self.device)
        score_action = action
        slate_mode = self.critic_action_representation == 'slate_mean_st'
        if slate_mode:
            ids = torch.as_tensor(np.stack([b['slate'] for b in batch]), device=self.device)
            action = self.features[ids].mean(1)
        reward = torch.tensor([b['reward'] for b in batch], device=self.device, dtype=torch.float32)[:, None]
        done = torch.tensor([b['done'] for b in batch], device=self.device, dtype=torch.float32)[:, None]
        with torch.no_grad():
            nz = self.target_projector(self.target_adapter(*self._batch(next_states)))
            noise = torch.as_tensor(rng.normal(0, 0.2, size=action.shape).clip(-0.5, 0.5), device=self.device, dtype=torch.float32)
            na = (self.target_actor(nz) + noise).clamp(-1, 1)
            if slate_mode:
                active = [i for i, b in enumerate(batch) if not b['done']]
                target = reward.clone()
                if active:
                    na = self.slate_action(na[active], [batch[i]['next_candidate_ids'] for i in active], target=True)
                    q1, q2 = self.target_critic(nz[active], na)
                    target[active] += self.gamma * torch.minimum(q1, q2)
            else:
                q1, q2 = self.target_critic(nz, na)
                target = reward + self.gamma * (1 - done) * torch.minimum(q1, q2)
        z = self.encode_states(states)
        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(list(self.adapter.parameters()) + list(self.projector.parameters()) + list(self.critic.parameters()), 10.0)
        self.encoder_optimizer.step()
        self.critic_optimizer.step()
        self.updates += 1
        logs = {'critic_loss': float(critic_loss.detach())}
        logs.update(self.auxiliary_update(batch))
        diagnostics = self.config.get('diagnostics', {}).get('enabled', False)
        if diagnostics:
            logs.update(q_target=float(target.mean()), q_prediction=float(q1.detach().mean()), td_abs=float((q1.detach() - target).abs().mean()), critic_gradient_norm=float(critic_grad), replay_action_saturation=float((score_action.abs() > 0.95).float().mean()))
        if self.updates % self.policy_delay == 0:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            z = self.encode_states(states)
            if not self.actor_updates_encoder:
                z = z.detach()
            actor_action = self.actor(z)
            critic_action = self.slate_action(actor_action, [b['candidate_ids'] for b in batch], straight_through=True) if slate_mode else actor_action
            actor_loss = -self.critic(z, critic_action)[0].mean()
            regularization = self.actor_regularization()
            if regularization is not None:
                actor_loss = actor_loss + regularization
                logs['item_regularization'] = float(regularization.detach())
            item_optimizer = getattr(self, 'item_optimizer', None)
            if item_optimizer is not None:
                item_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_parameters = list(self.actor.parameters())
            if self.actor_updates_encoder:
                actor_parameters = list(self.adapter.parameters()) + list(self.projector.parameters()) + actor_parameters
            if item_optimizer is not None:
                actor_parameters += list(self.item_embedding.parameters())
            actor_grad = torch.nn.utils.clip_grad_norm_(actor_parameters, 10.0)
            if self.actor_updates_encoder:
                self.encoder_optimizer.step()
            self.actor_optimizer.step()
            if item_optimizer is not None:
                item_optimizer.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)
            self._soft_targets()
            logs['actor_loss'] = float(actor_loss.detach())
            if diagnostics:
                logs.update(actor_gradient_norm=float(actor_grad), actor_saturation=float((actor_action.detach().abs() > 0.95).float().mean()))
        if diagnostics:
            for key, value in logs.items():
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite local update: ' + key)
                self._learning_sums[key] = self._learning_sums.get(key, 0.0) + value
                self._learning_counts[key] = self._learning_counts.get(key, 0) + 1
        return logs

    def _soft_targets(self):
        with torch.no_grad():
            for model, target in [(self.adapter, self.target_adapter), (self.projector, self.target_projector), (self.actor, self.target_actor), (self.critic, self.target_critic)]:
                for p, t in zip(model.parameters(), target.parameters()):
                    t.lerp_(p, self.tau)
            if hasattr(self, 'item_embedding') and self.config.get('method') not in ('gpfedrec', 'gp_uniform'):
                self.target_items.lerp_(self.item_embedding.weight, self.tau)

    def projector_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.projector.state_dict().items()}

    def load_projector(self, state):
        self.projector.load_state_dict(state)
        self.target_projector.load_state_dict(state)

    def policy_response(self, z, candidate_ids):
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        ids = torch.as_tensor(candidate_ids, device=self.device, dtype=torch.long)
        a = self.actor(z)
        features = self.scoring_features()[ids]
        logits = a @ features.T if ids.ndim == 1 else torch.einsum('pd,pid->pi', a, features)
        temperature = float(self.config.get('federation', {}).get('policy_temperature', 1.0))
        return torch.softmax(logits / temperature, dim=-1)

    def metric_summary(self, rng, options, fed_config):
        from .metric import estimate_local_metric
        cache = estimate_local_metric(self, rng, options, fed_config)
        self._metric_cache = cache
        return {'distances': cache['distances'].copy(), 'mass': cache['mass'].copy(), 'diagnostics': copy.deepcopy(cache['diagnostics'])}

    def fit_geometry(self, target, options, fed_config):
        cache = self._metric_cache
        if cache is None:
            raise RuntimeError('Metric preparation is required')
        d = torch.as_tensor(cache['distances'], device=self.device, dtype=torch.float32).detach()
        goal = torch.as_tensor(target, device=self.device, dtype=torch.float32).detach()
        output = {}
        optimizer = self.geometry_optimizer if self.geometry_optimizer is not None else self.encoder_optimizer
        for _ in range(int(fed_config.get('aux_steps', 20))):
            z = self.encode_states(cache['states'])
            metric = F.mse_loss(torch.cdist(z, z), d)
            align = (z - goal).square().sum(-1).mean()
            loss = float(fed_config.get('lambda_metric', 1.0)) * metric if options['metric'] else z.sum() * 0
            loss = loss + (float(fed_config.get('lambda_align', 1.0)) * align if options['alignment'] else z.sum() * 0)
            optimizer.zero_grad()
            loss.backward()
            if self.geometry_optimizer is not None:
                norm = torch.nn.utils.clip_grad_norm_(list(self.adapter.parameters()) + list(self.projector.parameters()), float(fed_config.get('geometry_grad_clip', 10.0)), error_if_nonfinite=True)
            optimizer.step()
            output = {'metric_loss': float(metric.detach()), 'alignment_loss': float(align.detach())}
            if self.geometry_optimizer is not None:
                output['gradient_norm_before_clip'] = float(norm)
        if self.config.get('diagnostics', {}).get('enabled', False):
            with torch.no_grad():
                z = self.encode_states(cache['states'])
                output.update(post_metric_loss=float(F.mse_loss(torch.cdist(z, z), d)), post_alignment_loss=float((z - goal).square().sum(-1).mean()), post_latent_mean_norm=float(z.norm(dim=-1).mean()))
        return output

    def anchor_occupancy(self, anchors, rng, fed_config):
        count = min(len(self.replay), int(fed_config.get('occupancy_sample_limit', 4096)))
        states = [b['state'] for b in self.replay.sample(count, rng)]
        if not states:
            raise ValueError('Empty local replay')
        total = np.zeros(len(anchors), dtype=np.float64)
        nearest = []
        norms = []
        diagnostic = self.config.get('diagnostics', {}).get('enabled', False)
        with torch.no_grad():
            a = torch.as_tensor(anchors, device=self.device, dtype=torch.float32)
            for start in range(0, count, 256):
                z = self.encode_states(states[start:start + 256])
                distance = torch.cdist(z, a)
                logits = -distance.square() / float(fed_config.get('sigma_anchor', 1.0)) ** 2
                total += torch.softmax(logits, dim=-1).sum(0).cpu().numpy()
                if diagnostic:
                    nearest.extend(distance.min(-1).values.cpu().tolist())
                    norms.extend(z.norm(dim=-1).cpu().tolist())
        if diagnostic:
            from .experiment_checks import quantiles
            self._occupancy_audit = dict(sample_count=count, nearest_anchor_distance=quantiles(nearest), latent_norm=quantiles(norms), anchor_norm=quantiles(np.linalg.norm(anchors, axis=1)), interpretation='Absolute proximity on the existing occupancy sample, before softmax normalization.')
        return total / total.sum()

    def distill(self, probes, candidate_ids, teacher, confidence, fed_config):
        guard = fed_config.get('transfer_guard')
        if guard is not None and self.config.get('method') != 'transfer_guard_off':
            from .replay_guard import guarded_proposal
            result = guarded_proposal(self, lambda: self._distill_proposal(probes, candidate_ids, teacher, confidence, fed_config), guard)
            with torch.no_grad():
                if len(probes):
                    target = torch.as_tensor(teacher, device=self.device, dtype=torch.float32)
                    p = self.policy_response(probes, candidate_ids)
                    result['transfer_guard']['retained_teacher_kl'] = float((target * (target.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean())
            return result
        return self._distill_proposal(probes, candidate_ids, teacher, confidence, fed_config)

    def _distill_proposal(self, probes, candidate_ids, teacher, confidence, fed_config):
        z = torch.as_tensor(probes, device=self.device, dtype=torch.float32).detach()
        t = torch.as_tensor(teacher, device=self.device, dtype=torch.float32).detach()
        c = torch.as_tensor(confidence, device=self.device, dtype=torch.float32).detach()
        audit = None
        if fed_config.get('distillation_audit', False):
            from .distillation_audit import DistillationAudit
            audit = DistillationAudit(self.actor, c)
        if len(z) == 0 or not torch.any(c > 0) or float(fed_config.get('lambda_distill', 1.0)) == 0 or (int(fed_config.get('distill_steps', 20)) == 0):
            result = {'distillation_loss': 0.0, 'distillation_update_count': 0}
            if audit is not None:
                result['audit'] = audit.finish()
            return result
        value = 0.0
        for _ in range(int(fed_config.get('distill_steps', 20))):
            p = self.policy_response(z, candidate_ids)
            kl = (t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1)
            loss = float(fed_config.get('lambda_distill', 1.0)) * (c * kl).mean()
            item_optimizer = getattr(self, 'item_optimizer', None)
            if item_optimizer is not None:
                item_optimizer.zero_grad()
                loss = loss + self.actor_regularization()
            self.actor_optimizer.zero_grad()
            loss.backward()
            if audit is not None:
                audit.observe_backward(kl, loss)
            self.actor_optimizer.step()
            value = float(loss.detach())
            if item_optimizer is not None:
                item_optimizer.step()
        result = {'distillation_loss': value, 'distillation_update_count': int(fed_config.get('distill_steps', 20))}
        if audit is not None:
            with torch.no_grad():
                after = self.policy_response(z, candidate_ids)
                kl_after = (t * (t.clamp_min(1e-12).log() - after.clamp_min(1e-12).log())).sum(-1)
                result['audit'] = audit.finish(kl_after, float(fed_config.get('lambda_distill', 1.0)) * (c * kl_after).mean())
        return result

    def coverage_mask(self, probes, config, round_index, client_id):
        from .coverage import local_coverage
        mask, self._coverage_report = local_coverage(self, probes, config, round_index, client_id)
        return mask

    def save_state(self, include_replay=True):
        names = ['adapter', 'projector', 'actor', 'critic', 'target_adapter', 'target_projector', 'target_actor', 'target_critic', 'encoder_optimizer', 'actor_optimizer', 'critic_optimizer']
        result = {name: copy.deepcopy(getattr(self, name).state_dict()) for name in names}
        if hasattr(self, 'item_embedding') and self.config.get('method') not in ('gpfedrec', 'gp_uniform'):
            result['learned_items'] = {'weight': copy.deepcopy(self.item_embedding.state_dict()), 'optimizer': copy.deepcopy(self.item_optimizer.state_dict()), 'target': self.target_items.detach().clone()}
        if self.geometry_optimizer is not None:
            result['geometry_optimizer'] = copy.deepcopy(self.geometry_optimizer.state_dict())
        result.update(updates=self.updates, history_shape=self.history_shape, context_dim=self.context_dim, item_features=self.item_features.copy(), config=copy.deepcopy(self.config))
        result['learning_statistics'] = (copy.deepcopy(self._learning_sums), copy.deepcopy(self._learning_counts))
        if include_replay:
            result['replay'] = self.replay.state_dict()
            result['estimator_state'] = copy.deepcopy(self._estimator_state)
        return result

    def load_state(self, state, load_replay=True):
        if hasattr(self, 'item_embedding') and self.config.get('method') not in ('gpfedrec', 'gp_uniform'):
            items = state['learned_items']
            self.item_embedding.load_state_dict(items['weight'])
            self.item_optimizer.load_state_dict(items['optimizer'])
            self.target_items = items['target'].to(self.device).clone()
        saved = state.get('config', {})
        if saved.get('agent', {}).get('critic_action_representation', 'score_vector') != self.critic_action_representation:
            raise ValueError('Checkpoint critic action representation differs')
        if saved.get('agent', {}).get('actor_updates_encoder', True) != self.actor_updates_encoder:
            raise ValueError('Checkpoint actor update convention differs')
        if (saved.get('federation', {}).get('geometry_optimizer', 'shared') == 'separate') != (self.geometry_optimizer is not None):
            raise ValueError('Checkpoint geometry_optimizer convention differs')
        if self.geometry_optimizer is not None:
            if 'geometry_optimizer' not in state:
                raise ValueError('Checkpoint missing geometry_optimizer')
            self.geometry_optimizer.load_state_dict(state['geometry_optimizer'])
        for name in ['adapter', 'projector', 'actor', 'critic', 'target_adapter', 'target_projector', 'target_actor', 'target_critic', 'encoder_optimizer', 'actor_optimizer', 'critic_optimizer']:
            if name in state:
                getattr(self, name).load_state_dict(state[name])
        self.updates = state.get('updates', 0)
        self._learning_sums, self._learning_counts = copy.deepcopy(state.get('learning_statistics', ({}, {})))
        if load_replay and 'replay' in state:
            self.replay.load_state_dict(state['replay'])
        self._estimator_state = copy.deepcopy(state.get('estimator_state'))
