from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..agents import LocalAgent

def graph_aggregate(tables, threshold_scale, layers):
    if tables.ndim != 3 or len(tables) < 2 or (not torch.isfinite(tables).all()):
        raise ValueError('Invalid client item tables')
    if not np.isfinite(threshold_scale) or threshold_scale <= 0 or type(layers) is not int or (layers < 1):
        raise ValueError('Invalid graph settings')
    flat = tables.flatten(1)
    unit = F.normalize(flat, p=2, dim=1, eps=1e-12)
    similarity = (unit @ unit.T).clamp(-1, 1)
    adjacency = (similarity > similarity.mean() * threshold_scale).to(tables.dtype)
    empty = adjacency.sum(1) == 0
    adjacency[empty] = torch.eye(len(tables), device=tables.device, dtype=tables.dtype)[empty]
    weights = adjacency / adjacency.sum(1, keepdim=True)
    personal = flat
    for _ in range(layers):
        personal = weights @ personal
    personal = personal.reshape_as(tables)
    return (personal.mean(0), personal, weights, similarity)

class GraphAgent(LocalAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = self.config['baselines']['gpfedrec']
        self.item_embedding = nn.Embedding.from_pretrained(self.features.detach().clone(), freeze=False)
        self.target_items = self.features.detach().clone()
        self.personalized_items = self.features.detach().clone()
        self.item_optimizer = torch.optim.Adam(self.item_embedding.parameters(), lr=self.config['agent']['lr'])
        self.baseline_statistics = {'updates': 0, 'sums': {}}

    def scoring_features(self, target=False):
        return self.target_items if target else self.item_embedding.weight

    def act(self, obs, rng, explore=False):
        with torch.no_grad():
            action = self.actor(self.encode_states([obs]))[0].cpu().numpy()
        if explore:
            action = np.clip(action + rng.normal(0, self.exploration_noise, size=action.shape), -1, 1).astype(np.float32)
        ids = np.asarray(obs.candidate_ids, dtype=np.int64)
        if len(ids) < self.slate_size:
            raise ValueError('Candidate set smaller than slate')
        with torch.no_grad():
            features = self.item_embedding.weight[torch.as_tensor(ids, device=self.device)].cpu().numpy()
        scores = features @ action
        ranking = np.argsort(-scores, kind='stable')[:self.slate_size]
        return (ids[ranking].copy(), action.copy())

    def actor_regularization(self):
        return self.graph['regularization'] * F.mse_loss(self.item_embedding.weight, self.personalized_items)

    def _soft_targets(self):
        super()._soft_targets()
        with torch.no_grad():
            self.target_items.lerp_(self.item_embedding.weight, self.tau)

    def update(self, rng):
        logs = super().update(rng)
        if logs:
            self.baseline_statistics['updates'] += 1
            for k, v in logs.items():
                self.baseline_statistics['sums'][k] = self.baseline_statistics['sums'].get(k, 0.0) + v
        return logs

    def load_item_round(self, global_items, personalized_items):
        if global_items.shape != self.features.shape or personalized_items.shape != self.features.shape:
            raise ValueError('Federated item table shape differs')
        if not torch.isfinite(global_items).all() or not torch.isfinite(personalized_items).all():
            raise ValueError('Nonfinite federated item table')
        with torch.no_grad():
            self.item_embedding.weight.copy_(global_items.to(self.device))
            self.target_items.copy_(global_items.to(self.device))
            self.personalized_items.copy_(personalized_items.to(self.device))
        self.item_optimizer.state.clear()

    def save_state(self, include_replay=True):
        result = super().save_state(include_replay)
        result['baseline'] = {k: deepcopy(getattr(self, k).state_dict()) for k in ('item_embedding', 'item_optimizer')}
        result['baseline'].update(target_items=self.target_items.detach().clone(), personalized_items=self.personalized_items.detach().clone(), statistics=deepcopy(self.baseline_statistics))
        return result

    def load_state(self, state, load_replay=True):
        if state.get('config', {}).get('baselines') != self.config.get('baselines'):
            raise ValueError('Baseline checkpoint settings differ')
        super().load_state(state, load_replay)
        saved = state['baseline']
        for k in ('item_embedding', 'item_optimizer'):
            getattr(self, k).load_state_dict(saved[k])
        self.target_items = saved['target_items'].to(self.device).clone()
        self.personalized_items = saved['personalized_items'].to(self.device).clone()
        self.baseline_statistics = deepcopy(saved['statistics'])
