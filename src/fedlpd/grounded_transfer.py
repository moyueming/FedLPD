from copy import deepcopy
import numpy as np
import torch
from .agents import LocalAgent
from .coverage import stream

class ReplayBridge(LocalAgent):

    def __init__(self, owner, round_index, client_id):
        self.__dict__ = owner.__dict__.copy()
        self.owner = owner
        for name in ('adapter', 'projector', 'target_adapter', 'target_projector'):
            setattr(self, name, deepcopy(getattr(owner, name)))
        params = list(self.adapter.parameters()) + list(self.projector.parameters())
        self.encoder_optimizer = torch.optim.Adam(params, lr=owner.config['agent']['lr'])
        self.geometry_optimizer = torch.optim.Adam(params, lr=owner.config['federation'].get('geometry_lr', owner.config['agent']['lr']))
        self._metric_cache = None
        cfg = owner.config['federation']['replay_bridge']
        rng = stream(owner.config['seed'], owner.config['environment'], round_index, client_id, 'replay_bridge')
        self.bank = [r['state'] for r in owner.replay.sample(min(len(owner.replay), cfg['sample_limit']), rng)]
        self.k = min(cfg['neighbors'], len(self.bank))
        if not self.k:
            raise ValueError('Replay bridge requires training history')
        self._bank_geometry = None
        self._bank_policy = None
        self.bridge_audit = {'sample_count': len(self.bank), 'neighbors': self.k, 'geometry_mutates_rl': False, 'query_space': 'aligned replay neighbors', 'raw_states_uploaded': False, 'candidate_semantics': 'public counterfactual item queries'}

    def neighbors(self, probes):
        with torch.no_grad():
            if self._bank_geometry is None:
                self._bank_geometry = torch.cat([self.encode_states(self.bank[i:i + 256]) for i in range(0, len(self.bank), 256)])
                self._bank_policy = torch.cat([self.owner.encode_states(self.bank[i:i + 256]) for i in range(0, len(self.bank), 256)])
            p = torch.as_tensor(probes, device=self.device, dtype=torch.float32)
            distances = torch.cdist(p, self._bank_geometry)
            chosen = torch.argsort(distances, stable=True, dim=-1)[:, :self.k]
            d = distances.gather(1, chosen)
            weights = 1 / d.clamp_min(1e-06)
            weights = weights / weights.sum(-1, keepdim=True)
            self.bridge_audit.update(mean_nearest_distance=float(d[:, 0].mean()), unique_neighbor_count=int(chosen.unique().numel()), mean_neighbor_distance=float(d.mean()))
            return (self._bank_policy[chosen], weights)

    def policy_response(self, probes, candidate_ids):
        z, w = self.neighbors(probes)
        ids = torch.as_tensor(candidate_ids, device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None].expand(len(z), -1)
        response = self.owner.policy_response(z.flatten(0, 1), ids.repeat_interleave(self.k, dim=0))
        return (response.reshape(len(z), self.k, -1) * w[:, :, None]).sum(1)

    def distill(self, probes, candidate_ids, teacher, confidence, fed_config):
        z, w = self.neighbors(probes)
        ids = torch.as_tensor(candidate_ids, device=self.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None].expand(len(z), -1)
        teacher = torch.as_tensor(teacher, device=self.device, dtype=torch.float32)
        conf = torch.as_tensor(confidence, device=self.device, dtype=torch.float32)
        result = self.owner._distill_proposal(z.flatten(0, 1), ids.repeat_interleave(self.k, dim=0), teacher.repeat_interleave(self.k, dim=0), (conf[:, None] * w * self.k).flatten(), fed_config)
        result['grounded_state_rows'] = len(z) * self.k
        return result
