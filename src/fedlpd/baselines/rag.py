from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..agents import LocalAgent, mlp

def mico_distance(x, y, angle_weight):
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    cosine = F.cosine_similarity(x, y, dim=-1, eps=1e-06).clamp(-1 + 1e-06, 1 - 1e-06)
    angle = torch.acos(cosine)
    angle = torch.where((nx > 1e-06) & (ny > 1e-06), angle, torch.zeros_like(angle))
    return nx + ny + angle_weight * angle

def rag_residual(distance, next_distance, reward_difference_squared, variance_sum, gamma, continuing=None):
    if continuing is not None:
        next_distance = next_distance * continuing.to(next_distance.dtype)
    residual = (distance - gamma * next_distance).square() - (reward_difference_squared - variance_sum)
    return residual.square().mean()

class GaussianModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, bounds):
        super().__init__()
        self.net = mlp(input_dim, hidden_dim, output_dim * 2)
        self.minimum, self.maximum = map(math.log, bounds)
        with torch.no_grad():
            self.net[-1].bias[output_dim:].fill_(-2.3)

    def forward(self, x):
        mean, logvar = self.net(x).chunk(2, dim=-1)
        return (mean, logvar.clamp(self.minimum, self.maximum))

def gaussian_nll(mean, logvar, target):
    return 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar)).mean()

class RAGAgent(LocalAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag = self.config['baselines']['fedrag']
        h = self.config['agent']['hidden_dim']
        bounds = (self.rag['variance_min'], self.rag['variance_max'])
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.reward_model = GaussianModel(self.latent_dim, h, 1, bounds).to(self.device)
            self.dynamics_model = GaussianModel(self.latent_dim + self.action_dim, h, self.latent_dim, bounds).to(self.device)
        self.model_optimizer = torch.optim.Adam(list(self.reward_model.parameters()) + list(self.dynamics_model.parameters()), lr=self.rag['model_lr'])
        self.global_projector = {k: v.detach().clone() for k, v in self.projector.named_parameters()}
        self.baseline_statistics = {'updates': 0, 'sums': {}}

    def load_projector(self, state):
        super().load_projector(state)
        self.global_projector = {k: v.detach().clone() for k, v in self.projector.named_parameters()}

    def auxiliary_update(self, batch):
        states = [b['state'] for b in batch]
        next_states = [b['next_state'] for b in batch]
        reward = torch.tensor([b['reward'] for b in batch], device=self.device, dtype=torch.float32)[:, None]
        done = torch.tensor([b['done'] for b in batch], device=self.device, dtype=torch.bool)
        ids = torch.as_tensor(np.stack([b['slate'] for b in batch]), device=self.device)
        action = self.features[ids].mean(1)
        with torch.no_grad():
            z = self.encode_states(states)
            nz = self.target_projector(self.target_adapter(*self._batch(next_states)))
        mean, logvar = self.reward_model(z)
        reward_loss = gaussian_nll(mean, logvar, reward)
        next_mean, next_logvar = self.dynamics_model(torch.cat([z, action], -1))
        transition_loss = gaussian_nll(next_mean[~done], next_logvar[~done], nz[~done]) if (~done).any() else next_mean.sum() * 0
        model_loss = reward_loss + transition_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.reward_model.parameters()) + list(self.dynamics_model.parameters()), 10.0, error_if_nonfinite=True)
        self.model_optimizer.step()
        with torch.no_grad():
            _, rv = self.reward_model(z)
            mu, _ = self.dynamics_model(torch.cat([z, action], -1))
            paired = torch.roll(torch.arange(len(batch), device=self.device), 1)
            next_distance = mico_distance(mu, mu[paired], self.rag['angle_weight'])
            reward_delta = (reward[:, 0] - reward[paired, 0]).square()
            variance = rv[:, 0].exp() + rv[paired, 0].exp()
            continuing = ~(done | done[paired])
        online = self.encode_states(states)
        distance = mico_distance(online, online[paired], self.rag['angle_weight'])
        metric = rag_residual(distance, next_distance, reward_delta, variance, self.gamma, continuing)
        proximal = 0.5 * sum(((p - self.global_projector[k]).square().sum() for k, p in self.projector.named_parameters()))
        loss = self.rag['metric_weight'] * metric + self.rag['proximal_weight'] * proximal
        self.encoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.adapter.parameters()) + list(self.projector.parameters()), 10.0, error_if_nonfinite=True)
        self.encoder_optimizer.step()
        logs = dict(rag_loss=float(metric.detach()), rag_proximal=float(proximal.detach()), reward_nll=float(reward_loss.detach()), transition_nll=float(transition_loss.detach()), reward_variance=float(rv.exp().mean()))
        if not all((np.isfinite(v) for v in logs.values())):
            raise FloatingPointError('Nonfinite FedRAG update')
        self.baseline_statistics['updates'] += 1
        for k, v in logs.items():
            self.baseline_statistics['sums'][k] = self.baseline_statistics['sums'].get(k, 0.0) + v
        return logs

    def save_state(self, include_replay=True):
        result = super().save_state(include_replay)
        result['baseline'] = {k: deepcopy(getattr(self, k).state_dict()) for k in ('reward_model', 'dynamics_model', 'model_optimizer')}
        result['baseline'].update(global_projector=deepcopy(self.global_projector), statistics=deepcopy(self.baseline_statistics))
        return result

    def load_state(self, state, load_replay=True):
        if state.get('config', {}).get('baselines') != self.config.get('baselines'):
            raise ValueError('Baseline checkpoint settings differ')
        super().load_state(state, load_replay)
        saved = state['baseline']
        for k in ('reward_model', 'dynamics_model', 'model_optimizer'):
            getattr(self, k).load_state_dict(saved[k])
        self.global_projector = {k: v.to(self.device).clone() for k, v in saved['global_projector'].items()}
        self.baseline_statistics = deepcopy(saved['statistics'])
