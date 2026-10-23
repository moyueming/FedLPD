import numpy as np
from fedlpd.types import StepResult
from fedlpd.data import PreparedData, FrozenResponse
from .base import SessionEnvironment, environment_source_hash

class KuaiSessionEnvironment(SessionEnvironment):
    backend_name = 'kuaisim-based-frozen-response-v1'

    def __init__(self, client_id, seed, config, data_dir, response_path, split, device='cpu', evaluation=False):
        if data_dir is None:
            raise ValueError('KuaiSim requires prepared data_dir; synthetic fallback is forbidden')
        if response_path is None:
            raise ValueError('KuaiSim requires fitted response_path')
        self._response = FrozenResponse(data_dir, response_path, device)
        self._data = self._response.data
        if not config.get('smoke', False) and (not self._data.metadata.get('publication_eligible', False)):
            raise ValueError('Smoke prepared data cannot be used in formal experiments')
        self._initial_rows = self._data.initial_rows(client_id, split)
        if len(self._initial_rows) == 0:
            raise ValueError(f'No {split} session initializations for client {client_id}')
        if config.get('task_sampling_audit') or config.get('training_candidate_bias'):
            from fedlpd.task_sampling import training_feedback_groups
            if not hasattr(self._data, '_task_feedback_groups'):
                self._data._task_feedback_groups = training_feedback_groups(self._data.items, self._data.splits, self._data.labels, len(self._data.catalog_features))
            groups, info = self._data._task_feedback_groups
            config = dict(config, _sampling_group_ids=groups, _sampling_group_info=info)
        super().__init__(client_id, seed, config, self._data.catalog_features, evaluation)
        self._temper = float(config.get('initial_temper', 20))
        self._user_id = 0
        self.provenance = {'backend': self.backend_name, 'upstream_commit': '2ae32aa25a0aac103194a66e6864e3d2ac6d6580', 'response_sha256': self._response.checkpoint_sha256, 'split': split, 'publication_eligible': not bool(config.get('smoke', False)) and self._data.metadata.get('publication_eligible', False)}
        self.provenance['environment_source_sha256'] = environment_source_hash(__file__)

    def reset(self, session_seed=None):
        self._new_rng(session_seed)
        idx = int(self.rng.choice(self._initial_rows))
        self._user_id = int(self._data.users[idx])
        self.history = self._data.history(idx, 20)
        self.exposed = set()
        self.request = 0
        self.done = False
        self._temper = float(self.config.get('initial_temper', 20))
        self.candidates = self._sample_candidates()
        return self._observation()

    def step(self, slate_ids):
        if self.done:
            raise RuntimeError('reset() required before stepping terminated session')
        ids = self._validate_slate(slate_ids)
        probabilities = self._response.probabilities(self._user_id, ids, self.history)
        feedback = (self.rng.random(probabilities.shape) < probabilities).astype(np.float32)
        reward = float(np.mean(0.8 * feedback[:, 1] + 0.2 * feedback[:, 2]))
        self.history.extend(((int(i), f.copy()) for i, f in zip(ids, feedback)))
        self.history = self.history[-20:]
        self.exposed.update(ids.tolist())
        self.request += 1
        self._temper += float(np.clip(reward - 2, -2, 0))
        leaving = self._temper < 1
        self.done = bool(leaving or self.request >= self.horizon)
        self.candidates = self._sample_candidates()
        return StepResult(self._observation(), reward, self.done, {'request': self.request, 'termination_reason': 'feedback_leave' if leaving else 'horizon' if self.done else None})

    def _private_state(self):
        return {'temper': self._temper, 'user_id': self._user_id}

    def _load_private_state(self, state):
        self._temper = state['temper']
        self._user_id = state['user_id']
