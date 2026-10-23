from __future__ import annotations
import copy
import numpy as np
from fedlpd.types import StepResult
from .base import SessionEnvironment, environment_source_hash

def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

class ControlledRecSim(SessionEnvironment):

    def __init__(self, client_id, seed, config, evaluation=False):
        self.topics = int(config.get('topics', 6))
        feature_dim = int(config.get('feature_dim', 16))
        if feature_dim < self.topics + 2:
            raise ValueError('feature_dim must be >= topics+2')
        crng = np.random.default_rng(int(config.get('catalog_seed', 1729)))
        n = int(config.get('catalog_size', 2000))
        categories = crng.integers(self.topics, size=n)
        features = np.zeros((n, feature_dim), dtype=np.float32)
        features[np.arange(n), categories] = 1
        features[:, self.topics:self.topics + 2] = crng.uniform(0.05, 0.95, size=(n, 2))
        if feature_dim > self.topics + 2:
            features[:, self.topics + 2:] = crng.normal(0, 0.15, size=(n, feature_dim - self.topics - 2))
        super().__init__(client_id, seed, config, features, evaluation)
        self._interest = np.full(self.topics, 1 / self.topics)
        self._fatigue = np.zeros(self.topics)
        self._satisfaction = 0.7
        self._last_reward = 0.0
        self._backend = None
        smoke = bool(config.get('smoke', False))
        self.provenance = {'backend': 'controlled-smoke-kernel' if smoke else 'recsim.SingleUserEnvironment', 'upstream_commit': '55e50e4be736d222ffe8c2477ed1981b40f91605', 'task_version': 'fedlpd-controlled-v1', 'publication_eligible': not smoke}
        self.provenance['environment_source_sha256'] = environment_source_hash(__file__)
        if not smoke:
            try:
                from .recsim_backend import load_backend
                SingleUserEnvironment, backend_info = load_backend()
                self.provenance.update(backend_info)
            except ImportError as exc:
                raise ImportError('Formal RecSim requires installed recsim framework. Install server requirements; --smoke is only nonpublication testing.') from exc
            self._backend = SingleUserEnvironment(_User(self), _Sampler(self), self.candidate_size, self.slate_size, resample_documents=True)

    def _reset_kernel(self):
        self.request = 0
        self.history = []
        self.exposed = set()
        self.done = False
        self._interest = self.rng.dirichlet(np.ones(self.topics) * 2)
        self._fatigue = self.rng.uniform(0, 0.05, size=self.topics)
        self._satisfaction = float(self.rng.uniform(0.6, 0.9))
        self._last_reward = 0.0

    def reset(self, session_seed=None):
        self._new_rng(session_seed)
        if self._backend is None:
            self._reset_kernel()
            self.candidates = self._sample_candidates()
        else:
            self._backend._num_candidates = self.candidate_size
            self._backend._document_sampler.queue = []
            _, docs = self._backend.reset()
            self.candidates = np.array([int(x) for x in docs], dtype=np.int64)
        return self._observation()

    def _simulate(self, ids):
        f = self.catalog_features[ids]
        cat = f[:, :self.topics].argmax(1)
        quality = f[:, self.topics]
        bait = f[:, self.topics + 1]
        affinity = self._interest[cat] * self.topics
        pc = _sigmoid(1.5 * affinity + 1.5 * bait - 0.8 * self._fatigue[cat] + self._satisfaction - 1.8)
        pl = _sigmoid(1.2 * affinity + 1.3 * quality + 0.6 * self._satisfaction - 1.5 * self._fatigue[cat] - 0.8 * bait - 1.5)
        pi = _sigmoid(quality + affinity - bait - 2)
        click = (self.rng.random(len(ids)) < pc).astype(np.float32)
        long = click * (self.rng.random(len(ids)) < pl)
        like = long * (self.rng.random(len(ids)) < pi)
        return np.stack((click, long, like), axis=1).astype(np.float32)

    def _advance(self, ids, feedback):
        f = self.catalog_features[ids]
        frequencies = f[:, :self.topics].mean(0)
        reward = float(np.mean(0.8 * feedback[:, 1] + 0.2 * feedback[:, 2]))
        self._last_reward = reward
        self._fatigue = np.clip(0.9 * self._fatigue + 0.2 * frequencies, 0, 1)
        self._interest = 0.97 * self._interest + 0.03 * frequencies + self.rng.normal(0, 0.002, self.topics)
        self._interest = np.maximum(self._interest, 0.0001)
        self._interest /= self._interest.sum()
        self._satisfaction = float(np.clip(0.88 * self._satisfaction + 0.12 * f[:, self.topics].mean() - 0.08 * f[:, self.topics + 1].mean() + 0.08 * reward - 0.01 + self.rng.normal(0, 0.015), 0, 1))
        self.history.extend(((int(i), x.copy()) for i, x in zip(ids, feedback)))
        self.history = self.history[-20:]
        self.exposed.update(ids.tolist())
        self.request += 1
        leaving = self.rng.random() < 0.015 + 0.15 * (1 - self._satisfaction) ** 2
        self.done = bool(leaving or self.request >= self.horizon)
        self._reason = 'feedback_leave' if leaving else 'horizon' if self.done else None

    def step(self, slate_ids):
        if self.done:
            raise RuntimeError('reset() required before stepping terminated session')
        ids = self._validate_slate(slate_ids)
        if self._backend is None:
            self._advance(ids, self._simulate(ids))
            self.candidates = self._sample_candidates()
        else:
            positions = [int(np.flatnonzero(self.candidates == x)[0]) for x in ids]
            _, docs, _, done = self._backend.step(positions)
            self.candidates = np.array([int(x) for x in docs], dtype=np.int64)
            if bool(done) != self.done:
                raise RuntimeError('RecSim termination dispatch mismatch')
        return StepResult(self._observation(), self._last_reward, self.done, {'request': self.request, 'termination_reason': self._reason})

    def _private_state(self):
        return {'interest': self._interest.copy(), 'fatigue': self._fatigue.copy(), 'satisfaction': self._satisfaction, 'last_reward': self._last_reward, 'reason': getattr(self, '_reason', None)}

    def _load_private_state(self, s):
        self._interest = s['interest'].copy()
        self._fatigue = s['fatigue'].copy()
        self._satisfaction = s['satisfaction']
        self._last_reward = s['last_reward']
        self._reason = s['reason']

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self._backend is not None:
            from recsim.document import CandidateSet
            from collections import OrderedDict
            candidate_set = CandidateSet()
            for iid in self.candidates:
                candidate_set.add_document(_Document(int(iid), self.catalog_features[iid]))
            self._backend._candidate_set = candidate_set
            self._backend._num_candidates = min(self.candidate_size, len(self.catalog_features) - len(self.exposed)) if self.done else self.candidate_size
            self._backend._current_documents = OrderedDict(candidate_set.create_observation())
            self._backend._document_sampler.queue = []

    def diagnostic_state(self):
        if not self.evaluation:
            raise PermissionError('Simulator truth is restricted to offline evaluation')
        return self.state_dict()

    def legal_items(self, snapshot):
        if not self.evaluation:
            raise PermissionError('Simulator truth is restricted to offline evaluation')
        return np.setdiff1d(np.arange(len(self.catalog_features)), list(snapshot['exposed']))

    def diagnostic_rollout(self, snapshot, slate_ids, seeds):
        if not self.evaluation:
            raise PermissionError('Simulator truth is restricted to offline evaluation')
        saved = self.state_dict()
        rewards = []
        states = []
        try:
            for seed in seeds:
                self.load_state_dict(snapshot)
                self.rng = np.random.default_rng(int(seed))
                if self.done:
                    raise ValueError('Diagnostic states must be nonterminal')
                ids = self._validate_slate(slate_ids, ignore_candidates=True)
                self._advance(ids, self._simulate(ids))
                rewards.append(self._last_reward)
                states.append(np.concatenate((np.zeros(self.topics * 2 + 2), [1.0])) if self.done else np.concatenate((self._interest, self._fatigue, [self._satisfaction, self.request / self.horizon, 0.0])))
        finally:
            self.load_state_dict(saved)
        return {'reward_samples': np.asarray(rewards, dtype=np.float64), 'next_state_samples': np.asarray(states, dtype=np.float64)}

class _Document:

    def __init__(self, iid, features):
        self._id = iid
        self.features = features

    def doc_id(self):
        return self._id

    def create_observation(self):
        return self.features.copy()

class _Response:

    def __init__(self, feedback):
        self.feedback = feedback

    def create_observation(self):
        return self.feedback.copy()

class _Sampler:

    def __init__(self, owner):
        self.owner = owner
        self.queue = []

    def sample_document(self):
        if not self.queue:
            self.queue = self.owner._sample_candidates().tolist()
        iid = self.queue.pop(0)
        return _Document(iid, self.owner.catalog_features[iid])

    def update_state(self, documents, responses):
        if self.owner.done:
            self.owner._backend._num_candidates = min(self.owner.candidate_size, len(self.owner.catalog_features) - len(self.owner.exposed))

    def reset_sampler(self):
        self.queue = []

class _User:

    def __init__(self, owner):
        self.owner = owner

    def reset(self):
        self.owner._reset_kernel()

    def reset_sampler(self):
        pass

    def create_observation(self):
        return np.empty(0, dtype=np.float32)

    def simulate_response(self, documents):
        return [_Response(x) for x in self.owner._simulate(np.array([d.doc_id() for d in documents]))]

    def update_state(self, documents, responses):
        self.owner._advance(np.array([d.doc_id() for d in documents]), np.array([r.feedback for r in responses]))

    def is_terminal(self):
        return self.owner.done
