from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
from fedlpd.types import Observation, StepResult

def environment_source_hash(*paths):
    digest = hashlib.sha256()
    for path in [Path(__file__), *(Path(p) for p in paths)]:
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()

class SessionEnvironment:

    def __init__(self, client_id, seed, config, catalog_features, evaluation=False):
        self.client_id = int(client_id)
        self.config = dict(config)
        self.evaluation = evaluation
        self.catalog_features = np.asarray(catalog_features, dtype=np.float32)
        self.catalog_features.setflags(write=False)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), self.client_id, 65242]))
        self.candidate_size = int(config.get('candidate_size', config.get('candidates', 100)))
        self.slate_size = int(config.get('slate_size', 5))
        self.horizon = int(config.get('horizon', 20))
        if len(self.catalog_features) < self.candidate_size + (self.horizon - 1) * self.slate_size:
            raise ValueError('catalog_size must cover candidate_size + (horizon-1)*slate_size without repeat exposures')
        self.history_length = 10 if self.client_id % 5 in [0, 3] else 0 if self.client_id % 5 == 2 else 20
        self.schema_labels = [0, 1] if self.client_id % 5 in [0, 3] else [2, 3]
        self.request = 0
        self.history = []
        self.exposed = set()
        self.done = True
        self.candidates = np.empty(0, dtype=np.int64)
        self.candidate_group_ids = None
        self._candidate_group_counts = np.zeros(5, dtype=np.int64)
        self._exposure_group_counts = np.zeros(5, dtype=np.int64)
        self._candidate_draws = 0
        if config.get('task_sampling_audit') or config.get('training_candidate_bias'):
            from fedlpd.task_sampling import content_groups
            supplied = config.get('_sampling_group_ids')
            self.candidate_group_ids = np.asarray(supplied, dtype=np.int64) if supplied is not None else content_groups(self.catalog_features, config.get('_sampling_environment', 'recsim'), config.get('topics', 6))

    def _new_rng(self, session_seed):
        if session_seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([int(session_seed), self.client_id, 659918]))

    def _sample_candidates(self):
        available = np.setdiff1d(np.arange(len(self.catalog_features)), np.fromiter(self.exposed, dtype=np.int64), assume_unique=False)
        if len(available) < self.candidate_size:
            if self.done:
                return available
            raise RuntimeError('Insufficient legal catalog items')
        bias = self.config.get('training_candidate_bias')
        if bias and self.config.get('_split', 'train') == 'train':
            from fedlpd.task_sampling import candidate_probabilities
            p = candidate_probabilities(self.candidate_group_ids, available, self.client_id, bias['strength'])
            selected = self.rng.choice(available, self.candidate_size, replace=False, p=p)
        else:
            selected = self.rng.choice(available, self.candidate_size, replace=False)
        if self.candidate_group_ids is not None:
            self._candidate_group_counts += np.bincount(self.candidate_group_ids[selected], minlength=5)
            self._candidate_draws += 1
        return selected

    def record_exposure(self, slate):
        if self.candidate_group_ids is not None:
            self._exposure_group_counts += np.bincount(self.candidate_group_ids[slate], minlength=5)

    def sampling_audit(self):
        from fedlpd.task_sampling import group_digest
        groups = self.candidate_group_ids
        return dict(client_id=self.client_id, split=self.config.get('_split', 'train'), biased_training=bool(self.config.get('training_candidate_bias')) and self.config.get('_split', 'train') == 'train', favored_group=self.client_id % 5, group_sha256=group_digest(groups) if groups is not None else None, grouping=self.config.get('_sampling_group_info', {'protocol': 'public-topic-modulo-five-v1'}), catalog_group_counts=np.bincount(groups, minlength=5).tolist() if groups is not None else None, candidate_draws=self._candidate_draws, candidate_items=int(self._candidate_group_counts.sum()), candidate_group_counts=self._candidate_group_counts.tolist(), exposure_group_counts=self._exposure_group_counts.tolist())

    def _visible_feedback(self, labels):
        return np.asarray([labels[0], labels[1]] if self.client_id % 5 in [0, 3] else [labels[2], 1 - labels[1]], dtype=np.float32)

    def _observation(self):
        f = self.catalog_features.shape[1]
        time = np.asarray([self.request / self.horizon, (self.horizon - self.request) / self.horizon], dtype=np.float32)
        if self.history_length:
            h = np.zeros((self.history_length, f + 3), dtype=np.float32)
            for j, (iid, labels) in enumerate(self.history[-self.history_length:]):
                if iid >= 0:
                    h[j, :f] = self.catalog_features[iid]
                h[j, f:f + 2] = self._visible_feedback(labels)
                h[j, -1] = 1
            context = time
        else:
            h = np.empty((0, f + 3), dtype=np.float32)
            summary = np.zeros(f + 2, dtype=np.float32)
            recent = self.history[-20:]
            for iid, labels in recent:
                if iid >= 0:
                    summary[:f] += self.catalog_features[iid]
                summary[f:] += labels[:2]
            summary /= max(len(recent), 1)
            context = np.concatenate((summary, time))
        return Observation(h.copy(), context.copy(), self.candidates.copy())

    def _validate_slate(self, slate_ids, ignore_candidates=False):
        raw = np.asarray(slate_ids)
        if raw.dtype.kind not in 'iu' and (raw.dtype.kind != 'f' or not np.isfinite(raw).all() or (not np.equal(raw, np.floor(raw)).all())):
            raise ValueError('slate IDs must be finite integer values')
        ids = np.asarray(raw, dtype=np.int64)
        if ids.shape != (self.slate_size,) or len(set(ids.tolist())) != self.slate_size:
            raise ValueError('slate must contain exactly slate_size distinct item IDs')
        if np.any(ids < 0) or np.any(ids >= len(self.catalog_features)):
            raise ValueError('Invalid item ID')
        legal = set(range(len(self.catalog_features))) - self.exposed if ignore_candidates else set(self.candidates.tolist())
        if not set(ids.tolist()) <= legal:
            raise ValueError('Slate contains unavailable items')
        return ids

    def state_dict(self):
        state = {'rng': self.rng.bit_generator.state, 'request': self.request, 'history': self.history, 'exposed': self.exposed, 'done': self.done, 'candidates': self.candidates, 'private': self._private_state()}
        if self.candidate_group_ids is not None:
            state['task_sampling'] = (self._candidate_group_counts, self._exposure_group_counts, self._candidate_draws)
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        s = copy.deepcopy(state)
        self.rng.bit_generator.state = s['rng']
        self.request = s['request']
        self.history = s['history']
        self.exposed = s['exposed']
        self.done = s['done']
        self.candidates = s['candidates']
        self._load_private_state(s['private'])
        if 'task_sampling' in s:
            self._candidate_group_counts, self._exposure_group_counts, self._candidate_draws = s['task_sampling']

    def _private_state(self):
        return {}

    def _load_private_state(self, state):
        pass
