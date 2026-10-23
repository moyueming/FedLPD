from dataclasses import dataclass, field
import numpy as np

@dataclass
class Observation:
    history: np.ndarray
    context: np.ndarray
    candidate_ids: np.ndarray

    def __post_init__(self):
        self.history = np.asarray(self.history, dtype=np.float32).copy()
        self.context = np.asarray(self.context, dtype=np.float32).copy()
        self.candidate_ids = np.asarray(self.candidate_ids, dtype=np.int64).copy()
        if self.history.ndim != 2 or self.context.ndim != 1 or self.candidate_ids.ndim != 1:
            raise ValueError('Observation shapes must be [history,features], [context], [candidates]')
        if not np.isfinite(self.history).all() or not np.isfinite(self.context).all():
            raise ValueError('Nonfinite visible observation')
        if len(np.unique(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('Candidate IDs must be unique')

    def state(self):
        return {'history': self.history.copy(), 'context': self.context.copy()}

@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

def validate_slate(slate, candidates, size):
    raw = np.asarray(slate)
    ids = np.asarray(slate, dtype=np.int64)
    if raw.ndim != 1 or ids.size != size or (not np.array_equal(raw, ids)):
        raise ValueError(f'Expected {size} integer item IDs')
    if len(np.unique(ids)) != size or not np.isin(ids, candidates).all():
        raise ValueError('Slate must contain distinct legal candidates')
    return ids
