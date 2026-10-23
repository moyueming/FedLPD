from __future__ import annotations
import copy
import numpy as np

def state_of(observation):
    if hasattr(observation, 'state'):
        return observation.state()
    if isinstance(observation, dict):
        return {'history': np.asarray(observation['history'], dtype=np.float32).copy(), 'context': np.asarray(observation['context'], dtype=np.float32).copy()}
    return {'history': np.asarray(observation[0], dtype=np.float32).copy(), 'context': np.asarray(observation[1], dtype=np.float32).copy()}

class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        self.data = []
        self.position = 0

    def __len__(self):
        return len(self.data)

    def add(self, observation, action, slate, reward, next_observation, done, episode_id, step_in_episode=None):
        record = {'state': state_of(observation), 'action': np.asarray(action, dtype=np.float32).copy(), 'slate': np.asarray(slate, dtype=np.int64).copy(), 'reward': float(reward), 'candidate_ids': np.asarray(observation.candidate_ids, dtype=np.int64).copy(), 'next_candidate_ids': np.asarray(getattr(next_observation, 'candidate_ids', []), dtype=np.int64).copy(), 'next_state': state_of(next_observation), 'done': bool(done), 'episode_id': int(episode_id), 'step_in_episode': None if step_in_episode is None else int(step_in_episode)}
        if len(self.data) < self.capacity:
            self.data.append(record)
        else:
            self.data[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)
        return [self.data[int(i)] for i in idx]

    def state_dict(self):
        return {'capacity': self.capacity, 'position': self.position, 'data': copy.deepcopy(self.data)}

    def load_state_dict(self, state):
        self.capacity = state['capacity']
        self.position = state['position']
        self.data = copy.deepcopy(state['data'])
