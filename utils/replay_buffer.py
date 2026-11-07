"""Single-environment replay buffer."""
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self._dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, state, next_state, action, reward, done, infos):
        idx = self._pointer
        self._states[idx] = state
        self._next_states[idx] = next_state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._dones[idx] = done
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices]
        )

