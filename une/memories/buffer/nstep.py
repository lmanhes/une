from collections import deque
from typing import Tuple

import numpy as np

from une.memories.utils.transition import Transition


class NStep(object):
    """
    Paper: http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
    """

    def __init__(
        self,
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ) -> None:
        self.n_step = n_step
        self.gamma = gamma

        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step reward, next_observation, and done."""
        # info of the last transition
        reward, next_observation, done = (
            self.n_step_buffer[-1].reward,
            self.n_step_buffer[-1].next_observation,
            self.n_step_buffer[-1].done,
        )

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition.reward, transition.next_observation, transition.done

            reward = r + self.gamma * reward * (1 - d)
            next_observation, done = (n_o, d) if d else (next_observation, done)

        return reward, next_observation, done

    def get_nstep_transition(self, transition: Transition):
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        # make a n-step transition
        reward, next_observation, done = self._get_n_step_info()
        observation, action = (
            self.n_step_buffer[0].observation,
            self.n_step_buffer[0].action,
        )

        return Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )
    