from collections import deque
from typing import Tuple, List, Union

import numpy as np
import torch

from une.memories.buffers.uniform import UniformBuffer
from une.memories.transitions import Transition, NStepTransition


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


class NStepUniformBuffer(UniformBuffer, NStep):
    """
    Paper: http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            n_step=n_step,
            gamma=gamma,
        )

    def add(self, transition: Transition):
        nstep_transition = super().get_nstep_transition(transition=transition)
        if nstep_transition:
            super().add(transition=nstep_transition)

    def sample_transitions(
        self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False
    ) -> NStepTransition:
        transition = super().sample_transitions(indices=indices, to_tensor=to_tensor)

        next_observations = self.observations[(np.array(indices)+1) % self.buffer_size]
        next_nstep_observations = self.next_observations[indices]
        if to_tensor:
            next_observations = (
                torch.from_numpy(next_observations).float().to(self.device)
            )
            next_nstep_observations = (
                torch.from_numpy(next_nstep_observations).float().to(self.device)
            )

        return NStepTransition(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            next_observation=next_observations,
            next_nstep_observation=next_nstep_observations,
            done=transition.done,
        )