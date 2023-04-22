from typing import Tuple, Union, List

import numpy as np

from une.memories.buffers.nstep import NStep
from une.memories.buffers.uniform import UniformBuffer
from une.memories.transitions import Transition


class EREBuffer(UniformBuffer):
    """
    Paper: https://arxiv.org/abs/1906.04009
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        gradient_steps: int = 4,
        eta: float = 0.996,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
        )
        self.gradient_steps = gradient_steps
        self.eta = eta

    @property
    def memory_type(self):
        return "ere"

    def sample_idxs(self, g_step: int, batch_size: int) -> Union[np.ndarray, List[int]]:
        if self.full:
            sorted_indices = (
                list(range(0, self.pos))[::-1]
                + list(range(self.pos, self.buffer_size))[::-1]
            )
        else:
            sorted_indices = list(range(0, self.pos))[::-1]
        ck = int(
            max(
                len(self) * self.eta ** ((g_step * 1000) / self.gradient_steps),
                5000,
            )
        )
        return np.random.choice(sorted_indices[:ck], size=batch_size, replace=False)

    def sample(
        self, g_step: int, batch_size: int, to_tensor: bool = False
    ) -> Transition:
        assert len(self) >= batch_size, "You must add more transitions"

        indices = self.sample_idxs(g_step=g_step, batch_size=batch_size)
        return self.sample_transitions(indices=indices, to_tensor=to_tensor)


class NStepEREBuffer(EREBuffer, NStep):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        gradient_steps: int = 4,
        eta: float = 0.997,
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            gradient_steps=gradient_steps,
            eta=eta,
            n_step=n_step,
            gamma=gamma
        )

    def add(self, transition: Transition):
        nstep_transition = super().get_nstep_transition(transition=transition)
        if nstep_transition:
            super().add(transition=nstep_transition)