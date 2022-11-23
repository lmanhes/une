from typing import Tuple, Union, List

from loguru import logger
import numpy as np
import torch

from une.memories.buffer.uniform import UniformBuffer, Transition


class EREBuffer(UniformBuffer):
    """
    Paper: https://arxiv.org/abs/1906.04009
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        device: str = "cpu",
        gradient_steps: int = 4,
        eta: float = 0.996,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size, observation_shape=observation_shape, device=device
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

    def sample(self, g_step: int, batch_size: int, to_tensor: bool = False) -> Transition:
        assert len(self) >= batch_size, "You must add more transitions"

        indices = self.sample_idxs(g_step=g_step, batch_size=batch_size)
        return self.sample_transitions(indices=indices, to_tensor=to_tensor)
