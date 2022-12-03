from collections import namedtuple
import psutil
from typing import Tuple, List, Union

from loguru import logger
import numpy as np
import torch

from une.memories.buffer.abstract import AbstractBuffer


Transition = namedtuple(
    "Transition",
    field_names=["observation", "action", "reward", "done", "next_observation"],
)


class UniformBuffer(AbstractBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.device = device

        self.observations = np.zeros(
            (self.buffer_size,) + self.observation_shape, observation_dtype
        )
        self.actions = np.zeros((self.buffer_size, 1))
        self.rewards = np.zeros((self.buffer_size,))
        self.next_observations = np.zeros(
            (self.buffer_size,) + self.observation_shape, observation_dtype
        )
        self.dones = np.zeros((self.buffer_size,))

        self.pos = 0
        self.full = False

        self.check_memory_usage()

    @property
    def memory_type(self):
        return "uniform"

    def check_memory_usage(self):
        mem_available = psutil.virtual_memory().available
        total_memory_usage = (
            self.observations.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
            + self.next_observations.nbytes
        )
 
        logger.info(
            f"Total memory usage : {total_memory_usage / 1e9} / {mem_available / 1e9} GB"
        )

        if total_memory_usage > mem_available:
            # Convert to GB
            total_memory_usage /= 1e9
            mem_available /= 1e9
            logger.warning(
                "This system does not have apparently enough memory to store the complete "
                f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
            )

    def add(self, transition: Transition):
        self.observations[self.pos] = np.array(transition.observation).copy()
        self.actions[self.pos] = np.array(transition.action).copy()
        self.rewards[self.pos] = np.array(transition.reward).copy()
        self.next_observations[self.pos] = np.array(transition.next_observation).copy()
        self.dones[self.pos] = np.array(transition.done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return np.random.choice(len(self), size=batch_size, replace=False)

    def sample_transitions(
        self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False
    ) -> Transition:
        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices].reshape(-1, 1)
        next_observations = self.next_observations[indices]
        dones = self.dones[indices].reshape(-1, 1)

        if to_tensor:
            observations = torch.from_numpy(observations).float().to(self.device)
            actions = torch.from_numpy(actions).to(torch.int8).to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            next_observations = (
                torch.from_numpy(next_observations).float().to(self.device)
            )
            dones = torch.from_numpy(dones).to(torch.int8).to(self.device)

        return Transition(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            done=dones,
        )

    def sample(self, batch_size: int, to_tensor: bool = False, **kwargs) -> Transition:
        assert len(self) >= batch_size, "You must add more transitions"

        indices = self.sample_idxs(batch_size=batch_size)
        return self.sample_transitions(indices=indices, to_tensor=to_tensor)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.pos
