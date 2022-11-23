from collections import namedtuple
from typing import Tuple, List, Union

from loguru import logger
import numpy as np
import torch

from une.memories.buffer.uniform import UniformBuffer, Transition
from une.memories.utils.segment_tree import SumSegmentTree, MinSegmentTree


TransitionPER = namedtuple(
    "Transition",
    field_names=[
        "observation",
        "action",
        "reward",
        "done",
        "next_observation",
        "indices",
        "weights",
    ],
)


class PERBuffer(UniformBuffer):
    """
    Paper: https://arxiv.org/pdf/1511.05952.pdf
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        device: str = "cpu",
        alpha: float = 0.7,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size, observation_shape=observation_shape, device=device
        )
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.prior_eps = prior_eps

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    @property
    def memory_type(self):
        return "per"

    def add(self, transition: Transition):
        super().add(transition=transition)

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return self._sample_proportional(batch_size=batch_size)

    def sample(
        self, batch_size: int, to_tensor: bool = False, **kwargs
    ) -> TransitionPER:
        assert len(self) >= batch_size, "You must add more transitions"

        indices = self.sample_idxs(batch_size=batch_size)
        weights = np.array([self._calculate_weight(i, self.beta) for i in indices])

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
            weights = torch.from_numpy(weights).float().to(self.device)

        return TransitionPER(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            done=dones,
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight