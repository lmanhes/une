from collections import deque
from dataclasses import fields
import psutil
from typing import Tuple, List, Union

from loguru import logger
import numpy as np
import torch

from une.memories.buffers.abstract import AbstractBuffer
from une.memories.utils.segment_tree import SumSegmentTree, MinSegmentTree
from une.memories.transitions import (
    RecurrentTransition,
    NStepRecurrentTransition,
    NStepPERRecurrentTransition,
)


class SequenceTracker(object):
    def __init__(self, sequence_size: int, over_lapping: int, buffer_size: int) -> None:
        self.sequence_size = sequence_size
        self.over_lapping = over_lapping

        self.n_sequences = buffer_size // (sequence_size - self.over_lapping)
        logger.info(f"Sequence length : {self.sequence_size}")
        logger.info(f"Max number of sequences : {self.n_sequences}")

        self.sequences = deque(maxlen=self.n_sequences)
        self.buffer = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def add(self, pos: int, done: bool):
        self.buffer.append(pos)

        if done:
            self.sequences.append(self.buffer)
            self.buffer = []

        elif len(self.buffer) == self.sequence_size:
            self.sequences.append(self.buffer)
            if self.over_lapping > 0:
                self.buffer = self.buffer[-self.over_lapping :]
            else:
                self.buffer = []

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return np.random.choice(len(self), size=batch_size, replace=False)

    def sample(self, batch_size: int) -> List[List[int]]:
        assert batch_size <= len(self), "You must add more transitions"

        indices = self.sample_idxs(batch_size=batch_size)
        sequences_idxs = [self.sequences[i] for i in indices]
        return sequences_idxs


class RecurrentUniformBuffer(AbstractBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        sequence_length: int = 80,
        burn_in: int = 40,
        over_lapping: int = 20,
        recurrent_dim: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()

        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.sequence_size = self.burn_in + self.sequence_length
        self.over_lapping = over_lapping
        self.recurrent_dim = recurrent_dim
        self.observation_shape = observation_shape
        self.device = device

        self.observations = np.zeros(
            (self.buffer_size,) + self.observation_shape,
            observation_dtype,
        )
        self.h_recurrents = np.zeros((self.buffer_size, recurrent_dim))
        self.c_recurrents = np.zeros((self.buffer_size, recurrent_dim))

        self.actions = np.zeros((self.buffer_size, 1))

        self.rewards = np.zeros((self.buffer_size,))
        self.dones = np.zeros((self.buffer_size))

        self.next_observations = np.zeros(
            (self.buffer_size,) + self.observation_shape,
            observation_dtype,
        )
        self.next_h_recurrents = np.zeros((self.buffer_size, recurrent_dim))
        self.next_c_recurrents = np.zeros((self.buffer_size, recurrent_dim))

        self.pos = 0
        self.full = False

        self.sequence_tracker = SequenceTracker(
            sequence_size=self.sequence_size,
            over_lapping=self.over_lapping,
            buffer_size=self.buffer_size,
        )

        self.check_memory_usage()

    @property
    def memory_type(self):
        return "sequence-uniform"

    def check_memory_usage(self):
        mem_available = psutil.virtual_memory().available
        total_memory_usage = (
            self.observations.nbytes
            + self.h_recurrents.nbytes
            + self.c_recurrents.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.dones.nbytes
            + self.next_observations.nbytes
            + self.next_h_recurrents.nbytes
            + self.next_c_recurrents.nbytes
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

    def add(self, transition: RecurrentTransition):
        self.sequence_tracker.add(self.pos, transition.done)

        self.observations[self.pos] = np.array(transition.observation).copy()
        self.h_recurrents[self.pos] = np.array(transition.h_recurrent).copy()
        self.c_recurrents[self.pos] = np.array(transition.c_recurrent).copy()
        self.actions[self.pos] = np.array(transition.action).copy()
        self.rewards[self.pos] = np.array(transition.reward).copy()
        self.next_observations[self.pos] = np.array(transition.next_observation).copy()
        self.next_h_recurrents[self.pos] = np.array(transition.next_h_recurrent).copy()
        self.next_c_recurrents[self.pos] = np.array(transition.next_c_recurrent).copy()
        self.dones[self.pos] = np.array(transition.done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def split_burnin(self, samples: RecurrentTransition) -> Tuple[RecurrentTransition]:
        burnin_samples, learning_samples = {}, {}
        for field in fields(samples):
            if field.name not in ["length"]:
                burnin_samples[field.name] = getattr(samples, field.name)[
                    :, : self.burn_in
                ]
                learning_samples[field.name] = getattr(samples, field.name)[
                    :, self.burn_in :
                ]
            else:
                burnin_samples[field.name] = getattr(samples, field.name)
                learning_samples[field.name] = getattr(samples, field.name)
        return RecurrentTransition(**burnin_samples), RecurrentTransition(
            **learning_samples
        )

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return self.sequence_tracker.sample(batch_size=batch_size)

    def sample_transitions(
        self, indices: List[List[int]], to_tensor: bool = False
    ) -> RecurrentTransition:
        batch_size = len(indices)

        observations = np.zeros(
            shape=(batch_size, self.sequence_size, *self.observation_shape)
        )
        h_recurrents = np.zeros(
            shape=(batch_size, self.sequence_size, self.recurrent_dim)
        )
        c_recurrents = np.zeros(
            shape=(batch_size, self.sequence_size, self.recurrent_dim)
        )

        actions = np.zeros(shape=(batch_size, self.sequence_size, 1))

        rewards = np.zeros(shape=(batch_size, self.sequence_size, 1))
        dones = np.zeros(shape=(batch_size, self.sequence_size, 1))

        next_observations = np.zeros(
            shape=(batch_size, self.sequence_size, *self.observation_shape)
        )
        next_h_recurrents = np.zeros(
            shape=(batch_size, self.sequence_size, self.recurrent_dim)
        )
        next_c_recurrents = np.zeros(
            shape=(batch_size, self.sequence_size, self.recurrent_dim)
        )

        lengths = np.ones(shape=(batch_size,)) * self.sequence_size
        masks = np.zeros((batch_size, self.sequence_size))

        for i, sequence_idxs in enumerate(indices):
            sequence_idxs = np.array(sequence_idxs)
            observations[i, : len(sequence_idxs)] = self.observations[sequence_idxs]
            h_recurrents[i, : len(sequence_idxs)] = self.h_recurrents[sequence_idxs]
            c_recurrents[i, : len(sequence_idxs)] = self.c_recurrents[sequence_idxs]

            actions[i, : len(sequence_idxs)] = self.actions[sequence_idxs]

            rewards[i, : len(sequence_idxs)] = self.rewards[sequence_idxs].reshape(
                -1, 1
            )
            dones[i, : len(sequence_idxs)] = self.dones[sequence_idxs].reshape(-1, 1)

            next_observations[i, : len(sequence_idxs)] = self.next_observations[
                sequence_idxs
            ]
            masks[i, : len(sequence_idxs)] = np.ones_like((sequence_idxs,))
            next_h_recurrents[i, : len(sequence_idxs)] = self.next_h_recurrents[
                sequence_idxs
            ]
            next_c_recurrents[i, : len(sequence_idxs)] = self.next_c_recurrents[
                sequence_idxs
            ]

            lengths[i] = len(sequence_idxs)

        if to_tensor:
            observations = torch.from_numpy(observations).float().to(self.device)
            h_recurrents = torch.from_numpy(h_recurrents).float().to(self.device)
            c_recurrents = torch.from_numpy(c_recurrents).float().to(self.device)

            actions = torch.from_numpy(actions).to(torch.int8).to(self.device)

            rewards = torch.from_numpy(rewards).float().to(self.device)

            next_observations = (
                torch.from_numpy(next_observations).float().to(self.device)
            )
            next_h_recurrents = (
                torch.from_numpy(next_h_recurrents).float().to(self.device)
            )
            next_c_recurrents = (
                torch.from_numpy(next_c_recurrents).float().to(self.device)
            )
            dones = torch.from_numpy(dones).to(torch.int8).to(self.device)
            masks = torch.from_numpy(masks).to(torch.bool)

        return RecurrentTransition(
            observation=observations,
            h_recurrent=h_recurrents,
            c_recurrent=c_recurrents,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            next_h_recurrent=next_h_recurrents,
            next_c_recurrent=next_c_recurrents,
            done=dones,
            mask=masks,
            length=lengths,
        )

    def sample(
        self, batch_size: int, to_tensor: bool = False, **kwargs
    ) -> RecurrentTransition:
        indices = self.sample_idxs(batch_size=batch_size)
        # print("indices : ", indices)
        return self.sample_transitions(indices=indices, to_tensor=to_tensor)

    @property
    def n_sequences(self):
        return len(self.sequence_tracker)

    def __len__(self) -> int:
        if self.full:
            return self.buffer_size
        else:
            return self.pos


class RecurrentNStepUniformBuffer(RecurrentUniformBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        sequence_length: int = 80,
        burn_in: int = 40,
        over_lapping: int = 20,
        recurrent_dim: int = 512,
        n_step: int = 3,
        gamma: float = 0.997,
        **kwargs,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            sequence_length=sequence_length,
            burn_in=burn_in,
            over_lapping=over_lapping,
            recurrent_dim=recurrent_dim,
            **kwargs,
        )
        self.n_step = n_step
        self.gamma = gamma

        self.n_step_buffer = deque(maxlen=self.n_step)

    def add(self, transition: RecurrentTransition):
        nstep_transition = self.get_nstep_transition(transition=transition)
        if nstep_transition:
            super().add(transition=nstep_transition)

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        (
            reward,
            next_observation,
            next_h_recurrent,
            next_c_recurrent,
            done,
        ) = (
            self.n_step_buffer[-1].reward,
            self.n_step_buffer[-1].next_observation,
            self.n_step_buffer[-1].next_h_recurrent,
            self.n_step_buffer[-1].next_c_recurrent,
            self.n_step_buffer[-1].done,
        )

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, n_h, n_c, d = (
                transition.reward,
                transition.next_observation,
                transition.next_h_recurrent,
                transition.next_c_recurrent,
                transition.done,
            )

            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_observation = n_o
                next_h_recurrent = n_h
                next_c_recurrent = n_c
                done = d

        return (
            reward,
            next_observation,
            next_h_recurrent,
            next_c_recurrent,
            done,
        )

    def get_nstep_transition(self, transition: RecurrentTransition):
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        # make a n-step transition
        (
            reward,
            next_observation,
            next_h_recurrent,
            next_c_recurrent,
            done,
        ) = self._get_n_step_info()
        observation, action, h_recurrent, c_recurrent = (
            self.n_step_buffer[0].observation,
            self.n_step_buffer[0].action,
            self.n_step_buffer[0].h_recurrent,
            self.n_step_buffer[0].c_recurrent,
        )

        return RecurrentTransition(
            h_recurrent=h_recurrent,
            c_recurrent=c_recurrent,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_h_recurrent=next_h_recurrent,
            next_c_recurrent=next_c_recurrent,
            done=done,
        )

    def sample_transitions(
        self, indices: List[List[int]], to_tensor: bool = False
    ) -> NStepRecurrentTransition:
        transition = super().sample_transitions(indices=indices, to_tensor=to_tensor)

        batch_size = len(indices)
        next_observations = np.zeros(
            shape=(batch_size, self.sequence_size, *self.observation_shape)
        )
        next_nstep_observations = np.zeros(
            shape=(batch_size, self.sequence_size, *self.observation_shape)
        )
        for i, sequence_idxs in enumerate(indices):
            next_observations[i, : len(sequence_idxs)] = self.observations[
                (np.array(sequence_idxs) + 1) % self.buffer_size
            ]
            next_nstep_observations[i, : len(sequence_idxs)] = self.next_observations[
                sequence_idxs
            ]

        if to_tensor:
            next_observations = (
                torch.from_numpy(next_observations).float().to(self.device)
            )
            next_nstep_observations = (
                torch.from_numpy(next_nstep_observations).float().to(self.device)
            )

        return NStepRecurrentTransition(
            observation=transition.observation,
            h_recurrent=transition.h_recurrent,
            c_recurrent=transition.c_recurrent,
            action=transition.action,
            reward=transition.reward,
            next_observation=next_observations,
            next_nstep_observation=next_nstep_observations,
            next_h_recurrent=transition.next_h_recurrent,
            next_c_recurrent=transition.next_c_recurrent,
            done=transition.done,
            mask=transition.mask,
            length=transition.length,
        )


class RecurrentNStepPERBuffer(RecurrentNStepUniformBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        sequence_length: int = 80,
        burn_in: int = 40,
        over_lapping: int = 20,
        recurrent_dim: int = 512,
        n_step: int = 3,
        gamma: float = 0.997,
        alpha: float = 0.7,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            sequence_length=sequence_length,
            burn_in=burn_in,
            over_lapping=over_lapping,
            recurrent_dim=recurrent_dim,
            n_step=n_step,
            gamma=gamma,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.prior_eps = prior_eps

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.sequence_tracker.n_sequences:
            tree_capacity *= 2
        logger.info(f"Tree capacity : {tree_capacity}")

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def split_burnin(
        self, samples: NStepPERRecurrentTransition
    ) -> Tuple[NStepPERRecurrentTransition]:
        burnin_samples, learning_samples = {}, {}
        for field in fields(samples):
            if field.name not in ["weights", "indices", "length"]:
                burnin_samples[field.name] = getattr(samples, field.name)[
                    :, : self.burn_in
                ]
                learning_samples[field.name] = getattr(samples, field.name)[
                    :, self.burn_in :
                ]
            else:
                burnin_samples[field.name] = getattr(samples, field.name)
                learning_samples[field.name] = getattr(samples, field.name)
        return NStepPERRecurrentTransition(
            **burnin_samples
        ), NStepPERRecurrentTransition(**learning_samples)

    def add(self, transition: NStepPERRecurrentTransition):
        super().add(transition=transition)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.sequence_tracker.n_sequences

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return self._sample_proportional(batch_size=batch_size)

    def sample_transitions(
        self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False
    ) -> NStepPERRecurrentTransition:
        sequences_idxs = [self.sequence_tracker[idx] for idx in indices]
        transition = super().sample_transitions(sequences_idxs, to_tensor)

        weights = np.array([self._calculate_weight(i, self.beta) for i in indices])
        if to_tensor:
            weights = torch.from_numpy(weights).float().to(self.device)

        return NStepPERRecurrentTransition(
            observation=transition.observation,
            h_recurrent=transition.h_recurrent,
            c_recurrent=transition.c_recurrent,
            action=transition.action,
            reward=transition.reward,
            next_observation=transition.next_observation,
            next_nstep_observation=transition.next_nstep_observation,
            next_h_recurrent=transition.next_h_recurrent,
            next_c_recurrent=transition.next_c_recurrent,
            done=transition.done,
            mask=transition.mask,
            length=transition.length,
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.sequence_tracker)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.sequence_tracker) - 1)
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
        max_weight = (p_min * len(self.sequence_tracker)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self.sequence_tracker)) ** (-beta)
        weight = weight / max_weight

        return weight
