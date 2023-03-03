from collections import deque
import psutil
from typing import Tuple, List, Union

from loguru import logger
import numpy as np
import torch

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.utils.transition import (
    TransitionRecurrentIn,
    TransitionRecurrentOut,
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

    def add(self, pos: int, done: bool):
        self.buffer.append(pos)

        if (len(self.buffer) == self.sequence_size) or done:
            self.sequences.append(self.buffer)
            self.buffer = self.buffer[-self.over_lapping:]

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return np.random.choice(len(self), size=batch_size, replace=False)

    def sample(self, batch_size: int) -> List[List[int]]:
        assert batch_size <= len(self), "You must add more transitions"

        indices = self.sample_idxs(batch_size=batch_size)
        #print(indices, [self.sequences[i] for i in indices])
        return [self.sequences[i] for i in indices]


class SequenceUniformBuffer(AbstractBuffer):
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
        #self.next_last_actions = np.zeros((self.buffer_size, 1))

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
            #+ self.next_last_actions.nbytes
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

    def add(self, transition: TransitionRecurrentIn):
        self.sequence_tracker.add(self.pos, transition.done)

        self.observations[self.pos] = np.array(transition.observation).copy()
        self.h_recurrents[self.pos] = np.array(transition.h_recurrent).copy()
        self.c_recurrents[self.pos] = np.array(transition.c_recurrent).copy()
        self.actions[self.pos] = np.array(transition.action).copy()
        self.rewards[self.pos] = np.array(transition.reward).copy()
        self.next_observations[self.pos] = np.array(transition.next_observation).copy()
        self.next_h_recurrents[self.pos] = np.array(transition.next_h_recurrent).copy()
        self.next_c_recurrents[self.pos] = np.array(transition.next_c_recurrent).copy()
        #self.next_last_actions[self.pos] = np.array(transition.next_last_action).copy()
        self.dones[self.pos] = np.array(transition.done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def split_burnin(
        self, samples: TransitionRecurrentOut
    ) -> Tuple[TransitionRecurrentOut]:
        burnin_samples, learning_samples = {}, {}
        for name, value in samples._asdict().items():
            if name != "lengths":
                burnin_samples[name] = value[:, : self.burn_in]
                learning_samples[name] = value[:, self.burn_in :]
            else:
                burnin_samples[name] = value
                learning_samples[name] = value
        return TransitionRecurrentOut(**burnin_samples), TransitionRecurrentOut(
            **learning_samples
        )

    def sample_idxs(self, batch_size: int) -> Union[np.ndarray, List[int]]:
        return self.sequence_tracker.sample(batch_size=batch_size)

    def sample_transitions(
        self, indices: List[List[int]], to_tensor: bool = False
    ) -> TransitionRecurrentOut:
        batch_size = len(indices)

        last_actions = np.zeros(shape=(batch_size, self.sequence_size, 1))
        last_rewards = np.zeros(shape=(batch_size, self.sequence_size, 1))
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
        #next_last_actions = np.zeros(shape=(batch_size, self.sequence_size, 1))
        # next_last_rewards = np.zeros(shape=(batch_size, self.sequence_size, 1))

        lengths = np.ones(shape=(batch_size,)) * self.sequence_size
        for i, sequence_idxs in enumerate(indices):
            observations[i, : len(sequence_idxs)] = self.observations[sequence_idxs]
            h_recurrents[i, : len(sequence_idxs)] = self.h_recurrents[sequence_idxs]
            c_recurrents[i, : len(sequence_idxs)] = self.c_recurrents[sequence_idxs]
            # last_actions[i, : len(sequence_idxs)] = self.actions[
            #     (np.array(sequence_idxs) - 1) % len(self)
            # ]
            # last_rewards[i, : len(sequence_idxs)] = self.rewards[
            #     (np.array(sequence_idxs) - 1) % len(self)
            # ].reshape(-1, 1)

            actions[i, : len(sequence_idxs)] = self.actions[sequence_idxs]

            rewards[i, : len(sequence_idxs)] = self.rewards[sequence_idxs].reshape(
                -1, 1
            )
            dones[i, : len(sequence_idxs)] = self.dones[sequence_idxs].reshape(-1, 1)

            next_observations[i, : len(sequence_idxs)] = self.next_observations[
                sequence_idxs
            ]
            next_h_recurrents[i, : len(sequence_idxs)] = self.next_h_recurrents[
                sequence_idxs
            ]
            next_c_recurrents[i, : len(sequence_idxs)] = self.next_c_recurrents[
                sequence_idxs
            ]
            # next_last_actions[i, : len(sequence_idxs)] = self.next_last_actions[
            #     sequence_idxs
            # ]
            # next_last_rewards[i, : len(sequence_idxs)] = self.next_last_rewards[
            #     sequence_idxs
            # ].reshape(-1, 1)

            lengths[i] = len(sequence_idxs)

        if to_tensor:
            observations = torch.from_numpy(observations).float().to(self.device)
            h_recurrents = torch.from_numpy(h_recurrents).float().to(self.device)
            c_recurrents = torch.from_numpy(c_recurrents).float().to(self.device)
            # last_actions = torch.from_numpy(last_actions).to(torch.int8).to(self.device)
            # last_rewards = torch.from_numpy(last_rewards).float().to(self.device)

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
            # next_last_actions = (
            #     torch.from_numpy(next_last_actions).to(torch.int8).to(self.device)
            # )
            # next_last_rewards = (
            #     torch.from_numpy(next_last_rewards).float().to(self.device)
            # )
            dones = torch.from_numpy(dones).to(torch.int8).to(self.device)

        return TransitionRecurrentOut(
            observation=observations,
            h_recurrent=h_recurrents,
            c_recurrent=c_recurrents,
            #last_action=last_actions,
            #last_reward=last_rewards,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            next_h_recurrent=next_h_recurrents,
            next_c_recurrent=next_c_recurrents,
            #next_last_action=next_last_actions,
            #next_last_reward=next_last_rewards,
            done=dones,
            lengths=lengths,
        )

    def sample(
        self, batch_size: int, to_tensor: bool = False, **kwargs
    ) -> TransitionRecurrentOut:
        indices = self.sample_idxs(batch_size=batch_size)
        return self.sample_transitions(indices=indices, to_tensor=to_tensor)

    def __len__(self) -> int:
        if self.full:
            return self.buffer_size
        else:
            return self.pos


class NStepSequenceUniformBuffer(SequenceUniformBuffer):
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
        gamma: float = 0.99,
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

    def add(self, transition: TransitionRecurrentIn):
        nstep_transition = self.get_nstep_transition(transition=transition)
        if nstep_transition:
            super().add(transition=nstep_transition)

    def _get_n_step_info(self) -> Tuple[np.int64, np.ndarray, bool]:
        (
            reward,
            next_observation,
            next_h_recurrent,
            next_c_recurrent,
            #next_last_action,
            done,
        ) = (
            self.n_step_buffer[-1].reward,
            self.n_step_buffer[-1].next_observation,
            self.n_step_buffer[-1].next_h_recurrent,
            self.n_step_buffer[-1].next_c_recurrent,
            #self.n_step_buffer[-1].next_last_action,
            self.n_step_buffer[-1].done,
        )

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, n_h, n_c, d = (
                transition.reward,
                transition.next_observation,
                transition.next_h_recurrent,
                transition.next_c_recurrent,
                #transition.next_last_action,
                transition.done,
            )

            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_observation = n_o
                next_h_recurrent = n_h
                next_c_recurrent = n_c
                #next_last_action = n_a
                done = d

        return (
            reward,
            next_observation,
            next_h_recurrent,
            next_c_recurrent,
            #next_last_action,
            done,
        )

    def get_nstep_transition(self, transition: TransitionRecurrentIn):
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
            #next_last_action,
            done,
        ) = self._get_n_step_info()
        observation, action = (
            self.n_step_buffer[0].observation,
            self.n_step_buffer[0].action,
        )

        return TransitionRecurrentIn(
            h_recurrent=transition.h_recurrent,
            c_recurrent=transition.c_recurrent,
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            next_h_recurrent=next_h_recurrent,
            next_c_recurrent=next_c_recurrent,
            #next_last_action=next_last_action,
            done=done,
        )
