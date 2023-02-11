from typing import Tuple, List, Union

import numpy as np
import torch

from une.memories.buffer.nstep import NStep
from une.memories.buffer.per import PERBuffer, NStepPERBuffer
from une.memories.buffer.uniform import UniformBuffer
from une.memories.utils.transition import TransitionEpisodic, TransitionNStepEpisodic, TransitionPEREpisodic, TransitionNStepPEREpisodic


class EpisodicBuffer(UniformBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device
        )
        self.episodic_rewards = np.zeros((self.buffer_size,))

    def add(self, transition: TransitionEpisodic):
        self.episodic_rewards[self.pos] = np.array(transition.episodic_reward).copy()
        super().add(transition=transition)

    def sample_transitions(
        self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False
    ) -> TransitionEpisodic:
        transition = super().sample_transitions(indices=indices, to_tensor=to_tensor)

        episodic_rewards = self.episodic_rewards[indices].reshape(-1, 1)
        if to_tensor:
            episodic_rewards = torch.from_numpy(episodic_rewards).float().to(self.device)

        return TransitionEpisodic(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            episodic_reward=episodic_rewards,
            next_observation=transition.next_observation,
            done=transition.done,
        )


class EpisodicNStepUniformBuffer(EpisodicBuffer, NStep):

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

    def add(self, transition: TransitionEpisodic):
        nstep_transition = super().get_nstep_transition(transition=transition)
        if nstep_transition:
            super().add(transition=nstep_transition)

    def sample_transitions(
        self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False
    ) -> TransitionNStepEpisodic:
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

        return TransitionNStepEpisodic(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            episodic_reward=transition.episodic_reward,
            next_observation=transition.next_observations,
            next_nstep_observation=next_nstep_observations,
            done=transition.done,
        )


class EpisodicPERBuffer(PERBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        alpha: float = 0.7,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            prior_eps=prior_eps,
        )
        self.episodic_rewards = np.zeros((self.buffer_size,))

    def add(self, transition: TransitionEpisodic):
        self.episodic_rewards[self.pos] = np.array(transition.episodic_reward).copy()
        super().add(transition=transition)

    def sample_transitions(self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False) -> TransitionPEREpisodic:
        transition = super().sample_transitions(indices, to_tensor)

        episodic_rewards = self.episodic_rewards[indices].reshape(-1, 1)
        if to_tensor:
            episodic_rewards = torch.from_numpy(episodic_rewards).float().to(self.device)

        return TransitionPEREpisodic(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            episodic_reward=episodic_rewards,
            next_observation=transition.next_observation,
            done=transition.done,
            indices=transition.indices,
            weights=transition.weights,
        )


class EpisodicNStepPERBuffer(NStepPERBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        device: str = "cpu",
        alpha: float = 0.7,
        beta: float = 0.4,
        prior_eps: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
        **kwargs
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=device,
            n_step=n_step,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            prior_eps=prior_eps,
        )
        self.episodic_rewards = np.zeros((self.buffer_size,))

    def add(self, transition: TransitionEpisodic):
        self.episodic_rewards[self.pos] = np.array(transition.episodic_reward).copy()
        super().add(transition=transition)

    def sample_transitions(self, indices: Union[np.ndarray, List[int]], to_tensor: bool = False) -> TransitionNStepPEREpisodic:
        transition = super().sample_transitions(indices, to_tensor)

        episodic_rewards = self.episodic_rewards[indices].reshape(-1, 1)
        if to_tensor:
            episodic_rewards = torch.from_numpy(episodic_rewards).float().to(self.device)

        return TransitionNStepPEREpisodic(
            observation=transition.observation,
            action=transition.action,
            reward=transition.reward,
            episodic_reward=episodic_rewards,
            next_observation=transition.next_observation,
            next_nstep_observation=transition.next_nstep_observation,
            done=transition.done,
            indices=transition.indices,
            weights=transition.weights,
        )
