from typing import Tuple, Union, Type

import numpy as np
import torch
import wandb

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.utils.transition import (
    TransitionEpisodic,
    TransitionNStepEpisodic,
    TransitionNStepPEREpisodic,
)
from une.representations.abstract import AbstractRepresentation
from une.algos.noisy_dqn import NoisyDQN
from une.curiosity.lifelong_episodic import LifeLongEpisodicModule


class NGUDQN(NoisyDQN):
    def __init__(
        self,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        features_dim: int,
        n_actions: int,
        representation_module_cls: Type[AbstractRepresentation],
        memory_buffer_cls: Type[AbstractBuffer],
        gamma: float = 0.99,
        batch_size: int = 32,
        gradient_steps: int = 1,
        buffer_size: int = int(1e6),
        n_step: int = 1,
        learning_rate: float = 2.5e-4,
        max_grad_norm: float = 10,
        target_update_interval_steps: int = 1e4,
        soft_update: bool = False,
        tau: float = 1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_decay_eps_max_steps: int = 1e3,
        use_gpu: bool = False,
        per_alpha: float = 0.7,
        per_beta: float = 0.4,
        intrinsic_reward_weight: float = 0.1,
        icm_features_dim: int = 256,
        icm_forward_loss_weight: float = 0.2,
        ecm_memory_size: int = 300,
        ecm_k: int = 10,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            features_dim=features_dim,
            n_actions=n_actions,
            representation_module_cls=representation_module_cls,
            memory_buffer_cls=memory_buffer_cls,
            gamma=gamma,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            buffer_size=buffer_size,
            n_step=n_step,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            target_update_interval_steps=target_update_interval_steps,
            soft_update=soft_update,
            tau=tau,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_decay_eps_max_steps=exploration_decay_eps_max_steps,
            use_gpu=use_gpu,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.icm_features_dim = icm_features_dim
        self.icm_forward_loss_weight = icm_forward_loss_weight
        self.ecm_memory_size = ecm_memory_size
        self.ecm_k = ecm_k

        self.lifelong_ecm = LifeLongEpisodicModule(
            encoder=self.q_net.representation_module,
            input_dim=self.features_dim,
            features_dim=self.icm_features_dim,
            actions_dim=n_actions,
            episodic_memory_size=int(self.ecm_memory_size),
            k=self.ecm_k,
            forward_loss_weight=self.icm_forward_loss_weight,
        ).to(self.device)

    @property
    def networks(self):
        return [self.q_net, self.lifelong_ecm]

    def parameters(self):
        return list(
            set(list(self.q_net.parameters()) + list(self.lifelong_ecm.parameters()))
        )

    def act(self, observation: np.ndarray, steps: int) -> int:
        action = self.choose_action(observation=observation, steps=steps)
        self._last_observation = observation
        self._last_action = action
        return action

    def memorize(self, observation: np.ndarray, reward: float, done: bool) -> None:
        assert self._last_observation is not None and self._last_action is not None

        if not isinstance(observation, (torch.Tensor, np.ndarray)):
            observation = np.array(observation)
        observation = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
        episodic_reward = self.lifelong_ecm.get_episodic_reward(observation=observation)

        transition = TransitionEpisodic(
            observation=self._last_observation,
            action=self._last_action,
            reward=reward,
            episodic_reward=episodic_reward,
            done=done,
            next_observation=observation,
        )
        self.memory_buffer.add(transition)

        if done:
            self.lifelong_ecm.reset()

    def compute_loss(
        self,
        samples_from_memory: Union[
            TransitionEpisodic,
            TransitionNStepEpisodic,
            TransitionNStepPEREpisodic,
        ],
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:

        intrinsic_loss, intrinsic_reward = self.lifelong_ecm(
            observation=samples_from_memory.observation,
            next_observation=samples_from_memory.next_observation,
            action=samples_from_memory.action,
            episodic_reward=samples_from_memory.episodic_reward,
        )

        wandb.log(
            {
                "intrinsic_reward": intrinsic_reward.mean(),
                "extrinsic_reward": samples_from_memory.reward.mean(),
            },
            step=steps,
        )

        current_q_values = self.compute_q_values(
            samples_from_memory=samples_from_memory
        )
        target_q_values = self.compute_target_q_values(
            samples_from_memory=samples_from_memory,
            intrinsic_reward=intrinsic_reward,
            intrinsic_reward_weight=self.intrinsic_reward_weight,
        )

        # Compute Huber loss (less sensitive to outliers)
        if elementwise:
            dqn_loss = self.criterion(
                current_q_values.squeeze(1),
                target_q_values.squeeze(1),
                reduction="none",
            )
        else:
            dqn_loss = self.criterion(
                current_q_values.squeeze(1), target_q_values.squeeze(1)
            )

        return dqn_loss + intrinsic_loss
