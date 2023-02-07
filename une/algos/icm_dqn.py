from typing import Tuple, Union, Type

import numpy as np
import torch
import wandb

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.buffer.uniform import Transition, TransitionNStep
from une.memories.buffer.per import TransitionPER, TransitionNStepPER
from une.representations.abstract import AbstractRepresentation
from une.algos.noisy_dqn import NoisyDQN
from une.curiosity.icm import IntrinsicCuriosityModule


class ICMDQN(NoisyDQN):
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
        icm_features_dim: int = 256,
        icm_alpha: float = 0.1,
        icm_beta: float = 0.2,
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

        self.icm_features_dim = icm_features_dim
        self.icm_alpha = icm_alpha
        self.icm_beta = icm_beta

        self.icm = IntrinsicCuriosityModule(
            encoder=self.q_net.representation_module,
            input_dim=self.features_dim,
            features_dim=self.icm_features_dim,
            actions_dim=n_actions,
        ).to(self.device)

    @property
    def networks(self):
        return [self.q_net, self.icm]

    def parameters(self):
        return list(set(list(self.q_net.parameters()) + list(self.icm.parameters())))

    def compute_loss(
        self,
        samples_from_memory: Union[
            Transition, TransitionNStep, TransitionPER, TransitionNStepPER
        ],
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:
        # Get current Q-values estimates
        # (batch_size, n_actions)
        # print(samples_from_memory.observation, samples_from_memory.next_observation)
        _, current_q_values = self.q_net(samples_from_memory.observation)
        # next_state, _ = self.q_net(samples_from_memory.next_observation)

        # Retrieve the q-values for the actions from the replay buffer
        # (batch_size, 1)
        current_q_values = torch.gather(
            current_q_values, dim=1, index=samples_from_memory.action.long()
        )

        forward_loss, inverse_loss = self.icm(
            observation=samples_from_memory.observation,
            next_observation=samples_from_memory.next_observation,
            action=samples_from_memory.action,
        )
        # print("forward / inverse : ", forward_loss, inverse_loss)

        intrinsic_reward = forward_loss.clone().detach().mean(-1)
        # print("intrinsic reward : ", intrinsic_reward)
        total_reward = intrinsic_reward.reshape(-1, 1) + samples_from_memory.reward
        # total_reward = samples_from_memory.reward

        with torch.no_grad():
            # Compute the next Q-values using the target network
            # (batch_size, n_actions)
            if self.n_step > 1:
                _, next_q_values = self.q_net_target(
                    samples_from_memory.next_nstep_observation
                )
            else:
                _, next_q_values = self.q_net_target(
                    samples_from_memory.next_observation
                )

            # Follow greedy policy: use the one with the highest value
            # (batch_size, 1)
            next_q_values = next_q_values.max(dim=1)[0].unsqueeze(1)

            # 1-step TD target
            target_q_values = (
                # samples_from_memory.reward
                total_reward
                + (1 - samples_from_memory.done) * self.gamma * next_q_values
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

        wandb.log(
            {
                "intrinsic_reward": intrinsic_reward.mean(),
                "extrinsic_reward": samples_from_memory.reward.mean(),
            },
            step=steps,
        )

        return (1 - self.icm_alpha) * dqn_loss + self.icm_alpha * (
            self.icm_beta * forward_loss.mean()
            + (1 - self.icm_beta) * inverse_loss.mean()
        )
