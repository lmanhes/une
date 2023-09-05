from typing import Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import wandb

from une.memories.buffers.abstract import AbstractBuffer
from une.memories.buffers.recurrent import RecurrentNStepPERBuffer
from une.memories.transitions import RecurrentTransition, NStepPERRecurrentTransition
from une.representations.abstract import AbstractRepresentation
from une.algos.r2d1.r2d1 import R2D1, RecurrentQNetwork
from une.exploration.icm import IntrinsicCuriosityModule


class ICMR2D1(R2D1):
    def __init__(
        self,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        features_dim: int,
        n_actions: int,
        representation_module_cls: Type[AbstractRepresentation],
        memory_buffer_cls: Type[AbstractBuffer],
        q_network_cls: Type[nn.Module] = RecurrentQNetwork,
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
        sequence_length: int = 80,
        burn_in: int = 40,
        over_lapping: int = 20,
        recurrent_dim: int = 256,
        recurrent_init_strategy: str = "first",
        intrinsic_reward_weight: float = 0.1,
        intrinsic_loss_weight: float = 0.1,
        icm_features_dim: int = 256,
        icm_forward_loss_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            features_dim=features_dim,
            n_actions=n_actions,
            representation_module_cls=representation_module_cls,
            memory_buffer_cls=memory_buffer_cls,
            q_network_cls=q_network_cls,
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
            sequence_length=sequence_length,
            burn_in=burn_in,
            over_lapping=over_lapping,
            recurrent_dim=recurrent_dim,
            recurrent_init_strategy=recurrent_init_strategy
        )
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.intrinsic_loss_weight = intrinsic_loss_weight
        self.icm_features_dim = icm_features_dim
        self.icm_forward_loss_weight = icm_forward_loss_weight

        self.icm = IntrinsicCuriosityModule(
            encoder=self.q_net.representation_module,
            input_dim=self.features_dim,
            features_dim=self.icm_features_dim,
            actions_dim=n_actions,
            forward_loss_weight=self.icm_forward_loss_weight,
            device=self.device
        ).to(self.device)


    @property
    def networks(self):
        return [self.q_net, self.icm]

    def parameters(self):
        return list(set(list(self.q_net.parameters()) + list(self.icm.parameters())))
    
    def memorize(self, observation: np.ndarray, reward: float, done: bool) -> None:
        assert (
            self._last_observation is not None
            and self._last_action is not None
            and self._last_h_recurrent is not None
            and self._last_c_recurrent is not None
        )

        with torch.no_grad():
            intrinsic_loss, intrinsic_reward = self.icm(
                observation=self._last_observation,
                next_observation=observation,
                action=self._last_action,
            )

        transition = RecurrentTransition(
            observation=self._last_observation,
            h_recurrent=self._last_h_recurrent.cpu().numpy(),
            c_recurrent=self._last_c_recurrent.cpu().numpy(),
            action=self._last_action,
            reward=(reward + self.intrinsic_reward_weight * intrinsic_reward).cpu().numpy(),
            done=done,
            next_observation=observation,
            next_h_recurrent=self._h_recurrent.cpu().numpy(),
            next_c_recurrent=self._c_recurrent.cpu().numpy(),
            # next_last_action=self._last_action,
        )
        self.memory_buffer.add(transition)

        self._last_reward = reward
        self._last_observation = observation
        self._last_h_recurrent = self._h_recurrent
        self._last_c_recurrent = self._c_recurrent

        if done:
            self.reset()

    def compute_loss(
        self,
        samples_from_memory: NStepPERRecurrentTransition,
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:
        if self.recurrent_init_strategy == "burnin":
            burnin_samples, learning_samples = self.memory_buffer.split_burnin(
                samples_from_memory
            )
            current_q_values = self.compute_q_values(
                learning_samples=learning_samples, burnin_samples=burnin_samples
            )
            target_q_values = self.compute_target_q_values(
                learning_samples=learning_samples, burnin_samples=burnin_samples,
                #intrinsic_reward=intrinsic_reward,
                #intrinsic_reward_weight=self.intrinsic_reward_weight,
            )
            
        else:
            learning_samples = samples_from_memory
            current_q_values = self.compute_q_values(learning_samples=learning_samples)
            target_q_values = self.compute_target_q_values(
                learning_samples=learning_samples,
                #intrinsic_reward=intrinsic_reward,
                #intrinsic_reward_weight=self.intrinsic_reward_weight,
            )

        intrinsic_loss, intrinsic_reward = self.icm(
                observation=learning_samples.observation,
                next_observation=learning_samples.next_observation,
                action=learning_samples.action,
            )

        loss = self.criterion(
            current_q_values, target_q_values.detach(), reduction="none"
        )
        total_loss = loss + self.intrinsic_loss_weight * intrinsic_loss

        priorities = self.calculate_priorities_from_td_error(td_error=loss)

        if self.memory_buffer_cls == RecurrentNStepPERBuffer:
            # masking
            for idx, length in enumerate(learning_samples.length):
                total_loss[idx, int(length) :] *= 0
            total_loss = torch.sum(total_loss, dim=1)
            total_loss = torch.mean(total_loss * learning_samples.weights.detach())  #
        else:
            total_loss = torch.mean(total_loss)

        # wandb.log(
        #     {
        #         "train_current_q_values": current_q_values.mean(),
        #         "train_target_q_values": target_q_values.mean(),
        #         # "train_td_error": loss.mean(),
        #     },
        #     step=steps,
        # )
        return total_loss, priorities
