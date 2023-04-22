from collections import OrderedDict
from typing import Tuple, Union, Type, List

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.buffer.sequence import (
    TransitionRecurrentIn,
    TransitionRecurrentOut,
    NStepSequencePERBuffer,
    TransitionRecurrentPEROut,
    TransitionEpisodicRecurrentIn
)
from une.representations.abstract import AbstractRepresentation
from une.algos.r2d1 import R2D1, RecurrentQNetwork
from une.curiosity.lifelong_episodic import LifeLongEpisodicModule


class NGUR2D1(R2D1):
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
        recurrent_init_strategy: str = "burnin",
        intrinsic_reward_weight: float = 0.1,
        icm_features_dim: int = 256,
        icm_forward_loss_weight: float = 0.2,
        ecm_memory_size: int = 300,
        ecm_k: int = 10,
        **kwargs,
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
            sequence_length=sequence_length,
            burn_in=burn_in,
            over_lapping=over_lapping,
            recurrent_dim=recurrent_dim,
            recurrent_init_strategy=recurrent_init_strategy,
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
            sequence=True
        ).to(self.device)

    def memorize(self, observation: np.ndarray, reward: float, done: bool) -> None:
        assert (
            self._last_observation is not None
            and self._last_action is not None
            and self._last_h_recurrent is not None
            and self._last_c_recurrent is not None
        )

        if not isinstance(observation, (torch.Tensor, np.ndarray)):
            observation = np.array(observation)
        observation = torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
        episodic_reward = self.lifelong_ecm.get_episodic_reward(observation=observation)

        transition = TransitionEpisodicRecurrentIn(
            observation=self._last_observation,
            h_recurrent=self._last_h_recurrent.cpu().numpy(),
            c_recurrent=self._last_c_recurrent.cpu().numpy(),
            action=self._last_action,
            reward=reward,
            done=done,
            next_observation=observation.cpu().numpy(),
            next_h_recurrent=self._h_recurrent.cpu().numpy(),
            next_c_recurrent=self._c_recurrent.cpu().numpy(),
            episodic_reward=episodic_reward,
        )
        self.memory_buffer.add(transition)

        self._last_reward = reward
        self._last_observation = observation
        self._last_h_recurrent = self._h_recurrent
        self._last_c_recurrent = self._c_recurrent

        if done:
            self.reset()
            self.lifelong_ecm.reset()

    def compute_loss(
        self,
        samples_from_memory: TransitionRecurrentOut,
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:
        intrinsic_loss, intrinsic_reward = self.lifelong_ecm(
            observation=samples_from_memory.observation,
            next_observation=samples_from_memory.next_observation,
            action=samples_from_memory.action,
            episodic_reward=samples_from_memory.episodic_reward,
        )
        # print("INTRINSIC MODULE : ", intrinsic_loss.shape, intrinsic_reward.shape)

        wandb.log({"intrinsic_reward": intrinsic_reward.mean()}, step=steps)

        # print("lengths : ", samples_from_memory.lengths, samples_from_memory.lengths.shape)
        if self.recurrent_init_strategy == "burnin":
            burnin_samples, learning_samples = self.memory_buffer.split_burnin(
                samples_from_memory
            )
            current_q_values = self.compute_q_values(
                learning_samples=learning_samples, burnin_samples=burnin_samples
            )
            target_q_values = self.compute_target_q_values(
                learning_samples=learning_samples,
                burnin_samples=burnin_samples,
                intrinsic_reward=intrinsic_reward,
                intrinsic_reward_weight=self.intrinsic_reward_weight,
            )
        else:
            learning_samples = samples_from_memory
            current_q_values = self.compute_q_values(learning_samples=learning_samples)
            target_q_values = self.compute_target_q_values(
                learning_samples=learning_samples
            )

        loss = self.criterion(
            current_q_values, target_q_values.detach(), reduction="none"
        )

        priorities = self.calculate_priorities_from_td_error(td_error=loss)

        if self.memory_buffer_cls == NStepSequencePERBuffer:
            # masking
            for idx, length in enumerate(learning_samples.lengths):
                loss[idx, int(length) :] *= 0
            loss = torch.sum(loss, dim=1)
            loss = torch.mean(loss * learning_samples.weights.detach())  #
        else:
            loss = torch.mean(loss)

        loss += intrinsic_loss

        wandb.log(
            {
                "train_current_q_values": current_q_values.mean(),
                "train_target_q_values": target_q_values.mean(),
                # "train_td_error": loss.mean(),
            },
            step=steps,
        )
        return loss, priorities
