from typing import Tuple, Union, Type

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from une.memories.buffer.abstract import AbstractBuffer
from une.representations.abstract import AbstractRepresentation
from une.algos.dqn import DQN
from une.exploration.noisy import NoisyLinear


class NoisyQNetwork(nn.Module):
    def __init__(
        self,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        features_dim: int,
        action_dim: int,
        device: str = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.representation_module = representation_module_cls(
            input_shape=observation_shape, features_dim=features_dim
        )

        self.q_net = NoisyLinear(features_dim, action_dim)

    def forward(self, observations: torch.Tensor):
        observations = observations.to(self.device)
        x = self.representation_module(observations)
        return x, self.q_net(x)
    
    def reset_noise(self):
        self.q_net.reset_noise()
    

class NoisyDQN(DQN):
    def __init__(
        self,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        features_dim: int,
        n_actions: int,
        representation_module_cls: Type[AbstractRepresentation],
        memory_buffer_cls: Type[AbstractBuffer],
        q_network_cls: Type[nn.Module] = NoisyQNetwork,
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
        use_gpu: bool = False,
        per_alpha: float = 0.7,
        per_beta: float = 0.4,
        **kwargs
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
            use_gpu=use_gpu,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

    def choose_action(self, observation: torch.Tensor, steps: int) -> int:
         return self.choose_greedy_action(observation=observation)
    
    def learn(self, steps: int) -> float:
        loss = super().learn(steps)
        self.q_net.reset_noise()
        self.q_net_target.reset_noise()
        return loss
