from pathlib import Path
from typing import Union, Tuple, Type

import numpy as np

from une.algos.dqn import DQN
from une.agents.abstract import AbstractAgent
from une.representations.abstract import AbstractRepresentation
from une.memories.buffer.uniform import Transition, UniformBuffer
from une.memories.buffer.ere import EREBuffer
from une.memories.buffer.per import PERBuffer


class DQNAgent(AbstractAgent):
    def __init__(
        self,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        observation_dtype: np.dtype,
        n_actions: int,
        memory_buffer_type: str = 'uniform',
        features_dim: int = 512,
        gamma: float = 0.99,
        batch_size: int = 32,
        gradient_steps: int = 1,
        buffer_size: int = int(1e6),
        learning_rate: float = 2.5e-4,
        max_grad_norm: float = 10,
        target_update_interval_steps: int = 1e4,
        soft_update: bool = False,
        tau: float = 1e-3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_decay_eps_max_steps: int = 1e4,
        train_freq: int = 4,
        use_gpu: bool = False,
        per_alpha: float = 0.7,
        per_beta: float = 0.4,
    ):
        super().__init__()
        self.train_freq = train_freq
        self.target_update_interval_steps = target_update_interval_steps
        self.soft_update = soft_update
        if self.soft_update:
            self.tau = tau
        else:
            self.tau = 1

        if memory_buffer_type == 'ere':
            memory_buffer_cls = EREBuffer
        elif memory_buffer_type == 'per':
            memory_buffer_cls = PERBuffer
        else:
            memory_buffer_cls = UniformBuffer

        self.algo = DQN(
            representation_module_cls=representation_module_cls,
            memory_buffer_cls=memory_buffer_cls,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            features_dim=features_dim,
            n_actions=n_actions,
            gamma=gamma,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            tau=self.tau,
            soft_update=soft_update,
            target_update_interval_steps=self.target_update_interval_steps,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_decay_eps_max_steps=exploration_decay_eps_max_steps,
            use_gpu=use_gpu,
            per_alpha=per_alpha,
            per_beta=per_beta
        )

        self.steps = 0

    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        self.steps += 1
        action = self.algo.choose_epsilon_greedy_action(observation, self.steps)

        if not evaluate and (self.steps % self.train_freq == 0):
            self.algo.learn(self.steps)

        return action

    def memorize(self, transition: Transition):
        self.algo.memory_buffer.add(transition)

    @property
    def epsilon(self) -> float:
        return self.algo.epsilon(steps=self.steps)

    def reset(self):
        raise NotImplementedError()

    def save(self, filename: Union[str, Path]):
        raise NotImplementedError()

    def load(self, filename: Union[str, Path]):
        raise NotImplementedError()
