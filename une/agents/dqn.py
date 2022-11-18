from pathlib import Path
from typing import Union, Tuple, Type

import numpy as np

from une.agents.abstract import AbstractAgent
from une.algos.dqn import DQN
from une.representations.abstract import AbstractRepresentation
from une.memories.buffer.uniform import Transition


class DQNAgent(AbstractAgent):
    def __init__(
        self,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        n_actions: int,
        features_dim: int = 512,
        gamma: float = 0.99,
        batch_size: int = 32,
        gradient_steps: int = 1,
        buffer_size: int = int(1e6),
        learning_rate: float = 2.5e-4,
        max_grad_norm: float = 10,
        target_update_interval_steps: int = 1e4,
        soft_update: bool = False,
        tau: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_decay_eps_max_steps: int = 1e4,
        train_freq: int = 4,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.train_freq = train_freq
        self.target_update_interval_steps = target_update_interval_steps
        self.soft_update = soft_update
        if self.soft_update:
            self.tau = tau
        else:
            self.tau = 1

        self.algo = DQN(
            representation_module_cls=representation_module_cls,
            observation_shape=observation_shape,
            features_dim=features_dim,
            n_actions=n_actions,
            gamma=gamma,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            tau=self.tau,
            target_update_interval_steps=self.target_update_interval_steps,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_decay_eps_max_steps=exploration_decay_eps_max_steps,
            use_gpu=use_gpu
        )

        self.steps = 0

    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        self.steps += 1
        action = self.algo.choose_epsilon_greedy_action(observation, self.steps)

        if self.steps % self.train_freq == 0 and not evaluate:
            loss = self.algo.learn()

        if (self.steps % self.target_update_interval_steps == 0) or self.soft_update:
            #print("Update target network")
            self.algo.soft_update_q_net_target()

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
