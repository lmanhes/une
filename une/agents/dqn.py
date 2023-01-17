import os
from pathlib import Path
from typing import Union, Tuple, Type, List

import numpy as np
import torch

from une.algos.dqn import DQN
from une.agents.abstract import AbstractAgent
from une.representations.abstract import AbstractRepresentation
from une.memories.buffer.uniform import UniformBuffer, NStepUniformBuffer
from une.memories.buffer.ere import EREBuffer, NStepEREBuffer
from une.memories.buffer.per import PERBuffer, NStepPERBuffer
from une.memories.utils.transition import Transition


class DQNAgent(AbstractAgent):
    def __init__(
        self,
        name: str,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        observation_dtype: np.dtype,
        n_actions: int,
        memory_buffer_type: str = "uniform",
        n_step: int = 1,
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
        warmup: int = 1e4,
        train_freq: int = 4,
        save_freq: int = 1e3,
        use_gpu: bool = False,
        per_alpha: float = 0.7,
        per_beta: float = 0.4,
        steps: int = 0,
        episodes: int = 0,
        **kwargs,
    ):
        super().__init__()
        if memory_buffer_type == "ere":
            if n_step > 1:
                memory_buffer_cls = NStepEREBuffer
            else:
                memory_buffer_cls = EREBuffer
        elif memory_buffer_type == "per":
            if n_step > 1:
                memory_buffer_cls = NStepPERBuffer
            else:
                memory_buffer_cls = PERBuffer
        else:
            if n_step > 1:
                memory_buffer_cls = NStepUniformBuffer
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
            n_step=n_step,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            tau=tau,
            soft_update=soft_update,
            target_update_interval_steps=target_update_interval_steps,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_decay_eps_max_steps=exploration_decay_eps_max_steps,
            use_gpu=use_gpu,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

        self.name = name
        self.warmup = warmup
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.steps = steps
        self.episodes = episodes

    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate:
            self.steps += 1
        action = self.algo.choose_epsilon_greedy_action(observation, self.steps)

        if (
            not evaluate
            and (self.steps % self.train_freq == 0)
            and (self.steps > self.warmup)
        ):
            self.algo.learn(self.steps)

        #if self.steps % self.save_freq == 0:
        #    self.save(f"artifacts/{self.name}.pt")

        return action

    def memorize(self, transition: Transition):
        self.algo.memory_buffer.add(transition)
        if transition.done:
            self.episodes += 1

    @property
    def epsilon(self) -> float:
        return self.algo.epsilon(steps=self.steps)

    def reset(self):
        raise NotImplementedError()

    @property
    def _excluded_save_params(self) -> List[str]:
        return ["algo"]

    def get_agent_params(self):
        data = self.__dict__.copy()
        for param_name in self._excluded_save_params:
            if param_name in data:
                data.pop(param_name, None)
        return data

    def save(self, path: Union[str, Path]):
        save_object = self.algo.get_algo_save_object()
        save_object.update({"agent_params": self.get_agent_params()})
        torch.save(save_object, path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        load_object = torch.load(path)

        params = load_object["agent_params"]
        params.update(load_object["algo_params"])
        agent = cls(**params)
        agent.algo.memory_buffer = load_object["memory_buffer"]
        agent.algo.q_net.load_state_dict(load_object["q_net_state_dict"])
        agent.algo.hard_update_q_net_target()
        for target_param in agent.algo.q_net_target.parameters():
            target_param.requires_grad = False
        agent.algo.optimizer.load_state_dict(load_object["optimizer_state_dict"])

        print("Memory size : ", len(agent.algo.memory_buffer))
        print("Global steps : ", agent.steps, agent.name)
        print("Epsilon : ", agent.epsilon, agent.steps)

        return agent
