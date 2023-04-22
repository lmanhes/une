from pathlib import Path
from typing import Union, Tuple, Type, List

from loguru import logger
import numpy as np
import torch

from une.algos.dqn.dqn import DQN
from une.algos.dqn.icm_dqn import ICMDQN
from une.algos.dqn.noisy_dqn import NoisyDQN
from une.algos.r2d1.r2d1 import R2D1
from une.representations.abstract import AbstractRepresentation
from une.memories.buffers.uniform import UniformBuffer
from une.memories.buffers.nstep import NStepUniformBuffer
from une.memories.buffers.ere import EREBuffer, NStepEREBuffer
from une.memories.buffers.per import PERBuffer, NStepPERBuffer
from une.memories.buffers.recurrent import (
    RecurrentUniformBuffer,
    RecurrentNStepUniformBuffer,
    RecurrentNStepPERBuffer,
)


class Agent(object):
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
        recurrent: bool = False,
        exploration: str = "epsilon-greedy",
        intrinsic_reward_weight: float = 0.1,
        icm_features_dim: int = 256,
        icm_forward_loss_weight: float = 0.2,
        ecm_memory_size: int = 300,
        ecm_k: int = 10,
        sequence_length: int = 80,
        burn_in: int = 40,
        over_lapping: int = 20,
        recurrent_dim: int = 256,
        recurrent_init_strategy: str = "burnin",
        **kwargs,
    ):
        super().__init__()
        if recurrent:
            algo_cls = R2D1
            if n_step > 1:
                if memory_buffer_type == "per":
                    memory_buffer_cls = RecurrentNStepPERBuffer
                else:
                    memory_buffer_cls = RecurrentNStepUniformBuffer
            else:
                memory_buffer_cls = RecurrentUniformBuffer
        else:
            if exploration == "noisy":
                algo_cls = NoisyDQN
            elif exploration == "icm":
                algo_cls = ICMDQN
            else:
                algo_cls = DQN

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

        logger.info(f"memory_buffer_cls : {memory_buffer_cls}")
        logger.info(f"algo_cls : {algo_cls}")

        self.algo = algo_cls(
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
            intrinsic_reward_weight=intrinsic_reward_weight,
            icm_features_dim=icm_features_dim,
            icm_forward_loss_weight=icm_forward_loss_weight,
            ecm_memory_size=ecm_memory_size,
            ecm_k=ecm_k,
            sequence_length=sequence_length,
            burn_in=burn_in,
            over_lapping=over_lapping,
            recurrent_dim=recurrent_dim,
            recurrent_init_strategy=recurrent_init_strategy,
        )

        self.name = name
        self.warmup = warmup
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.steps = steps
        self.episodes = episodes
        self.recurrent = recurrent

    def act(self, observation: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate:
            self.steps += 1

        act_random = (self.steps < self.warmup) and not evaluate
        if act_random:
            print(f"Act random : {act_random} -- steps {self.steps}")
        action = self.algo.act(
            observation=observation,
            steps=self.steps,
            random=act_random,
            evaluate=evaluate,
        )

        if (
            not evaluate
            and (self.steps % self.train_freq == 0)
            and (self.steps > self.warmup)
        ):
            # logger.info(f"Steps : {self.steps}")
            self.algo.learn(self.steps)

        # if self.steps % self.save_freq == 0:
        #    self.save(f"artifacts/{self.name}.pt")

        return action

    def reset(self):
        self.algo.reset()

    def memorize(self, observation: np.ndarray, reward: float, done: bool):
        self.algo.memorize(observation=observation, reward=reward, done=done)
        if done:
            self.episodes += 1

    @property
    def epsilon(self) -> float:
        return self.algo.epsilon(steps=self.steps)

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

        return agent
