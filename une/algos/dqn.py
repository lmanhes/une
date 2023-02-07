from copy import deepcopy
from typing import Tuple, Union, Type, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.buffer.uniform import Transition, TransitionNStep
from une.memories.buffer.per import TransitionPER, TransitionNStepPER
from une.representations.abstract import AbstractRepresentation


class QNetwork(nn.Module):
    def __init__(
        self,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        features_dim: int,
        action_dim: int,
        device: str = None,
    ):
        super().__init__()
        self.device = device
        self.representation_module = representation_module_cls(
            input_shape=observation_shape, features_dim=features_dim
        )

        self.q_net = nn.Linear(features_dim, action_dim)

    def forward(self, observations: torch.Tensor):
        observations = observations.to(self.device)
        x = self.representation_module(observations)
        return x, self.q_net(x)


class DQN:
    def __init__(
        self,
        observation_shape: Tuple[int],
        observation_dtype: np.dtype,
        features_dim: int,
        n_actions: int,
        representation_module_cls: Type[AbstractRepresentation],
        memory_buffer_cls: Type[AbstractBuffer],
        q_network_cls: Type[nn.Module] = QNetwork,
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
        **kwargs
    ):
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.features_dim = features_dim
        self.n_actions = n_actions

        self.representation_module_cls = representation_module_cls
        self.q_network_cls = q_network_cls
        self.memory_buffer_cls = memory_buffer_cls

        self.gamma = gamma
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.target_update_interval_steps = target_update_interval_steps
        self.soft_update = soft_update
        self.tau = tau

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_eps_max_steps = exploration_decay_eps_max_steps
        self.use_gpu = use_gpu

        self.per_alpha = per_alpha
        self.per_beta = per_beta

        self._epsilon = self.exploration_initial_eps

        if self.use_gpu:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        print("torch.get_num_threads : ", torch.get_num_threads())
        torch.set_num_threads(1)

        self.memory_buffer = memory_buffer_cls(
            buffer_size=self.buffer_size,
            n_step=self.n_step,
            observation_shape=self.observation_shape,
            observation_dtype=self.observation_dtype,
            device=self.device,
            gradient_steps=self.gradient_steps,
            per_alpha=self.per_alpha,
            per_beta=self.per_beta,
            gamma=self.gamma,
        )

        self.q_net = q_network_cls(
            representation_module_cls=self.representation_module_cls,
            observation_shape=self.observation_shape,
            features_dim=self.features_dim,
            action_dim=self.n_actions,
            device=self.device,
        ).to(self.device)

        self.q_net_target = (
            q_network_cls(
                representation_module_cls=self.representation_module_cls,
                observation_shape=self.observation_shape,
                features_dim=self.features_dim,
                action_dim=self.n_actions,
                device=self.device,
            )
            .to(self.device)
            .eval()
        )
        self.hard_update_q_net_target()
        self.q_net_target.eval()

        self.optimizer = None

        self.criterion = F.smooth_l1_loss

    @property
    def networks(self):
        return [self.q_net]

    def parameters(self):
        return self.q_net.parameters()

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def epsilon(self, steps: int) -> float:
        return self.exploration_final_eps + (
            self.exploration_initial_eps - self.exploration_final_eps
        ) * np.exp(-1.0 * steps / self.exploration_decay_eps_max_steps)

    def choose_greedy_action(self, observation: torch.Tensor) -> int:
        if not isinstance(observation, (torch.Tensor, np.ndarray)):
            observation = np.array(observation)

        with torch.no_grad():
            observation = (
                torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            )
            _, current_q_values = self.q_net(observation)
            action = current_q_values.argmax(dim=1).detach()
            if self.device in ["cuda", "mps"]:
                action = action.cpu()
            return action.numpy()[0]

    def choose_random_action(self) -> int:
        return np.random.choice(range(self.n_actions))

    def choose_action(self, observation: torch.Tensor, steps: int) -> int:
        if np.random.random() > self.epsilon(steps):
            return self.choose_greedy_action(observation)
        else:
            return self.choose_random_action()

    def hard_update_q_net_target(self) -> None:
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        for target_param in self.q_net_target.parameters():
            target_param.requires_grad = False

    def soft_update_q_net_target(self) -> None:
        params = deepcopy(list(self.q_net.parameters()))
        for param in params:
            param.requires_grad = False

        # for p1, p2 in zip(self.q_net_target.parameters(), params):
        #     p1.data.copy_(self.tau * p2.data + (1.0 - self.tau) * p1.data)

        with torch.no_grad():
            for param, target_param in zip(params, self.q_net_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(
                    target_param.data, param.data, alpha=self.tau, out=target_param.data
                )

        for target_param in self.q_net_target.parameters():
            target_param.requires_grad = False

        del params

    def compute_loss(
        self,
        samples_from_memory: Union[
            Transition, TransitionNStep, TransitionPER, TransitionNStepPER
        ],
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:
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
                samples_from_memory.reward
                + (1 - samples_from_memory.done) * self.gamma * next_q_values
            )

        # Get current Q-values estimates
        # (batch_size, n_actions)
        _, current_q_values = self.q_net(samples_from_memory.observation)

        # Retrieve the q-values for the actions from the replay buffer
        # (batch_size, 1)
        current_q_values = torch.gather(
            current_q_values, dim=1, index=samples_from_memory.action.long()
        )

        # Compute Huber loss (less sensitive to outliers)
        if elementwise:
            return self.criterion(
                current_q_values.squeeze(1),
                target_q_values.squeeze(1),
                reduction="none",
            )
        else:
            return self.criterion(
                current_q_values.squeeze(1), target_q_values.squeeze(1)
            )

    def learn(self, steps: int) -> float:
        if not self.optimizer:
            self.set_optimizer()

        if len(self.memory_buffer) < self.batch_size:
            return 0

        losses = []
        for g_step in range(self.gradient_steps + 1):
            samples_from_memory = self.memory_buffer.sample(
                batch_size=self.batch_size, to_tensor=True, g_step=g_step
            )

            if self.memory_buffer.memory_type == "per":
                elementwise_loss = self.compute_loss(
                    samples_from_memory=samples_from_memory,
                    elementwise=True,
                    steps=steps,
                )
                loss = torch.mean(elementwise_loss * samples_from_memory.weights)
            else:
                loss = self.compute_loss(
                    samples_from_memory=samples_from_memory, steps=steps
                )
            losses.append(loss.item())

            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # PER: update priorities
            if self.memory_buffer.memory_type == "per":
                loss_for_prior = elementwise_loss.detach()
                if self.device in ["cuda", "mps"]:
                    loss_for_prior = loss_for_prior.cpu()
                loss_for_prior = loss_for_prior.numpy()
                new_priorities = loss_for_prior + self.memory_buffer.prior_eps
                self.memory_buffer.update_priorities(
                    samples_from_memory.indices, new_priorities
                )

        if not self.soft_update and (steps % self.target_update_interval_steps == 0):
            self.hard_update_q_net_target()

        if self.soft_update:
            self.soft_update_q_net_target()

        return np.mean(losses, 0)

    @property
    def _excluded_save_params(self) -> List[str]:
        return ["memory_buffer", "q_net", "q_net_target", "optimizer", "criterion"]

    def get_algo_params(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        for param_name in self._excluded_save_params:
            if param_name in data:
                data.pop(param_name, None)
        return data

    def get_algo_save_object(self) -> Dict[str, Any]:
        return {
            "algo_params": self.get_algo_params(),
            "memory_buffer": self.memory_buffer,
            "q_net_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion": self.criterion,
        }
