from typing import Tuple, Union, Type

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from une.memories.buffer.abstract import AbstractBuffer
from une.memories.buffer.uniform import Transition
from une.memories.buffer.per import TransitionPER
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
        return self.q_net(x)


class DQN:
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
    ):
        self.observation_shape = observation_shape
        self.features_dim = features_dim
        self.n_actions = n_actions

        self.buffer_size = buffer_size

        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.soft_update = soft_update
        self.learning_rate = learning_rate
        self.target_update_interval_steps = target_update_interval_steps

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_eps_max_steps = exploration_decay_eps_max_steps
        self._epsilon = self.exploration_initial_eps

        if use_gpu:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        self.memory_buffer = memory_buffer_cls(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            device=self.device,
            gradient_steps=self.gradient_steps,
            per_alpha=per_alpha,
            per_beta=per_beta,
        )

        self.q_net = QNetwork(
            representation_module_cls=representation_module_cls,
            observation_shape=observation_shape,
            features_dim=features_dim,
            action_dim=self.n_actions,
            device=self.device,
        ).to(self.device)

        self.q_net_target = QNetwork(
            representation_module_cls=representation_module_cls,
            observation_shape=observation_shape,
            features_dim=features_dim,
            action_dim=self.n_actions,
            device=self.device,
        ).to(self.device)
        self.hard_update_q_net_target()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.learning_rate
        )

        self.criterion = F.smooth_l1_loss

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
            current_q_values = self.q_net(observation)
            action = current_q_values.argmax(dim=1)
            if self.device in ["cuda", "mps"]:
                action = action.cpu()
            return action.numpy()[0]

    def choose_random_action(self) -> int:
        return np.random.choice(range(self.n_actions))

    def choose_epsilon_greedy_action(
        self, observation: torch.Tensor, steps: int
    ) -> int:
        if np.random.random() > self.epsilon(steps):
            return self.choose_greedy_action(observation)
        else:
            return self.choose_random_action()

    def hard_update_q_net_target(self) -> None:
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def soft_update_q_net_target(self) -> None:
        for p1, p2 in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            p1.data.copy_(self.tau * p2.data + (1.0 - self.tau) * p1.data)
    
    def soft_update_q_net_target_old(self) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def compute_loss(
        self,
        samples_from_memory: Union[Transition, TransitionPER],
        elementwise: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            # Compute the next Q-values using the target network
            # (batch_size, n_actions)
            next_q_values = self.q_net_target(samples_from_memory.next_observation)

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
        current_q_values = self.q_net(samples_from_memory.observation)

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
        if len(self.memory_buffer) < self.batch_size:
            return 0

        losses = []
        for g_step in range(self.gradient_steps + 1):
            samples_from_memory = self.memory_buffer.sample(
                batch_size=self.batch_size, to_tensor=True, g_step=g_step
            )

            if self.memory_buffer.memory_type == "per":
                elementwise_loss = self.compute_loss(
                    samples_from_memory=samples_from_memory, elementwise=True
                )
                loss = torch.mean(elementwise_loss * samples_from_memory.weights)
            else:
                loss = self.compute_loss(samples_from_memory=samples_from_memory)

            # Optimize the policy
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # PER: update priorities
            if self.memory_buffer.memory_type == "per":
                loss_for_prior = elementwise_loss.detach()
                if self.device in ["cuda", "mps"]:
                    loss_for_prior = loss_for_prior.cpu()
                loss_for_prior = loss_for_prior.numpy()
                new_priorities = loss_for_prior + self.memory_buffer.prior_eps
                self.memory_buffer.update_priorities(samples_from_memory.indices, new_priorities)

            losses.append(loss.item())

        # if not self.soft_update and (steps % self.target_update_interval_steps == 0):
        #     self.hard_update_q_net_target()
        
        # if self.soft_update:
        #     self.soft_update_q_net_target()

        #return np.mean(losses, 0)
