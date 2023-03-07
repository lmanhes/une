from collections import OrderedDict
from typing import Tuple, Union, Type, List

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from une.algos.noisy_dqn import NoisyDQN, NoisyLinear
from une.memories.buffer.abstract import AbstractBuffer
from une.memories.buffer.sequence import TransitionRecurrentIn, TransitionRecurrentOut
from une.representations.abstract import AbstractRepresentation


def signed_hyperbolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed hyperbolic transform, inverse of signed_parabolic."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def signed_parabolic(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Signed parabolic transform, inverse of signed_hyperbolic."""
    z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
    return torch.sign(x) * (torch.square(z) - 1)


class RecurrentQNetwork(nn.Module):
    def __init__(
        self,
        representation_module_cls: Type[AbstractRepresentation],
        observation_shape: Union[int, Tuple[int]],
        features_dim: int,
        action_dim: int,
        recurrent_dim: int,
        burn_in: int,
        device: str = None,
    ):
        super().__init__()
        self.device = device
        self.burn_in = burn_in
        self.recurrent_dim = recurrent_dim
        self.action_dim = action_dim
        self.features_dim = features_dim
        self.observation_shape = observation_shape

        self.representation_module = representation_module_cls(
            input_shape=observation_shape, features_dim=features_dim
        )

        # self.recurrent = nn.LSTM(
        #    features_dim + action_dim + 1, self.recurrent_dim, batch_first=True
        # )
        self.recurrent = nn.LSTM(features_dim, self.recurrent_dim, batch_first=True)

        # self.advantage_head = nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("a_noisy1", NoisyLinear(self.recurrent_dim, self.recurrent_dim)),
        #             ("a_relu", nn.ReLU()),
        #             ("a_noisy2", NoisyLinear(self.recurrent_dim, self.action_dim)),
        #         ]
        #     )
        # )
        self.advantage_head = NoisyLinear(self.recurrent_dim, self.action_dim)

        # self.value_head = nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("v_noisy1", NoisyLinear(self.recurrent_dim, self.recurrent_dim)),
        #             ("v_relu", nn.ReLU()),
        #             ("v_noisy2", NoisyLinear(self.recurrent_dim, 1)),
        #         ]
        #     )
        # )
        self.value_head = NoisyLinear(self.recurrent_dim, 1)

    # self.q_net = NoisyLinear(recurrent_dim, action_dim)

    def reset_noise(self):
        self.advantage_head.reset_noise()
        self.value_head.reset_noise()

    # self.q_net.reset_noise()

    def init_recurrent(self, batch_size: int):
        return tuple(torch.zeros(1, batch_size, self.recurrent_dim) for _ in range(2))

    def forward(
        self,
        observation: torch.Tensor,
        # last_action: torch.Tensor,
        # last_reward: torch.Tensor,
        lengths: List[int] = None,
        h_recurrent: torch.Tensor = None,
        c_recurrent: torch.Tensor = None,
    ):
        assert observation.ndim == 2 + len(self.observation_shape)
        batch_size = observation.shape[0]
        sequence_size = observation.shape[1]

        if h_recurrent is None or c_recurrent is None:
            h_recurrent, c_recurrent = self.init_recurrent(batch_size=batch_size)

        # last_action_ohot = (
        #     F.one_hot(last_action.long(), num_classes=self.action_dim)
        #     .squeeze(1)
        #     .float()
        # ).squeeze(2)

        x = torch.flatten(observation, 0, 1)  # Merge batch and time dimension.
        # print("obs flatten : ", x.shape)
        x = self.representation_module(x)
        # print("obs state : ", x.shape)
        x = x.view(-1, sequence_size, self.features_dim)
        # print("obs state reshaped : ", x.shape)
        # x = torch.cat((x, last_action_ohot, last_reward), dim=-1)
        x, (h_recurrent, c_recurrent) = self.recurrent(x, (h_recurrent, c_recurrent))
        # print("recurrent out : ", x.shape, h_recurrent.shape)

        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        advantages = self.advantage_head(x)  # [T*B, action_dim]
        values = self.value_head(x)  # [T*B, 1]

        q_values = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))
        q_values = q_values.view(
            batch_size, sequence_size, -1
        )  # reshape to in the range [B, T, action_dim]

        # q_values = self.q_net(x)
        # print("q_values : ", q_values.shape)
        return q_values, (h_recurrent, c_recurrent)


class R2D1(NoisyDQN):
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
        **kwargs,
    ):
        self.recurrent_dim = recurrent_dim
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.over_lapping = over_lapping
        self.recurrent_init_strategy = recurrent_init_strategy

        logger.info(f"recurrent_init_strategy : {recurrent_init_strategy}")

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

        self._last_action = None
        self._last_observation = None
        self._last_reward = None
        self._last_h_recurrent, self._last_c_recurrent = self.q_net.init_recurrent(
            batch_size=1
        )
        self._h_recurrent = None
        self._c_recurrent = None

    def build_memory(self):
        self.memory_buffer = self.memory_buffer_cls(
            buffer_size=self.buffer_size,
            n_step=self.n_step,
            observation_shape=self.observation_shape,
            observation_dtype=self.observation_dtype,
            device=self.device,
            gradient_steps=self.gradient_steps,
            per_alpha=self.per_alpha,
            per_beta=self.per_beta,
            gamma=self.gamma,
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            over_lapping=self.over_lapping,
            recurrent_dim=self.recurrent_dim,
        )

    def build_networks(self):
        self.q_net = self.q_network_cls(
            representation_module_cls=self.representation_module_cls,
            observation_shape=self.observation_shape,
            features_dim=self.features_dim,
            action_dim=self.n_actions,
            recurrent_dim=self.recurrent_dim,
            burn_in=self.burn_in,
            device=self.device,
        ).to(self.device)

        self.q_net_target = (
            self.q_network_cls(
                representation_module_cls=self.representation_module_cls,
                observation_shape=self.observation_shape,
                features_dim=self.features_dim,
                action_dim=self.n_actions,
                recurrent_dim=self.recurrent_dim,
                burn_in=self.burn_in,
                device=self.device,
            )
            .to(self.device)
            .eval()
        )
        self.hard_update_q_net_target()
        self.q_net_target.eval()

    def choose_greedy_action(self, observation: torch.Tensor) -> int:
        if self._last_action is None:
            self._last_action = np.array(0)
        if self._last_reward is None:
            self._last_reward = np.array(0)

        if not isinstance(observation, (torch.Tensor, np.ndarray)):
            observation = np.array(observation)
        # if not isinstance(self._last_action, (torch.Tensor, np.ndarray)):
        #     last_action = np.array(self._last_action)
        # if not isinstance(self._last_reward, (torch.Tensor, np.ndarray)):
        #     last_reward = np.array(self._last_reward)
        # if not isinstance(self._last_h_recurrent, (torch.Tensor, np.ndarray)):
        #     self._last_h_recurrent = np.array(self._last_h_recurrent)
        # if not isinstance(self._last_c_recurrent, (torch.Tensor, np.ndarray)):
        #     self._last_c_recurrent = np.array(self._last_c_recurrent)

        # print("last_action : ", last_action)

        with torch.no_grad():
            observation = (
                torch.from_numpy(observation)
                .view(1, 1, *self.observation_shape)
                .float()
                .to(self.device)
            )
            # last_action = (
            #     torch.from_numpy(self._last_action)
            #     .view(1, 1, 1)
            #     .to(torch.int8)
            #     .to(self.device)
            # )
            # last_reward = (
            #     torch.from_numpy(self._last_action)
            #     .view(1, 1, 1)
            #     .float()
            #     .to(self.device)
            # )

            # if self._last_h_recurrent is not None:
            #     print(self._last_h_recurrent.shape, self._last_h_recurrent, np.zeros_like(self._last_h_recurrent).shape)
            #     print("RECURRENT ZERTOS : ", not np.array_equal(self._last_h_recurrent, np.zeros_like(self._last_h_recurrent)))
            # else:
            #     print("RECURRENT ZERTOS None: ")

            current_q_values, (h_recurrent, c_recurrent) = self.q_net(
                observation=observation,
                # last_action=last_action,
                # last_reward=last_reward,
                h_recurrent=self._last_h_recurrent,
                c_recurrent=self._last_c_recurrent,
            )

            action = current_q_values.argmax(dim=-1).detach()[0][0]
            if self.device in ["cuda", "mps"]:
                action = action.cpu()
            action = action.numpy()
            return action, (h_recurrent.detach(), c_recurrent.detach())
        
    def act(self, observation: np.ndarray, steps: int, random: bool = False, evaluate: bool = False) -> int:
        if random:
            action = self.choose_random_action()
            self._h_recurrent = self._last_h_recurrent
            self._c_recurrent = self._last_c_recurrent
        else:
            action, (h_recurrent, c_recurrent) = self.choose_greedy_action(
                observation=observation
            )
            self._h_recurrent = h_recurrent.detach().clone()
            self._c_recurrent = c_recurrent.detach().clone()
            
        self._last_observation = observation
        self._last_action = action
        if evaluate:
            self._last_h_recurrent = self._h_recurrent
            self._last_c_recurrent = self._c_recurrent
        return action

    def reset(self):
        self._last_action = None
        self._last_reward = None
        self._last_h_recurrent, self._last_c_recurrent = self.q_net.init_recurrent(
            batch_size=1
        )

    def memorize(self, observation: np.ndarray, reward: float, done: bool) -> None:
        assert (
            self._last_observation is not None
            and self._last_action is not None
            and self._last_h_recurrent is not None
            and self._last_c_recurrent is not None
        )

        transition = TransitionRecurrentIn(
            observation=self._last_observation,
            h_recurrent=self._last_h_recurrent.numpy(),
            c_recurrent=self._last_h_recurrent.numpy(),
            action=self._last_action,
            reward=reward,
            done=done,
            next_observation=observation,
            #next_h_recurrent=self._h_recurrent.numpy(),
            #next_c_recurrent=self._c_recurrent.numpy(),
            # next_last_action=self._last_action,
        )
        self.memory_buffer.add(transition)

        self._last_reward = reward
        self._last_observation = observation
        self._last_h_recurrent = self._h_recurrent
        self._last_c_recurrent = self._c_recurrent

        if done:
            self.reset()

    def get_first_step_recurrent(
        self, h_recurrent: torch.Tensor = None, c_recurrent: torch.Tensor = None
    ):
        h_recurrent = h_recurrent[:, 0:1, :]
        c_recurrent = c_recurrent[:, 0:1, :]
        h_recurrent = h_recurrent.swapaxes(0, 1)
        c_recurrent = c_recurrent.swapaxes(0, 1)
        return (h_recurrent, c_recurrent)

    def burn_in_unroll(
        self,
        observation: torch.Tensor,
        # last_action: torch.Tensor,
        # last_reward: torch.Tensor,
        h_recurrent: torch.Tensor,
        c_recurrent: torch.Tensor,
    ):
        with torch.no_grad():
            _, (h_recurrent, c_recurrent) = self.q_net(
                observation=observation,
                # last_action=last_action,
                # last_reward=last_reward,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
            )

        # print(
        #     "output unroll : ", h_recurrent, h_recurrent.shape, np.array_equal(h_recurrent[0, 0, :], h_recurrent[0, 17, :])
        # )
        return (h_recurrent, c_recurrent)

    def compute_q_values(
        self,
        learning_samples: TransitionRecurrentOut,
        burnin_samples: TransitionRecurrentOut = None,
    ):
        if self.recurrent_init_strategy == "zeros":
            h_recurrent, c_recurrent = self.q_net.init_recurrent(
                batch_size=learning_samples.observation.shape[0]
            )
        elif self.recurrent_init_strategy == "first":
            h_recurrent, c_recurrent = self.get_first_step_recurrent(
                h_recurrent=learning_samples.h_recurrent,
                c_recurrent=learning_samples.c_recurrent,
            )
        else:
            assert burnin_samples is not None
            h_recurrent, c_recurrent = self.get_first_step_recurrent(
                h_recurrent=burnin_samples.h_recurrent,
                c_recurrent=burnin_samples.c_recurrent,
            )
            h_recurrent, c_recurrent = self.burn_in_unroll(
                observation=burnin_samples.observation,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
                # last_action=burnin_samples.last_action,
                # last_reward=burnin_samples.last_reward,
            )

        current_q_values, _ = self.q_net(
            observation=learning_samples.observation,
            # last_action=learning_samples.last_action,
            # last_reward=learning_samples.last_reward,
            h_recurrent=h_recurrent,
            c_recurrent=c_recurrent,
            lengths=learning_samples.lengths,
        )
        # Retrieve the q-values for the actions from the replay buffer
        # (batch_size, 1)
        current_q_values = torch.gather(
            current_q_values, dim=-1, index=learning_samples.action.long()
        )
        return current_q_values.squeeze(-1)

    def compute_target_q_values(
        self,
        learning_samples: TransitionRecurrentOut,
        burnin_samples: TransitionRecurrentOut = None,
        intrinsic_reward: torch.Tensor = None,
        intrinsic_reward_weight: float = 0.1,
    ):
        if self.recurrent_init_strategy == "zeros":
            h_recurrent, c_recurrent = self.q_net.init_recurrent(
                batch_size=learning_samples.next_observation.shape[0]
            )
        elif self.recurrent_init_strategy == "first":
            h_recurrent, c_recurrent = self.get_first_step_recurrent(
                h_recurrent=learning_samples.h_recurrent,
                c_recurrent=learning_samples.c_recurrent,
            )
        else:
            assert burnin_samples is not None
            h_recurrent, c_recurrent = self.get_first_step_recurrent(
                h_recurrent=burnin_samples.h_recurrent,
                c_recurrent=burnin_samples.c_recurrent,
            )
            h_recurrent, c_recurrent = self.burn_in_unroll(
                observation=burnin_samples.next_observation,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
                # last_action=burnin_samples.last_action,
                # last_reward=burnin_samples.last_reward,
            )

        with torch.no_grad():
            # Compute the next Q-values using the target network
            # (batch_size, n_actions)

            target_q_values, _ = self.q_net_target(
                observation=learning_samples.next_observation,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
                # last_action=learning_samples.next_last_action,
                # last_reward=learning_samples.reward,
                lengths=learning_samples.lengths,
            )

            # DQN : Follow greedy policy: use the one with the highest value
            # (batch_size, 1)
            target_q_values = target_q_values.max(dim=-1)[0].unsqueeze(-1)

            # R2D1 : use best actions of Q_net for 'next_observation'
            # target_actions = torch.argmax(
            #     self.q_net(
            #         observation=learning_samples.next_observation,
            #         h_recurrent=h_recurrent.clone(),
            #         c_recurrent=c_recurrent.clone(),
            #         # last_action=learning_samples.next_last_action,
            #         # last_reward=learning_samples.reward,
            #     )[0],
            #     dim=-1,
            # ).unsqueeze(-1)
            # # # print("target actions : ", target_actions.shape, target_q_values.shape)
            # target_q_values = torch.gather(
            #     target_q_values, dim=-1, index=target_actions.long()
            # )
            target_q_values = signed_parabolic(target_q_values, 0.001)

            # augment reward with intrinsic reward if exists
            # reward = samples_from_memory.reward[:, self.burn_in :]
            reward = learning_samples.reward
            if intrinsic_reward is not None:
                reward += intrinsic_reward * intrinsic_reward_weight

            # print("YOLOOO : ", reward.shape)

            # dones = samples_from_memory.done[:, self.burn_in :]
            # 1-step TD target
            target_q_values = (
                reward + (1 - learning_samples.done) * self.gamma * target_q_values
            )
            #return target_q_values.squeeze(-1)
            return signed_hyperbolic(target_q_values, 0.001).squeeze(-1)

    def compute_loss(
        self,
        samples_from_memory: TransitionRecurrentOut,
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:

        # print("lengths : ", samples_from_memory.lengths, samples_from_memory.lengths.shape)
        if self.recurrent_init_strategy == "burnin":
            burnin_samples, learning_samples = self.memory_buffer.split_burnin(
                samples_from_memory
            )
            current_q_values = self.compute_q_values(
                learning_samples=learning_samples, burnin_samples=burnin_samples
            )
            target_q_values = self.compute_target_q_values(
                learning_samples=learning_samples, burnin_samples=burnin_samples
            )
        else:
            current_q_values = self.compute_q_values(
                learning_samples=samples_from_memory
            )
            target_q_values = self.compute_target_q_values(
                learning_samples=samples_from_memory
            )

        # current_q_values = current_q_values[:, :samples_from_memory.lengths.reshape(-1, 1)]
        # target_q_values = target_q_values[:, :samples_from_memory.lengths.reshape(-1, 1)]

        # Sums over time dimension.
        # td_error = current_q_values - target_q_values.detach()
        # loss = 0.5 * torch.sum(torch.square(td_error), dim=1)
        # loss = td_error**2
        # loss = torch.mean(0.5 * torch.square(td_error))
        # print("current_q_values : ", current_q_values.shape)
        loss = self.criterion(current_q_values, target_q_values)

        wandb.log(
            {
                "train_current_q_values": current_q_values.mean(),
                "train_target_q_values": target_q_values.mean(),
                # "train_td_error": td_error.mean(),
            },
            step=steps,
        )

        # print("loss : ", loss.shape)

        # loss = torch.mean(loss * weights.detach())
        return loss

    def learn(self, steps: int) -> float:
        if self.memory_buffer.n_sequences < self.batch_size:
            return 0
        loss = super().learn(steps)
        self.q_net.reset_noise()
        self.q_net_target.reset_noise()
        return loss
