from typing import Tuple, Union, Type, List

from loguru import logger
import numpy as np
import torch
import torch.nn as nn

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

        #self.recurrent = nn.LSTM(
        #    features_dim + action_dim + 1, self.recurrent_dim, batch_first=True
        #)
        self.recurrent = nn.LSTM(features_dim, self.recurrent_dim, batch_first=True)

        self.q_net = NoisyLinear(recurrent_dim, action_dim)

    def reset_noise(self):
        self.q_net.reset_noise()

    def init_recurrent(self, batch_size: int):
        return tuple(torch.zeros(batch_size, self.recurrent_dim) for _ in range(2))

    def get_first_step_recurrent(
        self, h_recurrent: torch.Tensor = None, c_recurrent: torch.Tensor = None
    ):
        h_recurrent = h_recurrent[:, 0, :].unsqueeze(0)
        c_recurrent = c_recurrent[:, 0, :].unsqueeze(0)
        return (h_recurrent, c_recurrent)

    def slice_burn_in(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, : self.burn_in, :], tensor[:, self.burn_in :, :]

    def burn_in_unroll(
        self,
        observation: torch.Tensor,
        #last_action: torch.Tensor,
        #last_reward: torch.Tensor,
        h_recurrent: torch.Tensor,
        c_recurrent: torch.Tensor,
    ):
        sequence_size = observation.shape[1]
        # print(
        #     "hidden before : ",
        #     h_recurrent.shape,
        #     np.array_equal(h_recurrent[0], h_recurrent[17]),
        # )

        h_recurrent_burn_in, c_recurrent_burn_in = self.get_first_step_recurrent(
            h_recurrent=h_recurrent, c_recurrent=c_recurrent
        )
        # print(
        #     "hidden first step : ", h_recurrent_burn_in.shape,
        #     np.array_equal(h_recurrent_burn_in[:, 0, :], c_recurrent_burn_in[:, 17, :])
        # )

        with torch.no_grad():
            observation = observation.reshape(-1, *self.observation_shape)
            x = self.representation_module(observation)
            x = x.view(-1, sequence_size, self.features_dim)
            #print("recurrent input : ", x.shape)
            #x = torch.cat((x, last_action, last_reward), dim=-1)
            _, (h_recurrent, c_recurrent) = self.recurrent(
                x, (h_recurrent_burn_in, c_recurrent_burn_in)
            )

        # print(
        #     "output unroll : ", h_recurrent, h_recurrent.shape, np.array_equal(h_recurrent[0, 0, :], h_recurrent[0, 17, :])
        # )
        return (h_recurrent, c_recurrent)

    def batched_inference(
        self,
        observation: torch.Tensor,
        #last_action: torch.Tensor,
        #last_reward: torch.Tensor,
        lengths: List[int],
        h_recurrent: torch.Tensor,
        c_recurrent: torch.Tensor,
    ):
        assert (
            lengths is not None
        ), "You must provide the list of sequence lengths of each batch element"

        sequence_size = observation.shape[1] - self.burn_in
        #print("LENGTHS : ", lengths)

        # last_action_ohot = (
        #     F.one_hot(last_action.long(), num_classes=self.action_dim)
        #     .squeeze(1)
        #     .float()
        # ).squeeze(2)

        observation_burn_in, observation = self.slice_burn_in(observation)
        h_recurrent_burn_in, h_recurrent = self.slice_burn_in(h_recurrent)
        c_recurrent_burn_in, c_recurrent = self.slice_burn_in(c_recurrent)
        # last_action_ohot_burn_in, last_action_ohot = self.slice_burn_in(
        #     last_action_ohot
        # )
        # last_reward_burn_in, last_reward = self.slice_burn_in(last_reward)
        lengths = [l - self.burn_in for l in lengths]

        h_recurrent, c_recurrent = self.burn_in_unroll(
            observation=observation_burn_in,
            #last_action=last_action_ohot_burn_in,
            #last_reward=last_reward_burn_in,
            h_recurrent=h_recurrent_burn_in,
            c_recurrent=c_recurrent_burn_in,
        )

        # h_recurrent, c_recurrent = self.get_first_step_recurrent(
        #     h_recurrent=h_recurrent, c_recurrent=c_recurrent
        # )

        observation = observation.reshape(-1, *self.observation_shape)
        x = self.representation_module(observation)
        x = x.view(-1, sequence_size, self.features_dim)
        #x = torch.cat((x, last_action_ohot, last_reward), dim=-1)
        x, (h_recurrent, c_recurrent) = self.recurrent(x, (h_recurrent, c_recurrent))
        x = self.q_net(x)
        return x, (h_recurrent, c_recurrent)

    def unbatched_inference(
        self,
        observation: torch.Tensor,
        #last_action: torch.Tensor,
        #last_reward: torch.Tensor,
        h_recurrent: torch.Tensor = None,
        c_recurrent: torch.Tensor = None,
    ):
        observation = observation.to(self.device)
        # last_action_ohot = (
        #     F.one_hot(last_action.long(), num_classes=self.action_dim)
        #     .squeeze(1)
        #     .float()
        # )

        if h_recurrent is None or c_recurrent is None:
            # logger.info(f"Init recurrent unbatched")
            h_recurrent, c_recurrent = self.init_recurrent(batch_size=1)

        x = self.representation_module(observation)
        #x = torch.cat((x, last_action_ohot, last_reward), dim=-1)
        x, (h_recurrent, c_recurrent) = self.recurrent(x, (h_recurrent, c_recurrent))
        x = self.q_net(x)
        return x, (h_recurrent, c_recurrent)

    def forward(
        self,
        observation: torch.Tensor,
        #last_action: torch.Tensor,
        #last_reward: torch.Tensor,
        lengths: List[int] = None,
        h_recurrent: torch.Tensor = None,
        c_recurrent: torch.Tensor = None,
        batched: bool = False,
    ):
        if batched:
            return self.batched_inference(
                observation=observation,
                #last_action=last_action,
                #last_reward=last_reward,
                lengths=lengths,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
            )
        else:
            return self.unbatched_inference(
                observation=observation,
                #last_action=last_action,
                #last_reward=last_reward,
                h_recurrent=h_recurrent,
                c_recurrent=c_recurrent,
            )


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
        **kwargs
    ):
        self.recurrent_dim = recurrent_dim
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self.over_lapping = over_lapping

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

        # self.observation_shape = observation_shape
        # self.observation_dtype = observation_dtype
        # self.features_dim = features_dim
        # self.n_actions = n_actions

        # self.representation_module_cls = representation_module_cls
        # self.q_network_cls = q_network_cls
        # self.memory_buffer_cls = memory_buffer_cls

        # self.gamma = gamma
        # self.batch_size = batch_size
        # self.gradient_steps = gradient_steps
        # self.buffer_size = buffer_size
        # self.n_step = n_step
        # self.learning_rate = learning_rate
        # self.max_grad_norm = max_grad_norm
        # self.target_update_interval_steps = target_update_interval_steps
        # self.soft_update = soft_update
        # self.tau = tau

        # self.exploration_initial_eps = exploration_initial_eps
        # self.exploration_final_eps = exploration_final_eps
        # self.exploration_decay_eps_max_steps = exploration_decay_eps_max_steps
        # self.use_gpu = use_gpu

        # self.per_alpha = per_alpha
        # self.per_beta = per_beta

        # self._epsilon = self.exploration_initial_eps

        # if self.use_gpu:
        #     if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        #         self.device = "mps"
        #     elif torch.cuda.is_available():
        #         self.device = "cuda"
        #     else:
        #         self.device = "cpu"
        # else:
        #     self.device = "cpu"

        # # RAM explode otherwise
        # torch.set_num_threads(1)

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
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            over_lapping=self.over_lapping,
            recurrent_dim=self.recurrent_dim,
        )

        # self.build_networks()
        # self.q_net = q_network_cls(
        #     representation_module_cls=self.representation_module_cls,
        #     observation_shape=self.observation_shape,
        #     features_dim=self.features_dim,
        #     action_dim=self.n_actions,
        #     recurrent_dim=self.recurrent_dim,
        #     burn_in=self.burn_in,
        #     device=self.device,
        # ).to(self.device)

        # self.q_net_target = (
        #     q_network_cls(
        #         representation_module_cls=self.representation_module_cls,
        #         observation_shape=self.observation_shape,
        #         features_dim=self.features_dim,
        #         action_dim=self.n_actions,
        #         recurrent_dim=self.recurrent_dim,
        #         burn_in=self.burn_in,
        #         device=self.device,
        #     )
        #     .to(self.device)
        #     .eval()
        # )
        # self.hard_update_q_net_target()
        # self.q_net_target.eval()

        # self.optimizer = None

        # self.criterion = F.smooth_l1_loss

        self._last_action = None
        self._last_observation = None
        self._last_reward = None
        self._last_h_recurrent = None
        self._last_c_recurrent = None

        for param_name, param_value in self.__dict__.copy().items():
            print(param_name, param_value)

    def build_networks(self):
        print("TOP")
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
            self._last_action = 0
        if self._last_reward is None:
            self._last_reward = 0

        if not isinstance(observation, (torch.Tensor, np.ndarray)):
            observation = np.array(observation)
        if not isinstance(self._last_action, (torch.Tensor, np.ndarray)):
            last_action = np.array(self._last_action)
        if not isinstance(self._last_reward, (torch.Tensor, np.ndarray)):
            last_reward = np.array(self._last_reward)
        # if not isinstance(self._last_h_recurrent, (torch.Tensor, np.ndarray)):
        #     self._last_h_recurrent = np.array(self._last_h_recurrent)
        # if not isinstance(self._last_c_recurrent, (torch.Tensor, np.ndarray)):
        #     self._last_c_recurrent = np.array(self._last_c_recurrent)

        # print("last_action : ", last_action)

        with torch.no_grad():
            observation = (
                torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            )
            last_action = (
                torch.from_numpy(last_action.reshape(-1, 1))
                .to(torch.int8)
                .to(self.device)
            )
            last_reward = (
                torch.from_numpy(last_reward.reshape(-1, 1)).float().to(self.device)
            )

            current_q_values, (h_recurrent, c_recurrent) = self.q_net(
                observation=observation,
                #last_action=last_action,
                #last_reward=last_reward,
                h_recurrent=self._last_h_recurrent,
                c_recurrent=self._last_c_recurrent,
                batched=False,
            )

            self._last_h_recurrent = h_recurrent.detach().clone()
            self._last_c_recurrent = c_recurrent.detach().clone()

            action = current_q_values.argmax(dim=1).detach()
            if self.device in ["cuda", "mps"]:
                action = action.cpu()
            return action.numpy()[0]

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
            c_recurrent=self._last_c_recurrent.numpy(),
            action=self._last_action,
            reward=reward,
            done=done,
            next_observation=observation,
        )
        self.memory_buffer.add(transition)

        self._last_reward = reward

        if done:
            self._last_action = None
            self._last_reward = None
            self._last_h_recurrent = None
            self._last_c_recurrent = None

    def compute_q_values(self, samples_from_memory: TransitionRecurrentOut):
        # Get current Q-values estimates
        # (batch_size, n_actions)
        current_q_values, _ = self.q_net(
            observation=samples_from_memory.observation,
            #last_action=samples_from_memory.last_action,
            #last_reward=samples_from_memory.last_reward,
            h_recurrent=samples_from_memory.h_recurrent,
            c_recurrent=samples_from_memory.c_recurrent,
            lengths=samples_from_memory.lengths,
            batched=True,
        )
        # Retrieve the q-values for the actions from the replay buffer
        # (batch_size, 1)

        # Pick only part of learning sequence
        actions = samples_from_memory.action[:, self.burn_in :]
        current_q_values = torch.gather(current_q_values, dim=-1, index=actions.long())
        return current_q_values

    def compute_target_q_values(
        self,
        samples_from_memory: TransitionRecurrentOut,
        intrinsic_reward: torch.Tensor = None,
        intrinsic_reward_weight: float = 0.1,
    ):
        with torch.no_grad():
            # Compute the next Q-values using the target network
            # (batch_size, n_actions)
            if self.n_step > 1:
                next_q_values, _ = self.q_net_target(
                    observation=samples_from_memory.next_observation,
                    #last_action=samples_from_memory.next_last_action,
                    #last_reward=samples_from_memory.next_last_reward,
                    h_recurrent=samples_from_memory.next_h_recurrent,
                    c_recurrent=samples_from_memory.next_c_recurrent,
                    lengths=samples_from_memory.lengths,
                    batched=True,
                )
            else:
                next_q_values, _ = self.q_net_target(
                    observation=samples_from_memory.next_observation,
                    #last_action=samples_from_memory.action,
                    #last_reward=samples_from_memory.reward,
                    h_recurrent=samples_from_memory.next_h_recurrent,
                    c_recurrent=samples_from_memory.next_c_recurrent,
                    lengths=samples_from_memory.lengths,
                    batched=True,
                )

            # Follow greedy policy: use the one with the highest value
            # (batch_size, 1)
            next_q_values = next_q_values.max(dim=-1)[0].unsqueeze(-1)

            # augment reward with intrinsic reward if exists
            reward = samples_from_memory.reward[:, self.burn_in :]
            if intrinsic_reward is not None:
                reward += intrinsic_reward * intrinsic_reward_weight

            dones = samples_from_memory.done[:, self.burn_in :]
            # 1-step TD target
            target_q_values = reward + (1 - dones) * self.gamma * next_q_values
            return target_q_values

    def compute_loss(
        self,
        samples_from_memory: Union[TransitionRecurrentOut, TransitionRecurrentOut],
        steps: int,
        elementwise: bool = False,
    ) -> torch.Tensor:

        current_q_values = self.compute_q_values(
            samples_from_memory=samples_from_memory
        )
        target_q_values = self.compute_target_q_values(
            samples_from_memory=samples_from_memory
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
