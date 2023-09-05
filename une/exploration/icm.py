from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_dim: int,
        actions_dim: int,
        encoder: nn.Module,
        forward_loss_weight: float,
        device: str = "cpu",
    ):
        super(IntrinsicCuriosityModule, self).__init__()
        self.encoder = encoder.to(device)
        self.forward_loss_weight = forward_loss_weight
        self.device = device

        self.inverse_net = nn.Sequential(
            nn.Linear(input_dim * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, actions_dim),
        )
        self.forward_net = nn.Sequential(
            nn.Linear(input_dim + actions_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, input_dim),
        )

        self.actions_dim = actions_dim
        self.forward_criterion = F.mse_loss
        self.inverse_criterion = F.cross_entropy

    def forward(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        next_observation: Union[np.ndarray, torch.Tensor],
        action: Union[np.int64, torch.Tensor],
    ):
        is_sequence = False
        if action.ndim == 3:
            is_sequence = True

        if isinstance(observation, np.ndarray):
            observation = (
                torch.from_numpy(observation).unsqueeze(0).float().to(self.device)
            )

        if isinstance(next_observation, np.ndarray):
            next_observation = (
                torch.from_numpy(next_observation).unsqueeze(0).float().to(self.device)
            )

        if isinstance(action, np.int64):
            action = (
                torch.from_numpy(np.array(action))
                .to(torch.int8)
                .to(self.device)
                .unsqueeze(0)
            )
        elif isinstance(action, np.ndarray):
            action = (
                torch.from_numpy(action).to(torch.int8).to(self.device).unsqueeze(0)
            )
        elif isinstance(action, torch.Tensor) and action.shape[-1] == 1:
            action = action.squeeze(-1)
        # print("action : ", action.shape)
        action_ohot = F.one_hot(action.long(), num_classes=self.actions_dim).float()

        # print("action_ohot : ", action_ohot.shape)

        if is_sequence:
            batch_size, sequence_size = observation.shape[:2]
            observation = observation.contiguous().view(
                batch_size * sequence_size, *observation.shape[2:]
            )
            next_observation = next_observation.contiguous().view(
                batch_size * sequence_size, *next_observation.shape[2:]
            )
            action_ohot = action_ohot.view(-1, self.actions_dim)

        # print("OBSERVATION : ", observation.shape)
        state = self.encoder(observation)
        next_state = self.encoder(next_observation)
        # print("STATE : ", state.shape)
        # print("action_ohot : ", action_ohot.shape)
        next_state_pred = self.forward_net(torch.cat((state, action_ohot), 1))
        action_pred = self.inverse_net(torch.cat((state, next_state), 1))

        forward_loss = self.forward_criterion(
            next_state_pred, next_state, reduction="none"
        )
        inverse_loss = self.inverse_criterion(
            action_pred, action_ohot, reduction="none"
        )

        # print("forward / inverse icm losses : ", forward_loss.shape, inverse_loss.shape)

        if is_sequence:
            intrinsic_loss = (
                (
                    self.forward_loss_weight * forward_loss
                    + (1 - self.forward_loss_weight) * inverse_loss.unsqueeze(-1)
                )
                .mean(-1)
                .reshape(batch_size, sequence_size)
            )
            intrinsic_reward = (
                forward_loss.clone()
                .detach()
                .mean(-1)
                .reshape(batch_size, sequence_size, 1)
            )
        else:
            intrinsic_loss = (
                self.forward_loss_weight * forward_loss.mean()
                + (1 - self.forward_loss_weight) * inverse_loss.mean()
            )
            intrinsic_reward = forward_loss.clone().detach().mean(-1).reshape(-1, 1)

        return intrinsic_loss, intrinsic_reward
