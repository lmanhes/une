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
        sequence: bool = False,
    ):
        super(IntrinsicCuriosityModule, self).__init__()
        self.encoder = encoder
        self.forward_loss_weight = forward_loss_weight
        self.sequence = sequence

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
        observation: torch.Tensor,
        next_observation: torch.Tensor,
        action: torch.Tensor,
    ):
        action_ohot = (
            F.one_hot(action.long(), num_classes=self.actions_dim).squeeze(1).float()
        )

        if self.sequence:
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
        intrinsic_loss = (
            self.forward_loss_weight * forward_loss.mean()
            + (1 - self.forward_loss_weight) * inverse_loss.mean()
        )

        if self.sequence:
            intrinsic_reward = (
                forward_loss.clone()
                .detach()
                .mean(-1)
                .reshape(batch_size, sequence_size, 1)
            )
        else:
            intrinsic_reward = forward_loss.clone().detach().mean(-1).reshape(-1, 1)

        return intrinsic_loss, intrinsic_reward
