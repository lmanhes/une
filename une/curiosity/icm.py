import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrinsicCuriosityModule(nn.Module):
    def __init__(
        self, input_dim: int, features_dim: int, actions_dim: int, encoder: nn.Module
    ):
        super(IntrinsicCuriosityModule, self).__init__()
        self.encoder = encoder

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
        state = self.encoder(observation)
        next_state = self.encoder(next_observation)
        action_ohot = (
            F.one_hot(action.long(), num_classes=self.actions_dim).squeeze(1).float()
        )
        next_state_pred = self.forward_net(torch.cat((state, action_ohot), 1))
        action_pred = self.inverse_net(torch.cat((state, next_state), 1))

        return (
            self.forward_criterion(next_state_pred, next_state, reduction="none"),
            self.inverse_criterion(action_pred, action_ohot, reduction="none"),
        )
