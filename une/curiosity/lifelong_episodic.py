import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from une.curiosity.episodic import EpisodicCuriosityModule
from une.curiosity.icm import IntrinsicCuriosityModule
from une.memories.utils.running_stats import RunningStats


class LifeLongEpisodicModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_dim: int,
        actions_dim: int,
        encoder: nn.Module,
        episodic_memory_size: int = 300,
        k: int = 10,
        forward_loss_weight: float = 0.2,
    ):
        super(LifeLongEpisodicModule, self).__init__()
        self.encoder = encoder
        self.forward_loss_weight = forward_loss_weight

        self.ecm = EpisodicCuriosityModule(
            encoder=self.encoder, memory_size=episodic_memory_size, k=k
        )

        self.icm = IntrinsicCuriosityModule(
            input_dim=input_dim,
            features_dim=features_dim,
            actions_dim=actions_dim,
            encoder=self.encoder,
            forward_loss_weight=self.forward_loss_weight
        )

        self.lifelong_reward_running_stats = RunningStats()

    def get_episodic_reward(self, observation: torch.Tensor):
        return self.ecm.get_episodic_reward(observation=observation)

    def forward(
        self,
        observation: torch.Tensor,
        next_observation: torch.Tensor,
        action: torch.Tensor,
        episodic_reward: torch.Tensor,
    ):
        icm_loss, icm_reward = self.icm(
            observation=observation, next_observation=next_observation, action=action
        )

        lifelong_reward = icm_reward.numpy()
        self.lifelong_reward_running_stats += lifelong_reward
        norm_lifelong_reward = 1 + (
            lifelong_reward - self.lifelong_reward_running_stats.mean
        ) / (self.lifelong_reward_running_stats.std + 1e-8)
        norm_lifelong_reward = norm_lifelong_reward
        
        episodic_reward = episodic_reward.numpy().flatten()
    
        total_reward = torch.from_numpy(episodic_reward * np.minimum(np.maximum(norm_lifelong_reward, 1.0), 5.0)).float().mean(-1).reshape(-1, 1)

        return icm_loss, total_reward

    def reset(self):
        self.ecm.reset()
