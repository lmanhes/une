import sys, os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym
import wandb
import torch.nn.functional as F

from une.agents.dqn import DQNAgent
from une.representations.tabular.mlp import GymMlp
from experiments.utils import train

seed = 42
env_name = "CartPole-v1"
env = gym.make(env_name)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

config = {
    "features_dim": 64,
    "target_update_interval_steps": 1e4,
    "train_freq": 1,
    "exploration_decay_eps_max_steps": 5e4,
    "learning_rate": 1e-4,
    "gradient_steps": 1,
    "tau": 1e-3,
    "soft_update": True,
    "use_gpu": False,
    "memory_buffer_type": 'per'
}

wandb.init(project=env_name, config=config)

agent = DQNAgent(
    representation_module_cls=GymMlp,
    observation_shape=env.observation_space.shape,
    n_actions=env.action_space.n,
    exploration_initial_eps=1,
    exploration_final_eps=0.025,
    buffer_size=int(1e4),
    **config
)

wandb.watch(agent.algo.q_net, F.smooth_l1_loss, log="all")

train(
    agent=agent,
    env=env,
    max_global_steps=2e5,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
