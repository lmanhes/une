import sys, os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym
import wandb
import torch.nn.functional as F

from une.agents.dqn import DQNAgent, Transition
from une.representations.tabular.mlp import GymMlp
from experiments.utils import train

env = gym.make("LunarLander-v2")

config = {
    "features_dim": 64,
    "target_update_interval_steps": 1e3,
    "train_freq": 4,
    "exploration_decay_eps_max_steps": 1e5,
    "learning_rate": 2.5e-3,
    "gradient_steps": 4,
    "tau": 1e-3,
    "soft_update": True,
    "use_gpu": False,
}

wandb.init(project="LunarLander-v2", config=config)

agent = DQNAgent(
    representation_module_cls=GymMlp,
    observation_shape=env.observation_space.shape,
    n_actions=env.action_space.n,
    exploration_initial_eps=1,
    exploration_final_eps=0.025,
    buffer_size=int(1e5),
    **config
)

wandb.watch(agent.algo.q_net, F.smooth_l1_loss, log="all")

train(
    agent=agent,
    env=env,
    max_global_steps=1e6,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
