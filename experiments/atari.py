import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.representations.vision.atari_cnn import AtariCnn
from une.agents.dqn import DQNAgent
from experiments.utils import make_atari_env, train


env_name = "Pong-v4"
env = make_atari_env(env_name)
print(env.observation_space.shape)

config = {
    "name": f"DQN_PER_{env_name}",
    "features_dim": 512,
    "target_update_interval_steps": 2e3,
    "train_freq": 4,
    "save_freq": 1e5,
    "exploration_decay_eps_max_steps": 1e6,
    "learning_rate": 2.5e-4,
    "gradient_steps": 4,
    "tau": 5e-3,
    "soft_update": False,
    "buffer_size": int(1e5),
    "n_step": 1,
    "use_gpu": False,
    "memory_buffer_type": "uniform"
}

wandb.init(project=env_name, config=config)

agent = DQNAgent(
    representation_module_cls=AtariCnn,
    observation_shape=env.observation_space.shape,
    observation_dtype=env.observation_space.dtype,
    n_actions=env.action_space.n,
    exploration_initial_eps=1,
    exploration_final_eps=0.025,
    **config,
)

wandb.watch(agent.algo.q_net, agent.algo.criterion, log="all")

train(
    agent=agent,
    env=env,
    max_global_steps=1e7,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
