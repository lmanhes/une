import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.agents.dqn import DQNAgent
from une.representations.tabular.mlp import GymMlp
from experiments.utils import train, make_gym_env, seed_agent

seed = 42
env_name = "MountainCar-v0"
env = make_gym_env(env_name=env_name, atari=False, video=True, seed=seed)


config = {
    "name": f"DQN_{env_name}",
    "features_dim": 64,
    "target_update_interval_steps": 1e2,
    "train_freq": 1,
    "save_freq": 5e4,
    "warmup": 5e2,
    "exploration_decay_eps_max_steps": 5e3,
    "learning_rate": 5e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(1e4),
    "n_step": 3,
    "use_gpu": False,
    "memory_buffer_type": 'per',
    "exploration": 'noisy',
    "curiosity": 'icm',
    "intrinsic_reward_weight": 0.5,
    "icm_features_dim": 64,
    "icm_forward_loss_weight": 0.5,
    "ecm_memory_size": 300,
    "ecm_k": 10
}

seed_agent(seed=seed)

wandb.init(project=env_name, config=config)
agent = DQNAgent(
    representation_module_cls=GymMlp,
    observation_shape=env.observation_space.shape,
    observation_dtype=env.observation_space.dtype,
    n_actions=env.action_space.n,
    exploration_initial_eps=1,
    exploration_final_eps=0.025,
    **config
)

wandb.watch(agent.algo.networks, log="all")

train(
    agent=agent,
    env=env,
    env_name=env_name,
    max_global_steps=5e4,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
