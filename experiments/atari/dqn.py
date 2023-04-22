import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.representations.vision.atari_cnn import AtariCnn
from une.agent import Agent
from experiments.utils import make_gym_env, train, seed_agent

seed = 42
env_name = "PongNoFrameskip-v4"
env = make_gym_env(env_name=env_name, atari=False, video=True, seed=seed)

config = {
    "name": f"DQN_{env_name}",
    "features_dim": 64,
    "target_update_interval_steps": 1e2,
    "train_freq": 1,
    "save_freq": 5e4,
    "warmup": 0,
    "gamma": 0.99,
    "max_grad_norm": 10,
    "exploration_decay_eps_max_steps": 5e3,
    "learning_rate":1e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(2e5),
    "n_step": 3,
    "use_gpu": False,
    "memory_buffer_type": 'per',
    "exploration": 'noisy',
    "intrinsic_reward_weight": 0.01,
    "icm_features_dim": 64,
    "icm_forward_loss_weight": 0.5,
    "recurrent": False,
    "per_alpha": 0.7,
    "per_beta": 0.4,
    "batch_size": 32
}
seed_agent(seed=seed)

wandb.init(project=env_name, config=config)
agent = Agent(
    representation_module_cls=AtariCnn,
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
    max_global_steps=1e7,
    max_episode_steps=10000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
