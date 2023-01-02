import sys, os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.agents.dqn import DQNAgent
from une.representations.tabular.mlp import GymMlp
from experiments.utils import train, make_gym_env


env_name = "LunarLander-v2"
env = make_gym_env(env_name=env_name, atari=False, video=True)

config = {
    "name": f"DQN_{env_name}",
    "features_dim": 64,
    "target_update_interval_steps": 1e3,
    "train_freq": 1,
    "save_freq": 5e4,
    "exploration_decay_eps_max_steps": 3e4,
    "learning_rate": 5e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(1e5),
    "n_step": 1,
    "use_gpu": False,
    "memory_buffer_type": "per",
}

wandb.init(project=env_name, config=config)

agent = DQNAgent(
    representation_module_cls=GymMlp,
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
    env_name=env_name,
    max_global_steps=3e5,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
    logs=True
)

env.close()
wandb.finish()
