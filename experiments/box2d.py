import sys, os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.agents.dqn import DQNAgent
from une.representations.tabular.mlp import GymMlp
from experiments.utils import train, make_gym_env, seed_agent


seed = 42
resume = False
run_id = ""

env_name = "LunarLander-v2"
env = make_gym_env(env_name=env_name, atari=False, video=True, seed=seed)

config = {
    "name": f"{env_name}",
    "features_dim": 64,
    "target_update_interval_steps": 1e3,
    "train_freq": 4,
    "save_freq": 1e4,
    "exploration_decay_eps_max_steps": 3e4,
    "learning_rate": 5e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(1e5),
    "n_step": 3,
    "use_gpu": False,
    "memory_buffer_type": "per",
    "exploration": "noisy",
    "curiosity": "icm",
    "icm_features_dim": 128,
    "icm_alpha": 0.1,
    "icm_beta": 0.2
}

seed_agent(seed=seed)

if resume:
    run = wandb.init(id=run_id, project=env_name, resume=True)
    agent = DQNAgent.load(path=f"{config['name']}.pt")
else:
    run = wandb.init(project=env_name, config=config)
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
    global_steps=agent.steps if resume else 0,
    n_episodes=agent.episodes if resume else 0,
    max_global_steps=3e5,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
    logs=True,
)

env.close()
wandb.finish()
