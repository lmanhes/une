import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.representations.vision.atari_cnn import AtariCnn
from une.agents.dqn import DQNAgent
from experiments.utils import make_gym_env, train


resume = False
run_id = "2vwo3tib"

env_name = "BreakoutNoFrameskip-v4"
env = make_gym_env(env_name, atari=True, video=True)
print(env.observation_space.shape)

config = {
    "name": f"DQN_{env_name}",
    "features_dim": 512,
    "target_update_interval_steps": 1e3,
    "train_freq": 1,
    "warmup": 2e4,
    "save_freq": 1e5,
    "exploration_decay_eps_max_steps": 1e6,
    "learning_rate": 1e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(5e4),
    "n_step": 3,
    "use_gpu": True,
    "memory_buffer_type": "per",
}

if resume:
    wandb.init(id=run_id, project=env_name, resume=True)
    agent = DQNAgent(
        representation_module_cls=AtariCnn,
        observation_shape=env.observation_space.shape,
        observation_dtype=env.observation_space.dtype,
        n_actions=env.action_space.n,
        exploration_initial_eps=1,
        exploration_final_eps=0.025,
        **config,
    ).load(path=f"artifacts/{config['name']}.pt")
else:
    wandb.init(project=env_name, config=config, resume=resume)
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
    env_name=env_name,
    max_global_steps=1e7,
    max_episode_steps=3000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
