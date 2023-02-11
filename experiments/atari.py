import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.representations.vision.atari_cnn import AtariCnn
from une.agents.dqn import DQNAgent
from experiments.utils import make_gym_env, train, seed_agent


resume = False
run_id = ""

env_name = "PrivateEyeNoFrameskip-v4"
seed = 42
env = make_gym_env(env_name, atari=True, video=True, seed=seed, n_frame_stack=4)
print(env.observation_space.shape)

config = {
    "name": f"{env_name}",
    "features_dim": 512,
    "target_update_interval_steps": 1e3,
    "train_freq": 4,
    "warmup": 2e4,
    "save_freq": 1e4,
    "exploration_decay_eps_max_steps": 1e6,
    "learning_rate": 1e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(1e5),
    "n_step": 3,
    "use_gpu": False,
    "memory_buffer_type": "per",
    "exploration": "noisy",
    "curiosity": "ngu",
    "intrinsic_reward_weight": 0.5,
    "icm_features_dim": 256,
    "icm_forward_loss_weight": 0.5,
    "ecm_memory_size": 3000,
    "ecm_k": 10
}

seed_agent(seed=seed)

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

wandb.watch(agent.algo.networks, log="all")

train(
    agent=agent,
    env=env,
    env_name=env_name,
    global_steps=agent.steps if resume else 0,
    n_episodes=agent.episodes if resume else 0,
    max_global_steps=1e7,
    max_episode_steps=10000,
    eval_every_n_episodes=1,
)

env.close()
wandb.finish()
