import sys, os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wandb

from une.agent import Agent
from une.representations.vision.minigrid_cnn import MinigridCnn
from une.representations.tabular.mlp import GymMlp
from experiments.utils import make_gym_env, seed_agent, train

seed = 42
resume = False
run_id = ""

env_name = "MiniGrid-FourRooms-v0"
env = make_gym_env(env_name=env_name, minigrid=True, flat=True, video=True, seed=seed)
print(env.observation_space, env.observation_space.dtype)

config = {
    "name": f"DQN_{env_name}",
    "features_dim": 64,
    "target_update_interval_steps": 1e2,
    "train_freq": 1,
    "save_freq": 5e4,
    "warmup": 0,
    "gamma": 0.99,
    "max_grad_norm": 5,
    "exploration_decay_eps_max_steps": 5e3,
    "learning_rate":1e-4,
    "gradient_steps": 1,
    "tau": 5e-3,
    "soft_update": True,
    "buffer_size": int(2e5),
    "n_step": 5,
    "use_gpu": True,
    "memory_buffer_type": 'per',
    "exploration": 'noisy',
    "curiosity": "ngu",
    "intrinsic_reward_weight": 0.01,
    "icm_features_dim": 64,
    "icm_forward_loss_weight": 0.5,
    "ecm_memory_size": 300,
    "ecm_k": 10,
    "recurrent": True,
    "sequence_length": 10,
    "burn_in": 0,
    "over_lapping": 5,
    "recurrent_dim": 64,
    "recurrent_init_strategy": "first",
    "per_alpha": 0.7,
    "per_beta": 0.4,
    "batch_size": 32
}
seed_agent(seed=seed)

if resume:
    run = wandb.init(id=run_id, project=env_name, resume=True)
    agent = Agent.load(path=f"{config['name']}.pt")
else:
    run = wandb.init(project=env_name, config=config)
    agent = Agent(
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
