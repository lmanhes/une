import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from une.memories.buffer.sequence import SequenceUniformBuffer
from une.representations.vision.atari_cnn import AtariCnn
from une.algos.r2d1 import R2D1
from experiments.utils import make_gym_env

env_name = "PongNoFrameskip-v4"
seed = 42
env = make_gym_env(env_name, atari=True, video=False, seed=seed, n_frame_stack=4)
print(env.observation_space.shape)

# sequence_buffer = SequenceUniformBuffer(
#     buffer_size=100000,
#     observation_shape=env.observation_space.shape,
#     observation_dtype=env.observation_space.dtype,
#     sequence_length=80,
#     burn_in=40,
#     over_lapping=20,
#     recurrent_dim=512,
# )

# network = RecurrentQNetwork(
#     representation_module_cls=AtariCnn,
#     observation_shape=env.observation_space.shape,
#     features_dim=512,
#     action_dim=env.action_space.n,
#     recurrent_dim=512,
# )

r2d1 = R2D1(
    representation_module_cls=AtariCnn,
    observation_shape=env.observation_space.shape,
    observation_dtype=env.observation_space.dtype,
    features_dim=512,
    n_actions=env.action_space.n,
    recurrent_dim=512,
    memory_buffer_cls=SequenceUniformBuffer,
    buffer_size=100000,
    sequence_length=5,
    burn_in=5,
    over_lapping=2,
)

for episode in range(5):
    observation, info = env.reset(seed=seed)
    done = False
    episode_step = 0
    while not done:
        episode_step += 1
        # network_out = network(
        #     observation=torch.from_numpy(np.array(observation)).unsqueeze(0).float(),
        #     last_action=torch.from_numpy(np.array(env.action_space.sample()).reshape(-1, 1)),
        #     last_reward=torch.from_numpy(np.array(0).reshape(-1, 1)),
        # )

        action = r2d1.act(observation=observation, steps=1)
        #print("action : ", action)

        #action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        r2d1.memorize(observation=next_observation, reward=reward, done=done)

        observation = next_observation
    
    print(f"\nEPISODE {episode} -- {episode_step} steps")


print(len(r2d1.memory_buffer), len(r2d1.memory_buffer.sequence_tracker))
batch = r2d1.memory_buffer.sample(batch_size=20, to_tensor=True)
print(batch.observation.shape, batch.action.shape, batch.h_recurrent.shape)
# r2d1.learn(steps=1)
print(np.array_equal(batch.h_recurrent[12], batch.h_recurrent[17]))

#batch_q_values = r2d1.compute_q_values(batch)
#print("BATCH Q VALUES : ", batch_q_values.shape)

print(r2d1.memory_buffer.sequence_tracker.sequences[0])
print(r2d1.memory_buffer.sequence_tracker.sequences[1])
print(r2d1.memory_buffer.sequence_tracker.sequences[2])