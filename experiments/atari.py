import sys, os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym

from une.agents.dqn import DQNAgent, Transition
from une.representations.vision.atari_cnn import AtariCnn
from une.representations.vision.preprocessing import (
    VisionPreprocessing
)


env = gym.make("Pong-v4", render_mode=None)
env.action_space.seed(42)

preprocessing = VisionPreprocessing(
    to_grayscale=True,
    resize=True,
    new_size=84,
    normalize=True,
    channel_first=False,
    stack=True,
    n_stack=4
)

observation = env.reset()
print(observation.shape)
observation = preprocessing(observation)

agent = DQNAgent(
    representation_module_cls=AtariCnn,
    observation_shape=observation.shape,
    features_dim=512,
    n_actions=env.action_space.n,
    target_update_interval_steps=1000,
    gradient_steps=1,
    n_max_eps_steps=100000,
    buffer_size=int(1e5)
)

for episode in range(1000):

    # Training
    observation = env.reset()
    preprocessing.reset()
    observation = preprocessing(observation)
    done = False
    steps = 0
    episode_reward = 0
    start = time.time()

    while not done:
        steps += 1
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        next_observation = preprocessing(next_observation)
        episode_reward += reward

        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation,
        )
        agent.memorize(transition)
        observation = next_observation

    end = time.time()
    speed = round(steps/(end-start), 3)
    print(f"\nTraining: steps {steps} : episode {episode} -- {episode_reward} rewards -- {speed} fps -- (epsilon {agent.algo.greedy_exploration.epsilon})")


    if episode % 100 == 0:

        # Evaluation
        observation = env.reset()
        preprocessing.reset()
        observation = preprocessing(observation)
        done = False
        episode_reward = 0
        steps = 0
        start = time.time()

        while not done and steps < 3000:
            steps += 1
            action = agent.act(observation, evaluate=True)
            next_observation, reward, done, info = env.step(action)
            next_observation = preprocessing(next_observation)
            episode_reward += reward

            observation = next_observation

        end = time.time()
        speed = round(steps/(end-start), 3)
        print(f"Evaluation: steps {steps} : episode {episode} -- {episode_reward} rewards -- {speed} fps")

env.close()
