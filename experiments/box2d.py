import sys, os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym

from une.agents.dqn import DQNAgent, Transition
from une.representations.tabular.mlp import GymMlp


env = gym.make("LunarLander-v2")


observation = env.reset()
print(observation.shape)

agent = DQNAgent(
    representation_module_cls=GymMlp,
    observation_shape=observation.shape,
    features_dim=64,
    n_actions=env.action_space.n,
    target_update_interval_steps=1e4,
    train_freq=4,
    gradient_steps=1,
    exploration_initial_eps=1,
    exploration_final_eps=0.025,
    exploration_decay_eps_max_steps=1e5,
    buffer_size=int(1e5),
    learning_rate=1e-4
)

total_steps = 0
for episode in range(3000):

    print(f"\nEpisode {episode+1} -- total steps {total_steps}")

    # Training
    observation = env.reset()
    done = False
    steps = 0
    episode_reward = 0
    start = time.time()

    while not done and steps < 3000:
        steps += 1
        total_steps += 1
        action = agent.act(observation)
        #print("training action : ", action)
        next_observation, reward, done, info = env.step(action)
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
    print(f"Training: steps {steps} : episode {episode} -- {episode_reward} rewards -- {speed} fps -- (epsilon {agent.epsilon})")


    if episode % 100 == 0:

        # Evaluation
        observation = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        start = time.time()

        while not done and steps < 3000:
            steps += 1
            action = agent.act(observation, evaluate=True)
            next_observation, reward, done, info = env.step(action)
            episode_reward += reward

            observation = next_observation

        end = time.time()
        speed = round(steps/(end-start), 3)
        print(f"Evaluation: steps {steps} : episode {episode} -- {episode_reward} rewards -- {speed} fps")
env.close()
