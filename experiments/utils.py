import time

import numpy as np
import gym
import wandb
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from une.agents.abstract import AbstractAgent
from une.agents.dqn import Transition


def make_atari_env(env_name: str, seed: int = 42):
    env = gym.make(env_name)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def eval(
    agent: AbstractAgent, env: gym.Env, global_steps: int, max_episode_steps: int = 3000
):
    observation = env.reset()
    done = False
    episode_steps = 0
    episode_reward = 0
    start = time.time()

    while not done and episode_steps < max_episode_steps:
        episode_steps += 1
        action = agent.act(observation, evaluate=True)
        next_observation, reward, done, info = env.step(action)
        episode_reward += reward

        observation = next_observation

    end = time.time()
    speed = round(episode_steps / (end - start), 3)
    wandb.log(
        {
            "eval_fps": speed,
            "eval_episode_reward": episode_reward,
            "eval_episode_steps": episode_steps,
        },
        step=global_steps,
    )


def train(
    agent: AbstractAgent,
    env: gym.Env,
    max_global_steps: int = 1e7,
    max_episode_steps: int = 3000,
    eval_every_n_episodes: int = 10,
):
    global_steps = 0
    n_episodes = 0
    while global_steps < max_global_steps:
        observation = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0
        start = time.time()

        while not done and episode_steps < max_episode_steps:
            episode_steps += 1
            global_steps += 1
            action = agent.act(observation)
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

        n_episodes += 1
        end = time.time()
        speed = round(episode_steps / (end - start), 3)
        wandb.log(
            {
                "train_fps": speed,
                "train_epsilon": agent.epsilon,
                "train_episode_reward": episode_reward,
                "train_episode_steps": episode_steps,
                "train_n_episodes": n_episodes,
            },
            step=global_steps,
        )

        # print(f"Training: steps {steps} : episode {episode} -- {episode_reward} rewards -- {speed} fps -- (epsilon {agent.epsilon})")

        if n_episodes % eval_every_n_episodes == 0:
            eval(
                agent=agent,
                env=env,
                global_steps=global_steps,
                max_episode_steps=max_episode_steps,
            )
