import random
import time

import gymnasium as gym
from gymnasium.wrappers.record_video import capped_cubic_video_schedule
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from loguru import logger
import numpy as np
import torch
import wandb

from une.agents.abstract import AbstractAgent


def make_gym_env(
    env_name: str,
    seed: int = 42,
    atari: bool = False,
    video: bool = False,
    n_frame_stack: int = 1,
):
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video:
        env = gym.wrappers.RecordVideo(env, f"videos/{env_name}", disable_logger=True)

    if atari:
        env = AtariPreprocessing(env)

    if n_frame_stack > 1:
        env = FrameStack(env, num_stack=n_frame_stack)

    seed_env(env=env, seed=seed)
    return env


def seed_env(env: gym.Env, seed: int = 42):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def seed_agent(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval(
    agent: AbstractAgent,
    env: gym.Env,
    global_steps: int,
    seed: int,
    max_episode_steps: int = 3000,
    logs: bool = False,
):
    observation, info = env.reset(seed=seed)
    done = False
    episode_steps = 0
    episode_reward = 0
    start = time.time()
    agent.reset()
    
    while not done:
        episode_steps += 1
        action = agent.act(observation=observation, evaluate=True)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated) or (episode_steps > max_episode_steps)

        episode_reward += reward

        observation = next_observation

    end = time.time()
    speed = round(episode_steps / (end - start), 3)
    if logs:
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
    env_name: str,
    global_steps: int = 0,
    n_episodes: int = 0,
    max_global_steps: int = 1e7,
    max_episode_steps: int = 3000,
    eval_every_n_episodes: int = 10,
    seed: int = 42,
    logs: bool = True,
):
    said_full = False
    while global_steps < max_global_steps:
        observation, info = env.reset(seed=seed)
        done = False
        episode_steps = 0
        episode_reward = 0
        start = time.time()
        agent.reset()

        while not done:
            episode_steps += 1
            global_steps += 1
            action = agent.act(observation=observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = (terminated or truncated) or (episode_steps > max_episode_steps)

            episode_reward += reward
            
            agent.memorize(observation=next_observation, reward=reward, done=done)

            if done:
                agent.episodes += 1
                n_episodes += 1

            observation = next_observation
            if not said_full and agent.algo.memory_buffer.full:
                logger.warning(f"Memory is full at {global_steps} steps")
                said_full = True

        end = time.time()
        speed = round(episode_steps / (end - start), 3)
        if logs:
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

        if n_episodes % eval_every_n_episodes == 0:
            eval(
                agent=agent,
                env=env,
                global_steps=global_steps,
                max_episode_steps=max_episode_steps,
                seed=seed,
                logs=logs,
            )

        # log gameplay video in wandb
        if logs:
            if capped_cubic_video_schedule(episode_id=n_episodes):
                mp4 = f"videos/{env_name}/rl-video-episode-{n_episodes}.mp4"
                wandb.log(
                    {"gameplays": wandb.Video(mp4)},
                    step=global_steps,
                )
