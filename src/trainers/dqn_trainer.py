import time

import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.agents.double_agent import DoubleDQNAgent
from src.constants import (
    BATCH_SIZE,
    COPY_STEPS,
    ENDING_POSISTION_PICKLE,
    EPISODES,
    EPSILON,
    EPSILON_DECAY_RATE,
    GAMMA,
    LEARNING_RATE,
    MEMORY_SIZE,
    MIN_EPSILON,
    MIN_WANDB_VIDEO_REWARD,
    NUM_IN_QUEUE_PICKLE,
    TOTAL_REWARDS_PICKLE,
    WANDB_DDQN_PROJECT,
    WANDB_ENTITY,
)
from src.environment import create_mario_env


def run(pretrained, num_episodes=EPISODES, wandb_name=None):

    should_log = bool(wandb_name)

    if should_log:
        wandb.init(
            project=WANDB_DDQN_PROJECT,
            name=wandb_name,
            entity=WANDB_ENTITY,
            config={
                "BATCH_SIZE": BATCH_SIZE,
                "COPY_STEPS": COPY_STEPS,
                "ENDING_POSISTION_PICKLE": ENDING_POSISTION_PICKLE,
                "EPSILON": EPSILON,
                "EPSILON_DECAY_RATE": EPSILON_DECAY_RATE,
                "GAMMA": GAMMA,
                "LEARNING_RATE": LEARNING_RATE,
                "MEMORY_SIZE": MEMORY_SIZE,
                "MIN_EPSILON": MIN_EPSILON,
                "NUM_IN_QUEUE_PICKLE": NUM_IN_QUEUE_PICKLE,
                "TOTAL_REWARDS_PICKLE": TOTAL_REWARDS_PICKLE,
                "EPISODES": num_episodes,
            },
        )

    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DoubleDQNAgent(env, state_space, action_space)
    if pretrained:
        agent.load()

    total_rewards = []
    max_episode_reward = 0

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        frames = []
        state = torch.Tensor(np.array([state]))
        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            steps += 1

            state_next, reward, done, info = env.step(int(action[0]))
            frames.append(env.frame)
            total_reward += reward
            state_next = torch.Tensor(np.array([state_next]))
            reward = torch.tensor(np.array([reward])).unsqueeze(0)

            done = torch.tensor(np.array([int(done)])).unsqueeze(0)

            agent.remember(state, action, reward, state_next, done)
            agent.replay()

            state = state_next
            if done:
                if total_reward > max_episode_reward:
                    max_episode_reward = total_reward
                    if should_log and total_reward > MIN_WANDB_VIDEO_REWARD:
                        wandb.log(
                            {
                                "video": wandb.Video(
                                    np.stack(frames, 0).transpose(0, 3, 1, 2),
                                    str(total_reward),
                                    fps=25,
                                    format="mp4",
                                )
                            }
                        )

                if should_log:
                    wandb.log(
                        {
                            "mean_last_10_episodes": np.mean(total_rewards[-10:]),
                            "episode_reward": np.sum(total_reward),
                            "epsilon": agent.epsilon,
                        },
                        step=ep_num,
                    )
                break

            agent.update_epsilon()

        total_rewards.append(total_reward)

        tqdm.write(
            "Total reward after episode {} is {}".format(ep_num + 1, total_reward)
        )
    agent.save()
    env.close()


def play():
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DoubleDQNAgent(env, state_space, action_space, memory_size=0)
    agent.model.load(agent.device)
    state = env.reset()
    state = torch.Tensor(np.array([state]))

    while True:
        action = agent.play(state)
        state_next, _, done, _ = env.step(int(action[0]))
        env.render()
        time.sleep(0.05)
        state_next = torch.Tensor(np.array([state_next]))
        state = state_next
        if done:
            break
    env.close()
