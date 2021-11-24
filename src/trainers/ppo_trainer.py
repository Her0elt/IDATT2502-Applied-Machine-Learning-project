import time

import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.agents.ppo_agent import PPOAgent
from src.constants import (
    ACTOR_LEARNING_RATE,
    CHECKPOINT_AMOUNT,
    CLIP_RANGE,
    CRITIC_LEARNING_RATE,
    EPISODES,
    MIN_WANDB_VIDEO_REWARD,
    OPTIMIZER_EPSILON,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_LAMBDA,
    STEP_AMOUNT,
    UPDATE_FREQUENCY,
    WANDB_ENTITY,
    WANDB_PPO_PROJECT,
)
from src.environment import create_mario_env


def run(pretrained, num_episodes=EPISODES, wandb_name=None):

    should_log = bool(wandb_name)

    if should_log:
        wandb.init(
            project=WANDB_PPO_PROJECT,
            name=wandb_name,
            entity=WANDB_ENTITY,
            config={
                "ACTOR_LEARNING_RATE": ACTOR_LEARNING_RATE,
                "CLIP_RANGE": CLIP_RANGE,
                "CRITIC_LEARNING_RATE": CRITIC_LEARNING_RATE,
                "OPTIMIZER_EPSILON": OPTIMIZER_EPSILON,
                "STEP_AMOUNT": STEP_AMOUNT,
                "UPDATE_FREQUENCY": UPDATE_FREQUENCY,
                "EPISODES": num_episodes,
                "PPO_GAMMA": PPO_GAMMA,
                "PPO_LAMBDA": PPO_LAMBDA,
                "PPO_EPOCHS": PPO_EPOCHS,
            },
        )

    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = PPOAgent(state_space, action_space)
    flags = []
    if pretrained:
        agent.load()
    episodic_reward = []
    max_episode_reward = 0
    play_episode = 1

    for ep_num in tqdm(range(num_episodes)):
        total_reward = []
        frames = []
        states = np.zeros((STEP_AMOUNT, 4, 84, 84), dtype=np.float32)
        actions = np.zeros(STEP_AMOUNT, dtype=np.int32)
        rewards = np.zeros(STEP_AMOUNT, dtype=np.float32)
        dones = np.zeros(STEP_AMOUNT, dtype=bool)
        prev_log_probs = np.zeros(STEP_AMOUNT, dtype=np.float32)
        values = np.zeros(STEP_AMOUNT, dtype=np.float32)
        state = env.reset()
        for step in range(STEP_AMOUNT):
            states[step] = state
            actions[step], values[step], prev_log_probs[step] = agent.act(state)
            state, rewards[step], dones[step], info = env.step(actions[step])
            frames.append(env.frame)
            episodic_reward.append(rewards[step])
            if dones[step]:
                flags.append(int(info["flag_get"]))
                total_episode_reward = np.sum(episodic_reward)
                total_reward.append(total_episode_reward)
                if total_episode_reward > max_episode_reward:
                    max_episode_reward = total_episode_reward
                    if should_log and total_episode_reward > MIN_WANDB_VIDEO_REWARD:
                        try:

                            wandb.log(
                                {
                                    "video": wandb.Video(
                                        np.stack(frames, 0).transpose(0, 3, 1, 2),
                                        str(total_episode_reward),
                                        fps=25,
                                        format="mp4",
                                    )
                                }
                            )
                        except Exception:
                            tqdm.write("something happend while logging")

                if should_log:
                    wandb.log(
                        {
                            "mean_last_10_episodes": np.mean(total_reward[-10:]),
                            "episode_reward": np.sum(total_episode_reward),
                            "nr_found_flag_last_10": np.sum(flags[-10:]),
                        },
                        step=play_episode,
                    )

                episodic_reward = []
                play_episode += 1
                frames = []
                env.reset()

        # Adds the an extra value so you can calculate advantages with
        # rewards[i] + self.gamma * values[i + 1] * mask - values[i]
        _, last_value, _ = agent.act(state)
        values = np.append(values, last_value)
        states = torch.tensor(states.reshape(states.shape[0], *states.shape[1:]))
        agent.train(
            to_tensor(states),
            to_tensor(actions),
            to_tensor(rewards),
            to_tensor(dones),
            to_tensor(prev_log_probs),
            to_tensor(values),
        )

        if ep_num % CHECKPOINT_AMOUNT == 0:
            agent.save()

        tqdm.write(
            "Mean total reward after episode {} is {}".format(
                ep_num, np.mean(total_reward)
            )
        )


def to_tensor(list):
    list = torch.tensor(list, device="cuda" if torch.cuda.is_available() else "cpu")
    return list


def play():
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = PPOAgent(state_space, action_space)
    agent.load()
    state = env.reset()
    while True:
        action = agent.play(state)
        state_next, _, done, _ = env.step(int(action[0]))
        env.render()
        time.sleep(0.05)
        state = state_next
        if done:
            break
    env.close()
