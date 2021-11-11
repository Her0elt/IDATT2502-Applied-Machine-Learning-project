import torch
from tqdm import tqdm
import time
import numpy as np
from constants import EPISODES
from double_agent import DoubleDQNAgent
from environment import create_mario_env


def run(training_mode, pretrained, num_episodes=EPISODES):

    env = create_mario_env() 
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    if pretrained:
        agent = DoubleDQNAgent(env, state_space, action_space, memory_size=0)
        agent.load()
    else:
        agent = DoubleDQNAgent(env, state_space, action_space)

    total_rewards = []

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor(np.array([state]))
        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            steps += 1

            state_next, reward, done, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor(np.array([state_next]))
            reward = torch.tensor(np.array([reward])).unsqueeze(0)

            done = torch.tensor(np.array([int(done)])).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next, done)
                agent.replay()

            state = state_next
            if done:
                break

        total_rewards.append(total_reward)

        print(
            "Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1])
        )
        num_episodes += 1
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


# run(training_mode=True, pretrained=True)
play()

