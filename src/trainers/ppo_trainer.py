import numpy as np
from tqdm import tqdm
import torch

from src.agents.ppo_agent import PPOAgent
from src.constants import CHECKPOINT_AMOUNT, EPISODES, STEP_AMOUNT
from src.environment import create_mario_env


def run(training_mode, pretrained, num_episodes=EPISODES):
    env = create_mario_env()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = PPOAgent(state_space, action_space)
    total_reward = []
    episodic_reward = []
    for ep_num in tqdm(range(num_episodes)):
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
            state, rewards[step], dones[step], _ = env.step(actions[step])
            state = state.__array__()
            env.render()
            episodic_reward.append(rewards[step])
            if dones[step]:
                total_reward.append(np.sum(episodic_reward))
                episodic_reward = []
                env.reset()
        #adds the an extra value so you can calculate advantages with
        # rewards[i] + self.gamma * values[i + 1] * mask - values[i]
        _, last_value, _ = agent.act(state)
        values = np.append(values, last_value)
        agent.train(to_tensor(states), to_tensor(actions), to_tensor(rewards), to_tensor(dones), to_tensor(prev_log_probs), to_tensor(values))
        
        if ep_num % CHECKPOINT_AMOUNT == 0:
            agent.save()


def to_tensor(list):
    list = torch.tensor(list, device="cuda" if torch.cuda.is_available() else "cpu")
    return list
    
