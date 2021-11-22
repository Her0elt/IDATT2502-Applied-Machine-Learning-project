import pickle
import random

import numpy as np
import torch
from torch import nn
from src.models.dqn import DQN

from src.constants import (
    BATCH_SIZE,
    ENDING_POSISTION_PICKLE,
    EPSILON,
    EPSILON_DECAY_RATE,
    GAMMA,
    LEARNING_RATE,
    MEMORY_SIZE,
    MIN_EPSILON,
    NUM_IN_QUEUE_PICKLE,
    OPTIMIZER_EPSILON,
    TOTAL_REWARDS_PICKLE,
)
from src.replay_buffer import ReplyBuffer


class DQNAgent:
    def __init__(
        self,
        env,
        input_shape,
        action_shape,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        min_epsilon=MIN_EPSILON,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        memory_size=MEMORY_SIZE,
    ):
        self.state_space = input_shape
        self.env = env
        self.action_space = action_shape
        self.memory_size = memory_size
        self.memory = ReplyBuffer(input_shape, memory_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = DQN(input_shape, action_shape).to(self.device)
        self.step = 0
        self.ending_position = 0
        self.num_in_queue = 0
        self.loss_func = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, eps=OPTIMIZER_EPSILON
        )

    def load(self):
        self.model.load(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory.load()
        self.memory_size = MEMORY_SIZE
        with open(ENDING_POSISTION_PICKLE, "rb") as f:
            self.ending_position = pickle.load(f)
        with open(NUM_IN_QUEUE_PICKLE, "rb") as f:
            self.num_in_queue = pickle.load(f)

    def save(self, total_rewards):
        self.model.save()
        self.memory.save()
        with open(ENDING_POSISTION_PICKLE, "wb") as f:
            pickle.dump(self.ending_position, f)
        with open(NUM_IN_QUEUE_PICKLE, "wb") as f:
            pickle.dump(self.num_in_queue, f)
        with open(TOTAL_REWARDS_PICKLE, "wb") as f:
            pickle.dump(total_rewards, f)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action_values = self.model(
                torch.tensor(state, dtype=torch.float32, device=self.device)
            )
            action = torch.argmax(action_values, dim=1).item()
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)
        self.step += 1
        return action

    def play(self, state):
        return (
            torch.argmax(self.model(state.to(self.device)))
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
        )

    def remember(self, state, action, reward, next_state, done):
        self.ending_position = (self.ending_position + 1) % self.memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.memory_size)
        self.memory.append(
            state, action, reward, next_state, done, self.ending_position
        )

    def update_q_values(self, reward, done, next_state):
        return reward + torch.mul(
            (self.gamma * self.model(next_state).max(1).values.unsqueeze(1)), 1 - done,
        )

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def replay(self):

        if self.batch_size > self.num_in_queue:
            return

        state, action, reward, next_sates, done = self.memory.recall(
            self.num_in_queue, self.batch_size, self.device
        )
        target = self.update_q_values(reward, done, next_sates)
        current = self.model(state).gather(1, action.long())
        loss = self.loss_func(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
