import pickle
import random

import numpy as np
import torch
from torch import nn

from constants import (
    BATCH_SIZE,
    COPY_STEPS,
    EPSILON,
    EPSILON_DECAY_RATE,
    GAMMA,
    LEARNING_RATE,
    MEMORY_SIZE,
    MIN_EPSILON,
)
from model import DQN
from replay_buffer import ReplyBuffer


class DoubleDQNAgent:
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
        copy=COPY_STEPS,
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
        self.target_model = DQN(input_shape, action_shape).to(self.device)
        self.step = 0
        self.copy = copy
        self.ending_position = 0
        self.num_in_queue = 0
        self.loss_func = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def load(self):
        self.model = self.model.load()
        self.target_model = self.target_model.save(target=True)
        self.memory.load()
        with open("ending_position.pkl", "rb") as f:
            self.ending_position = pickle.load(f)
        with open("num_in_queue.pkl", "rb") as f:
            self.num_in_queue = pickle.load(f)

    def save(self, total_rewards):
        self.model.save()
        self.target_model.save(target=True)
        self.memory.save()
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(self.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(self.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

    def act(self, state):
        # Epsilon-greedy action

        self.step += 1
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
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
        # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
        return reward + torch.mul(
            (self.gamma * self.target_model(next_state).max(1).values.unsqueeze(1)),
            1 - done,
        )

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def copy_weights(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):

        if self.step % self.copy == 0:
            self.copy_weights()

        if self.memory_size > self.num_in_queue:
            return

        state, action, reward, next_sates, done = self.memory.recall(
            self.num_in_queue, self.batch_size, self.device
        )
        target = self.update_q_values(reward, done, next_sates)
        current = self.model(state).gather(1, action.long())
        loss = self.loss_func(current, target)
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()