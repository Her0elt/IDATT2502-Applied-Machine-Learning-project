import random

import numpy as np

from constants import (
    BATCH_SIZE,
    COPY_STEPS,
    EPSILON,
    EPSILON_DECAY_RATE,
    GAMMA,
    MIN_EPSILON,
    SKIP_AND_STACK_AMOUNT,
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
    ):
        self.state_size = input_shape
        self.env = env
        self.action_size = action_shape
        self.memory = ReplyBuffer()
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = epsilon
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.model = DQN(input_shape, action_shape)
        self.target_model = DQN(input_shape, action_shape)

    def action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def preprocess_state(self, state):
        return np.array(np.reshape(state, (1,)+ self.state_size), dtype=np.float32)

    def update_q_func(self, reward, next_state, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.max(next_state)

    def update_q_values(
        self, action_sample, reward_sample, done_sample, target, target_next
    ):
        for i in range(self.batch_size):
            target[i][action_sample[i]] = self.update_q_func(
                reward_sample[i], target_next[i], done_sample[i]
            )

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)
    
    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        
        if len(self.memory) < self.batch_size:
            return

        indices = np.random.choice(range(len(self.memory)), size=self.batch_size)
        states = np.array([self.memory.state[i][0] for i in indices])
        action_sample = np.array([self.memory.action[i] for i in indices])
        reward_sample = np.array([self.memory.reward[i] for i in indices])
        next_states = np.array([self.memory.next_state[i][0] for i in indices])
        done_sample = np.array([self.memory.done[i] for i in indices])
        target = self.model.predict(states)
        target_next = self.target_model.predict(next_states)
        self.update_q_values(
            action_sample, reward_sample, done_sample, target, target_next
        )
        self.model.fit(
            np.array(states), np.array(target), batch_size=self.batch_size, verbose=0
        )
        self.update_epsilon()
