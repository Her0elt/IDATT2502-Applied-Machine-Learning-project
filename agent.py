from replay_buffer import ReplyBuffer
import numpy as np

import random
from model import DQN


class DQNAgent:
    def __init__(self, env,  input_shape, action_shape, epsilon_decay_rate=0.9999, min_epsilon=0.01):
        self.state_size = input_shape
        self.env = env
        self.action_size = action_shape
        self.memory = ReplyBuffer()
        self.min_epsilon=min_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = 1
        self._current_score = 0
        self.batch_size = 32
        self.gamma = 0.99
        self.model = DQN(input_shape, action_shape)


    def action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)
    

    def preprocess_state(self, state):
        return np.reshape(state, (1,) + self.state_size) 
  
    def update_q_func(self,reward, next_state, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.max(next_state)

    def update_q_values(self, minibatch, target, target_next ):
        for index, (_, action, reward, _, done) in enumerate(minibatch):
            target[index][action] = self.update_q_func(reward, target_next[index], done)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.min_epsilon, self.epsilon)


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
        target_next = self.model.predict(next_states)
        for i in range(self.batch_size):
            if done_sample[i]:
                target[i][action_sample[i]] = reward_sample[i]
            else:
                target[i][action_sample[i]] = reward_sample[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(np.array(states), np.array(target), batch_size=self.batch_size, verbose=0)
        self.update_epsilon()