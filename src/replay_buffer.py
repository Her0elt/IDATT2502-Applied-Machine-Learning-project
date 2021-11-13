import random

import torch

from src.constants import (
    ACTION_SAVE_NAME,
    DONE_SAVE_NAME,
    NEXT_STATE_SAVE_NAME,
    REWARD_SAVE_NAME,
    STATE_SAVE_NAME,
)


class ReplyBuffer:
    def __init__(self, state_space, memory_size):
        self.state_space = state_space
        self.sate_mem = torch.zeros(memory_size, *self.state_space)
        self.action_mem = torch.zeros(memory_size, 1)
        self.reward_mem = torch.zeros(memory_size, 1)
        self.next_state_mem = torch.zeros(memory_size, *self.state_space)
        self.done_mem = torch.zeros(memory_size, 1)

    def append(self, state, action, reward, next_state, done, end_pos):
        self.sate_mem[end_pos] = state.float()
        self.action_mem[end_pos] = action.float()
        self.reward_mem[end_pos] = reward.float()
        self.next_state_mem[end_pos] = next_state.float()
        self.done_mem[end_pos] = done.float()

    def recall(self, num_in_queue, memory_sample_size, device):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(num_in_queue), k=memory_sample_size)

        state = self.sate_mem[idx].to(device)
        action = self.action_mem[idx].to(device)
        reward = self.reward_mem[idx].to(device)
        next_state = self.next_state_mem[idx].to(device)
        done = self.done_mem[idx].to(device)

        return state, action, reward, next_state, done

    def save(self):
        torch.save(self.sate_mem, STATE_SAVE_NAME)
        torch.save(self.action_mem, ACTION_SAVE_NAME)
        torch.save(self.reward_mem, REWARD_SAVE_NAME)
        torch.save(self.next_state_mem, NEXT_STATE_SAVE_NAME)
        torch.save(self.done_mem, DONE_SAVE_NAME)

    def load(self):
        self.sate_mem = torch.load(STATE_SAVE_NAME)
        self.action_mem = torch.load(ACTION_SAVE_NAME)
        self.reward_mem = torch.load(REWARD_SAVE_NAME)
        self.next_state_mem = torch.load(NEXT_STATE_SAVE_NAME)
        self.done_mem = torch.load(DONE_SAVE_NAME)
