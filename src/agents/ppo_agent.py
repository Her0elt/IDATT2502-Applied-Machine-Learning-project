import numpy as np
import torch
from torch import nn

from src.constants import (
    ACTOR_LEARNING_RATE,
    CLIP_RANGE,
    CRITIC_LEARNING_RATE,
    OPTIMIZER_EPSILON,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_LAMBDA,
    STEP_AMOUNT,
    UPDATE_FREQUENCY,
)
from src.models.ppo import PPO


class PPOAgent:
    def __init__(self, state_shape, action_shape):
        self.gamma = PPO_GAMMA
        self.lamda = PPO_LAMBDA
        self.epochs = PPO_EPOCHS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = PPO(state_shape, action_shape).to(self.device)
        self.policy_old = PPO(state_shape, action_shape).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss_func = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": ACTOR_LEARNING_RATE},
                {"params": self.policy.critic.parameters(), "lr": CRITIC_LEARNING_RATE},
            ],
            eps=OPTIMIZER_EPSILON,
        )

    def save(self):
        self.policy.save()

    def load(self):
        self.policy.load(device=self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def play(self, state):
        pi, _ = self.policy(
            torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        )
        action = pi.sample()
        return action

    def act(self, state):
        with torch.no_grad():
            pi, v = self.policy(
                torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
            )
            value = v.cpu().numpy()
            action = pi.sample()
            return action, value, pi.log_prob(action).cpu().numpy()

    def calculate_advantages(self, rewards, dones, values):
        returns = []
        gae = 0
        npValues = values.cpu().numpy()
        for i in reversed(range(len(rewards))):
            mask = 1.0 - int(dones[i])
            delta = (
                rewards.cpu().numpy()[i]
                + self.gamma * npValues[i + 1] * mask
                - npValues[i]
            )
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + npValues[i])
        returns = np.array(returns)
        adv = returns - npValues[:-1]
        return (
            torch.tensor(returns, device=self.device, dtype=torch.float32),
            torch.tensor(
                (adv - np.mean(adv)) / (np.std(adv) + 1e-8),
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def calculate_loss(self, states, actions, prev_log_probs, returns, advantages):
        pi, value = self.policy(states)
        ratio = torch.exp(pi.log_prob(actions) - prev_log_probs)
        clipped_ratio = ratio.clamp(min=1 - CLIP_RANGE, max=1 + CLIP_RANGE)
        policy_reward = torch.min(ratio * advantages, clipped_ratio * advantages)
        entropy_bonus = pi.entropy()
        mse_loss = self.loss_func(value, returns)
        loss = -policy_reward + 0.5 * mse_loss - 0.01 * entropy_bonus
        return loss.mean()

    def train(self, states, actions, rewards, dones, prev_log_probs, values):
        returns, advantages = self.calculate_advantages(rewards, dones, values)
        indexes = torch.randperm(STEP_AMOUNT)
        step_amount = STEP_AMOUNT // UPDATE_FREQUENCY
        for batch_start in range(0, STEP_AMOUNT, step_amount):
            batch_end = batch_start + step_amount
            batch_indexes = indexes[batch_start:batch_end]
            for _ in range(self.epochs):
                loss = self.calculate_loss(
                    states[batch_indexes],
                    actions[batch_indexes],
                    prev_log_probs[batch_indexes],
                    returns[batch_indexes],
                    advantages[batch_indexes],
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())
