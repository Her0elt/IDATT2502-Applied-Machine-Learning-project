import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.constants import PPO_MODEL_SAVE_NAME


class PPO(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPO, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return Categorical(logits=self.actor(conv_out)), self.critic(conv_out).reshape(-1)

    def save(self):
        torch.save(self.state_dict(), PPO_MODEL_SAVE_NAME)

    def load(self, device):
        self.load_state_dict(
            torch.load(PPO_MODEL_SAVE_NAME, map_location=torch.device(device))
        )
