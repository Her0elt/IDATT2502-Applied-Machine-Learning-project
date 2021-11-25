from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import copy

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
            nn.Flatten(),
        )
        self.conv2 = copy.deepcopy(self.conv)
        conv_out_size = self._get_conv_out(input_shape)
        self.actor = nn.Sequential(
            self.conv,
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )
        self.critic = nn.Sequential(
            self.conv2,
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape: Tuple):
        """Function to get the output shape of the cnn

        Args:
            shape (Tuple): the input shape to the cnn

        Returns:
            int: the output size of the cnn
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: Any):
        """Function to use the network on a given state in the shape of (1, *input_shape)

        Args:
            x (Any): given state to calculate from

        Returns:
            Tuple: the given action to preform, and the given q-tabell for the given state
        """
        return (
            Categorical(logits=self.actor(x)),
            self.critic(x).reshape(-1),
        )

    def save(self):
        """Function to save the model to file
        """
        torch.save(self.state_dict(), PPO_MODEL_SAVE_NAME)

    def load(self, device: str):
        """Function to load a model to a given device (cpu or gpu)

        Args:
            device (str): the given device
        """
        self.load_state_dict(
            torch.load(PPO_MODEL_SAVE_NAME, map_location=torch.device(device))
        )
