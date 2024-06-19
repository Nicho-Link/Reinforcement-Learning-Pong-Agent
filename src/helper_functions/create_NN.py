from torch import nn
import copy
import numpy as np


class DDQN(nn.Module):
    def __init__(self, state_shape: tuple, num_actions: int):
        """Initializes the DQN model

        Args:
            state_shape (tuple): The shape of the states
            num_actions (int): The number of actions
        """
        super().__init__()
        frames, height, width = state_shape

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.target = copy.deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state: np.array, model: str):
        """forward pass for the model

        Args:
            state (np.array): The state
            model (str): The model to be used

        Returns:
            np.array: The output of the model
        """
        if model == "online":
            return self.online(state)
        elif model == "target":
            return self.target(state)
