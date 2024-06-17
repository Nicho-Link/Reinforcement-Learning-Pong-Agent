import random
import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, capacity: int = 10000):
        """Initializes the Experience Replay Buffer

        Args:
            capacity (int, optional): The maximum number of experiences that can be stored in the buffer. Defaults to 10000.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(
        self,
        state: np.array,
        action: int,
        next_state: np.array,
        reward: int,
        resetnow: bool,
    ):
        """Adds an experience to the buffer

        Args:
            state (np.array): The initial state
            action (int): The action taken
            next_state (np.array): The next state
            reward (int): The reward received
            resetnow (bool): Flag to indicate if the episode has ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, next_state, reward, resetnow))
        else:
            self.buffer[self.position] = (state, action, next_state, reward, resetnow)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Samples a batch of experiences from the buffer

        Args:
            batch_size (int): Size of the batch to be sampled

        Returns:
            list: A list of experiences
        """
        return random.sample(self.buffer, batch_size)
