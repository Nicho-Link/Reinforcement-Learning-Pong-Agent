# Bugs fixed with the help of Pytorch Tutorial: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

import torch
import numpy as np
import torch.nn.functional as F
import os

# Import own Functions
from src.helper_functions.create_NN import DDQN
from src.helper_functions.create_ExpRepBuf import ExperienceReplayBuffer


class EpsilonGreedyAgent:
    def __init__(
        self,
        num_actions: int,
        state_shape: tuple,
        checkpoint_folder: str,
        model_folder: str,
        wantcuda: bool = True,
        starting_point: str = None,
        learning_rate: float = 0.00025,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9995,
        batch_size: int = 32,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        exp_before_training: int = 100000,
        online_update_every: int = 3,
        exp_before_target_sync: int = 10000,
        save_every: int = 50,
    ):
        """Initialize an Super-Mario-Bros Agent

        Args:
            num_actions (int): Number of possible actions
            state_shape (tuple): The shape of the states
            checkpoint_folder (str): The path to the checkpoint folder
            model_folder (str): The path to the model folder
            wantcuda (bool, optional): True if you want to use CUDA (GPU). Defaults to True.
            starting_point (str, optional): The path to your starting-/checkpoint. Defaults to None.
            learning_rate (float, optional): Your learning rate. Defaults to 0.00025.
            epsilon_start (float, optional): At which value Epsilon should start. Defaults to 1.0.
            epsilon_min (float, optional): At which value Epsilon should stop. Defaults to 0.1.
            epsilon_decay (float, optional): At which rate Epsilon should decay. Defaults to 0.9995.
            batch_size (int, optional): The size of the batch from Replay Buffer. Defaults to 32.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            buffer_size (int, optional): The size of the Replay Buffer. Defaults to 100000.
            exp_before_training (int, optional): Number of experiences before training starts. Defaults to 100000.
            online_update_every (int, optional): Number of episodes before the online model gets updated. Defaults to 3.
            exp_before_target_sync (int, optional): Number of experiences before the target model gets updated. Defaults to 10000.
            save_every (int, optional): Number of episodes before the model gets saved. Defaults to 50.
        """
        if wantcuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.checkpoint_folder = checkpoint_folder
        self.model_folder = model_folder
        self.starting_point = starting_point
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.exp_before_training = exp_before_training
        self.online_update_every = online_update_every
        self.exp_before_target_sync = exp_before_target_sync
        self.save_every = save_every

        self.current_step = 0

        self.model = DDQN(self.state_shape, self.num_actions).float()
        self.model.to(device=self.device)

        self.memory = ExperienceReplayBuffer(self.buffer_size)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        
        self.loss_function = torch.nn.HuberLoss(delta=1.0)

        if self.starting_point != None:
            self.loadModel(self.starting_point)

    def selectAction(self, state: np.array):
        """Select an action based on the current state

        Args:
            state (np.array): The current state

        Returns:
            int: The action index that should be taken
        """

        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        if np.random.rand() < self.epsilon:
            # Exploration
            action = np.random.randint(self.num_actions)
        else:
            # Exploitation
            state = state.unsqueeze(0)
            action_values = self.model(state, model="online")
            action = torch.argmax(action_values, axis=1).item()

        self.current_step = self.current_step + 1

        return action

    def decayEpsilon(self, strat: str = "exp"):
        """Decay the epsilon value

        Args:
            strat (str, optional): The strategy to decay the epsilon value Possible values: "exp" for exponential or "lin" for linear. Remember to adjust the decay. Defaults to "exp".
        """
        if strat == "exp":
            self.epsilon = self.epsilon * self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        elif strat == "lin":
            self.epsilon = self.epsilon - self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        else:
            print("No valid strategy for epsilon decay")

    def saveExp(
        self,
        state: np.array,
        action: int,
        next_state: np.array,
        reward: int,
        resetnow: bool,
    ):
        """Save the experience in the Replay Buffer

        Args:
            state (np.array): The initial state
            action (int): The action that was taken
            next_state (np.array): The next state
            reward (int): The reward that was received
            resetnow (bool): If the episode is over after this action
        """
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor([action]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        resetnow = torch.FloatTensor([resetnow]).to(self.device)

        self.memory.add(state, action, next_state, reward, resetnow)

    def useExp(self):
        """Sample a batch from the Replay Buffer

        Returns:
            tuple: A tuple with torch.Tensors of the states, actions, next_states, rewards and resets
        """
        batch = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, resets = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        resets = torch.stack(resets)

        return (
            states,
            actions.squeeze(),
            next_states,
            rewards.squeeze(),
            resets.squeeze(),
        )

    def estimateTDerror(self, state: np.array, action: int):
        """Estimate the TD error

        Args:
            state (np.array): The current state
            action (int): The action that was taken

        Returns:
            float: The current Q value
        """
        action = action.long()
        current_Q = self.model(state, model="online")[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    def estimateQTarget(
        self, next_states: torch.Tensor, rewards: torch.Tensor, resets: torch.Tensor
    ):
        """Estimate the Q target

        Args:
            next_states (torch.Tensor): A batch of next states
            rewards (torch.Tensor): A batch of rewards
            resets (torch.Tensor): A batch of resets

        Returns:
            float: The Q target
        """
        with torch.no_grad():
            next_state_Q = self.model(next_states, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.model(next_states, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
        return (rewards + (1 - resets.float()) * self.gamma * next_Q).float()

    def update_Q_online_get_loss(self, td_estimation: float, q_target: float):
        """Update the online model and get the loss

        Args:
            td_estimation (float): The current Q value
            q_target (float): The Q target

        Returns:
            float: The loss
        """
        loss = self.loss_function(td_estimation, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Sync the target model with the online model"""
        self.model.target.load_state_dict(self.model.online.state_dict())

    def learn_get_TDest_loss(self):
        """Learn from the experiences and get the TD error and loss

        Returns:
            tuple: A tuple with the TD error and the loss
        """
        if self.current_step % self.exp_before_target_sync == 0:
            self.sync_Q_target()

        if self.current_step < self.exp_before_training:
            return 0, 0

        states, actions, next_states, rewards, resets = self.useExp()

        td_estimation = self.estimateTDerror(states, actions)

        q_target = self.estimateQTarget(next_states, rewards, resets)

        loss = self.update_Q_online_get_loss(td_estimation, q_target)

        return (td_estimation.mean().item(), loss)

    def loadModel(self, path: str):
        """Load a model from a checkpoint

        Args:
            path (str): The path to the checkpoint. The checkpoint should be a dictionary with the keys "epsilon", "model" and "optimizer" while "model" and "optimizer" are the state_dicts of the model and optimizer.
        """
        checkpoint = torch.load(path, map_location=self.device)
        epsilon = checkpoint["epsilon"]
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

        print(f"Model loaded from {path} with epsilon {epsilon}")
        self.model.load_state_dict(model)
        self.optimizer.load_state_dict(optimizer)
        self.epsilon = epsilon
