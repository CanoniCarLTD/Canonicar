"""
    ** Data saving and loading functions needs to be added **
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import sys
from .parameters import *

# Set device to cuda if available, otherwise cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, action_std_init):  # Reordered arguments
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(device)

        # Actor network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)  # Output layer for all actions

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Get mean values for each control
        action_mean = self.fc4(x)
        steering_mean = torch.tanh(action_mean[:, 0:1])  # tanh to restrict to [-1, 1]
        throttle_mean = torch.sigmoid(
            action_mean[:, 1:2]
        )  # sigmoid to restrict to [0, 1]
        brake_mean = torch.sigmoid(action_mean[:, 2:3])  # sigmoid to restrict to [0, 1]

        # Combine into a single action tensor
        action_mean = torch.cat([steering_mean, throttle_mean, brake_mean], dim=1)
        return action_mean

    def set_action_std(self, new_action_std):
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(device)

    def get_dist(self, state):
        action_mean = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        return dist

    def sample_action(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Clamp actions to appropriate ranges
        action_np = action.detach().cpu().numpy()
        action_np[:, 0] = np.clip(action_np[:, 0], -1.0, 1.0)  # steering
        action_np[:, 1] = np.clip(action_np[:, 1], 0.0, 1.0)  # throttle
        action_np[:, 2] = np.clip(action_np[:, 2], 0.0, 1.0)  # brake

        # Convert back to tensor
        action = torch.FloatTensor(action_np).to(device)

        return action, action_logprob

    def evaluate_actions(self, state, action):
        dist = self.get_dist(state)
        action_logprobs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_logprobs, dist_entropy


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()

        # Critic network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value


"""
    PPOMemory WILL BE DELETED, BUT KEEPING FOR IDEAS OF WHAT TO SAVE
"""


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
from .parameters import *

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, input_dim=203, action_dim=3):
        self.input_dim = input_dim if PPO_INPUT_DIM is None else PPO_INPUT_DIM
        print(f"Action input dimension: {self.input_dim}")
        self.action_dim = action_dim
        print(f"Action output dimension: {self.action_dim}")
        self.action_std = ACTION_STD_INIT
        print(f"Action std init: {self.action_std}")

        # Initialize actor and critic networks
        self.actor = ActorNetwork(self.input_dim, action_dim, self.action_std).to(
            device
        )
        self.critic = CriticNetwork(self.input_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # Experience storage
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        self.learn_step_counter = 0  # Track PPO updates

        # Load model if specified
        if MODEL_LOAD and CHECKPOINT_FILE:
            self.load_models(CHECKPOINT_FILE)

        # Ensure checkpoint directory exists
        if not os.path.exists(PPO_CHECKPOINT_DIR):
            os.makedirs(PPO_CHECKPOINT_DIR)

    def select_action(self, state):
        """Select an action and return its log probability."""
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

        # Check if the input dimension matches the expected input dimension
        if state.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, but got {state.shape[1]}"
            )

        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state)
        print(
            f"\nSteering: {action[0][0]}, Throttle: {action[0][1]}, Brake: {action[0][2]}\n"
        )
        return action.cpu().numpy()[0], log_prob

    def store_transition(self, state, action, prob, val, reward, done):
        """Store experience for PPO updates. (every stored data will be used in the next update)"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def save_models(self, directory):
        print("... saving models ...")
        torch.save(
            self.actor.state_dict(),
            os.path.join(directory, "actor.pth"),
        )
        torch.save(
            self.critic.state_dict(),
            os.path.join(directory, "critic.pth"),
        )

    def load_models(self, directory):
        print(f"... loading models {directory} ...")
        try:
            self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
            self.critic.load_state_dict(
                torch.load(os.path.join(directory, "critic.pth"))
            )
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")

    def decay_action_std(self):
        """Decay action standard deviation for exploration-exploitation tradeoff."""
        if self.learn_step_counter % (TOTAL_TIMESTEPS // 10) == 0:
            self.action_std = max(
                MIN_ACTION_STD, self.action_std - ACTION_STD_DECAY_RATE
            )
            self.actor.set_action_std(self.action_std)
            print(f"Action std decayed to: {self.action_std}")

    def normalize_advantages(self, advantages):
        """Normalize advantages for stability."""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def compute_gae(self, values, rewards, dones, gamma=GAMMA, lam=LAMBDA_GAE):
        """Compute Generalized Advantage Estimation (GAE)."""
        values = torch.cat(
            (values, torch.zeros(1).to(device))
        )  # Add zero for last next_value
        advantages = torch.zeros_like(rewards).to(device)
        gae = 0

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages[step] = gae

        return self.normalize_advantages(advantages)

    def normalize_rewards(self):
        """Normalize rewards before training."""
        self.rewards = (self.rewards - np.mean(self.rewards)) / (
            np.std(self.rewards) + 1e-8
        )

    def learn(self):
        """
        Perform PPO training using stored experiences (from latest batch).
        """
        # Convert lists to tensors
        states = torch.tensor(self.states, dtype=torch.float32).to(device)
        actions = torch.tensor(self.actions, dtype=torch.float32).to(device)
        old_probs = torch.tensor(self.probs, dtype=torch.float32).to(device).detach()
        values = torch.tensor(self.vals, dtype=torch.float32).to(device).detach()
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)

        # Normalize rewards
        self.normalize_rewards()

        # Compute advantages
        advantages = self.compute_gae(values, rewards, dones)

        # Perform PPO optimization steps
        for _ in range(NUM_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)  # Shuffle indices for mini-batch sampling
            for start in range(0, len(indices), BATCH_SIZE):
                batch = indices[start : start + BATCH_SIZE]

                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = batch_advantages + values[batch]  # Proper critic target

                # Get new action probabilities and entropy
                new_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = self.critic(batch_states).squeeze()

                # PPO ratio
                ratio = torch.exp(new_probs - batch_old_probs)

                # PPO loss terms
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - POLICY_CLIP, 1 + POLICY_CLIP)
                    * batch_advantages
                )

                # Actor loss
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()
                )

                # Critic loss
                critic_loss = VF_COEF * F.mse_loss(state_values, batch_returns.detach())

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5
                )  # Gradient clipping
                self.actor_optimizer.step()

                # Optimize critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 0.5
                )  # Gradient clipping
                self.critic_optimizer.step()

        # Increment learn step counter & decay action standard deviation
        self.learn_step_counter += 1
        self.decay_action_std()

        # Clear stored experiences
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
