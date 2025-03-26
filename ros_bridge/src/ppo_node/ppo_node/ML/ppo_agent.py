import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
import sys
from .parameters import *

import ppo_node # check that it doesnt affect something badly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################################
#                                       ACTOR AND CRITIC NETWORKS
##################################################################################################

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, action_std_init):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        # self.action_var = torch.full(
        #     (action_dim,), action_std_init * action_std_init
        # ).to(device) # Been replaced by the line below log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(action_std_init))  # log_std is learnable


        # Actor network layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)  # Output layer for all actions
        
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):        
        try:
            x = F.relu(self.fc1(state))
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after fc1 in ActorNetwork{x}")
        except ValueError as e:
            print(e)
            return None

        try:
            x = F.relu(self.fc2(x))
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after fc2 in ActorNetwork{x}")
        except ValueError as e:
            print(e)
            return None

        try:
            x = F.relu(self.fc3(x))
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after fc3 in ActorNetwork {x}")
        except ValueError as e:
            print(e)
            return None

        try:
            action_mean = self.fc4(x)
            if torch.isnan(action_mean).any():
                raise ValueError(f"NaN detected after fc4 in ActorNetwork{x}")
        except ValueError as e:
            print(e)
            return None

        try:
            steering_mean = torch.tanh(action_mean[:, 0:1])  # tanh to restrict to [-1, 1]
            throttle_mean = torch.sigmoid(action_mean[:, 1:2])  # sigmoid to restrict to [0, 1]
            brake_mean = torch.sigmoid(action_mean[:, 2:3])  # sigmoid to restrict to [0, 1]

            action_mean = torch.cat([steering_mean, throttle_mean, brake_mean], dim=1)
            if torch.isnan(action_mean).any():
                raise ValueError("NaN detected after concatenating means in ActorNetwork")
        except ValueError as e:
            print(e)
            return None


        return action_mean

    # def set_action_std(self, new_action_std):
    #     self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def get_dist(self, state):
        action_mean = self.forward(state)
        # action_var = self.action_var.expand_as(action_mean) # Been replaced by the line below action_std
        action_std = torch.exp(self.log_std)
        
        # Check for NaNs after expand_as
        if torch.isnan(action_std).any():
            raise ValueError("NaN detected after expand_as in ActorNetwork")
        
        # cov_mat = torch.diag_embed(action_var) # Been replaced by the line below cov_mat
        cov_mat = torch.diag_embed(action_std.expand_as(action_mean))
                
        # Check for NaNs after diag_embed
        if torch.isnan(cov_mat).any():
            raise ValueError("NaN detected after diag_embed in ActorNetwork")
        
        dist = MultivariateNormal(action_mean, cov_mat)
        return dist

    def sample_action(self, state):
        dist = self.get_dist(state)
        action = dist.sample()
        
        # Check for NaNs after sampling action
        if torch.isnan(action).any():
            raise ValueError("NaN detected after sampling action in ActorNetwork")
        
        action_logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Clamp actions to appropriate ranges
        action_np = action.detach().cpu().numpy()
        action_np[:, 0] = np.clip(action_np[:, 0], -1.0, 1.0)  # steering
        action_np[:, 1] = np.clip(action_np[:, 1], 0.0, 1.0)  # throttle
        action_np[:, 2] = np.clip(action_np[:, 2], 0.0, 1.0)  # brake

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

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value


##################################################################################################
#                                       PPO AGENT
##################################################################################################

class PPOAgent:
    def __init__(self, input_dim=198, action_dim=3, summary_writer=None):
        print("\nInitializing PPO Agent...\n")
        print("device: ", device)
        self.input_dim = input_dim if PPO_INPUT_DIM is None else PPO_INPUT_DIM
        print(f"Action input dimension: {self.input_dim}")
        self.action_dim = action_dim
        print(f"Action output dimension: {self.action_dim}")
        self.action_std = ACTION_STD_INIT
        
        self.summary_writer = summary_writer
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(self.input_dim, action_dim, self.action_std).to(
            device
        )
        self.critic = CriticNetwork(self.input_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LEARNING_RATE)

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
        
        self.prefix = "Train" if TRAIN else "Test" # For logging

        # print(f"Initial action std: {self.actor.log_std.exp().detach().cpu().numpy()}") # To check that log_std.exp() is around ACTION_STD_INIT:

        # Ensure checkpoint directory exists
        if not os.path.exists(PPO_CHECKPOINT_DIR):
            os.makedirs(PPO_CHECKPOINT_DIR)

    ##################################################################################################
    #                                        SELECT ACTION
    ##################################################################################################

    def select_action(self, state):
        """Select an action and return its log probability."""
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

        # Check if the input dimension matches the expected input dimension
        if state.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, but got {state.shape[1]}"
            )
            
        # Check for NaN values in the input state
        if torch.isnan(state).any():
            raise ValueError(f"NaN detected in input state: {state}")

        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state)
        # print(f"\nSteering: {action[0][0]}, Throttle: {action[0][1]}, Brake: {action[0][2]}\n")
        return action.cpu().numpy()[0], log_prob

    ##################################################################################################
    #                                     STORE LAST EXPERIENCE
    ##################################################################################################

    def store_transition(self, state, action, prob, val, reward, done):
        """Store experience for PPO updates. (every stored data will be used in the next update)"""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    ##################################################################################################
    #                                       SAVE AND LOAD MODELS
    ##################################################################################################

    def save_model_and_optimizers(self, directory):
        print("Saving model + optimizer...")
        try:

            # Save model parameters
            torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
            torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

            # Save optimizer states
            torch.save(
                self.actor_optimizer.state_dict(),
                os.path.join(directory, "actor_optim.pth"),
            )
            torch.save(
                self.critic_optimizer.state_dict(),
                os.path.join(directory, "critic_optim.pth"),
            )

            print(f"Model + optimizer saved to {directory}")
        except Exception as e:
            print(f"❌ Error saving model + optimizer: {e}")

    def load_model_and_optimizers(self, directory):
        print(f"Loading model + optimizer from: {directory}")
        try:
            # Load model weights
            self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
            self.critic.load_state_dict(
                torch.load(os.path.join(directory, "critic.pth"))
            )

            # Load optimizer states
            self.actor_optimizer.load_state_dict(
                torch.load(os.path.join(directory, "actor_optim.pth"))
            )
            self.critic_optimizer.load_state_dict(
                torch.load(os.path.join(directory, "critic_optim.pth"))
            )

            print("Model and optimizer states loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model and optimizer: {e}")

    ##################################################################################################
    #                                  DECAY ACTION STD - NOT USED
    ##################################################################################################

    # Removed for now - trying to use a learnable log_std instead of decaying action_std
    
    # def decay_action_std(self, episode_num): 
    #     """
    #     Decay the action standard deviation for the exploration-exploitation tradeoff.
    #     At the beginning of training, the agent should explore more, so the action_std is higher.
    #     Due to frequent collisions at the start of training, we wait for more episodes before starting to decay the action_std.
    #     As training progresses, the agent should exploit more, so the action_std is gradually reduced.
    #     """
        
    #     if episode_num < 5000:
    #         decay_freq = 2500
    #     else:
    #         decay_freq = 1000

    #     if episode_num % decay_freq == 0 and self.action_std > MIN_ACTION_STD:
    #         self.action_std = max(
    #             MIN_ACTION_STD, self.action_std - ACTION_STD_DECAY_RATE
    #         )
    #         self.actor.set_action_std(self.action_std)
    #         if self.summary_writer is not None:
    #             self.summary_writer.add_scalar("Exploration/ActionStd", self.action_std, episode_num)
    
    ##################################################################################################
    #                                       NORMALIZE ADVANTAGES
    ##################################################################################################
    
    def normalize_advantages(self, advantages):
        """Normalize advantages for stability."""
        if advantages.numel() > 1:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            return advantages

    ##################################################################################################
    #                                           COMPUTE GAE
    ##################################################################################################

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

    ##################################################################################################
    #                                       NORMALIZE REWARDS
    ##################################################################################################

    def normalize_rewards(self):
        """Normalize rewards before training."""
        self.rewards = (self.rewards - np.mean(self.rewards)) / (
            np.std(self.rewards) + 1e-8
        )

    ##################################################################################################
    #                                       MAIN PPO ALGORITHM
    ##################################################################################################


    def learn(self):
        """
        Perform PPO training using stored experiences (from latest batch).
        returns: actor_loss, critic_loss, entropy
        """
        # Convert lists to numpy arrays first
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        old_probs = np.array(self.probs, dtype=np.float32)
        values = np.array(self.vals, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        # Convert lists to tensors
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        old_probs = torch.tensor(old_probs).to(device).detach()
        values = torch.tensor(values).to(device).detach()
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)

        # Normalize rewards
        self.normalize_rewards()

        # Compute advantages
        advantages = self.compute_gae(values, rewards, dones)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        # Perform PPO optimization steps
        for _ in range(NUM_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)  # Shuffle indices for mini-batch sampling
            for start in range(0, len(indices), MINIBATCH_SIZE):
                batch = indices[start : start + MINIBATCH_SIZE]

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
                
                # A new suggestion to calculate the critic loss
                old_values_batch = values[batch].detach().view(-1)  # use stored values
                new_values = state_values.view(-1)
                clipped_values = old_values_batch + (new_values - old_values_batch).clamp(-0.2, 0.2)
                
                value_loss_unclipped = F.mse_loss(new_values, batch_returns.detach().view(-1))
                value_loss_clipped = F.mse_loss(clipped_values, batch_returns.detach().view(-1))
                critic_loss = VF_COEF * torch.max(value_loss_unclipped, value_loss_clipped)
                
                # Critic loss - Old implementation
                # critic_loss = VF_COEF * F.mse_loss(state_values.view(-1), batch_returns.detach().view(-1))
                
                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5
                )  # Gradient clipping
                self.actor_optimizer.step()
                
                with torch.no_grad(): 
                    self.actor.log_std.clamp_(np.log(0.05), np.log(1.5)) # Prevent entropy from exploding
                    
                # Optimize critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 0.5
                )  # Gradient clipping
                self.critic_optimizer.step()
    
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1
                   
            # For KL divergence monitoring
            with torch.no_grad():
                old_log_probs = batch_old_probs
                new_log_probs = new_probs
                kl_div = (old_log_probs - new_log_probs).mean()
                print(f"KL Divergence: {kl_div.item():.4f}")
                # Log to tensorboard
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar("Exploration/KL divergence", kl_div.item(), self.learn_step_counter)
                # # Optional: stop update early if KL is too high
                # if kl_div.item() > MAX_KL:
                #     print(f"[KL WARNING] KL divergence {kl_div.item():.4f} too high. Breaking PPO epoch early.")
                #     break

        # Increment learn step counter
        self.learn_step_counter += 1
        
        if self.summary_writer is not None:
            current_action_std = torch.exp(self.actor.log_std).mean().item()
            current_log_std = self.actor.log_std.data.cpu().numpy()
            
            print(f"[Learn Step {self.learn_step_counter}] log_std: {current_log_std}, action_std: {current_action_std}")

            self.summary_writer.add_scalar("Exploration/learned action std", current_action_std, self.learn_step_counter)
            self.summary_writer.add_scalar("Exploration/mean log std", current_log_std.mean().item(), self.learn_step_counter)


        # Clear stored experiences
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # Return averaged metrics
        return (
            total_actor_loss / num_batches,
            total_critic_loss / num_batches,
            total_entropy / num_batches
        )
