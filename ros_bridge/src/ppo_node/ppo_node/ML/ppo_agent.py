import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal, Normal
from .parameters import *

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # might reduce performance time! Uncomment for debugging CUDA errors
os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################################
#                                       ACTOR AND CRITIC NETWORKS
##################################################################################################


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        
        log_std_init = torch.tensor([np.log(0.3), np.log(0.8)], dtype=torch.float32)

        self.log_std = nn.Parameter(log_std_init.clone())

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc1 → tanh: {x}")
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc2 → tanh: {x}")
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc3 → tanh: {x}")
        raw_action_mean = self.fc4(x)
        return raw_action_mean

    def get_dist(self, state):
        raw_action_mean = self.forward(state)
        action_std = torch.exp(self.log_std)
        cov_mat = torch.diag_embed(action_std.expand_as(raw_action_mean))
        dist = MultivariateNormal(raw_action_mean, cov_mat)
        return dist

    def sample_action(self, state):
        dist = self.get_dist(state)
        raw_action = dist.rsample() # Reparameterization trick
        
        normal = Normal(0, 1)
        cdf_action = normal.cdf(raw_action)  # (batch_size, 2)
    
        log_prob = dist.log_prob(raw_action).unsqueeze(-1)        
        if not torch.isfinite(log_prob).all():
            raise RuntimeError(f"NaN/Inf in log_prob: {log_prob}")

        steer = 2.0 * cdf_action[:, 0:1] - 1.0      # [-1, 1]
        throttle = cdf_action[:, 1:2]              # [0, 1]
        action = torch.cat([steer, throttle], dim=1)

        return action.detach(), log_prob  # detach if you're not backproping through

    def evaluate_actions(self, state, action):
        normal = Normal(0, 1)
        
        steer, throttle = action[:, 0:1], action[:, 1:2]
        
        steer_raw = normal.icdf(((steer + 1.0) / 2.0).clamp(1e-4, 1 - 1e-4))     # map [-1, 1] → [0, 1] → raw
        throttle_raw = normal.icdf(throttle.clamp(1e-4, 1 - 1e-4))               # already [0, 1]
        
        raw_action = torch.cat([steer_raw, throttle_raw], dim=1)
        
        dist = self.get_dist(state)
        log_prob = dist.log_prob(raw_action)
        total_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, total_entropy
    
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
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc1")
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc2")
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc3")
        value = self.fc4(x)
        return value


##################################################################################################
#                                       PPO AGENT
##################################################################################################


class PPOAgent:
    def __init__(self, input_dim=197, action_dim=2, summary_writer=None, logger=None):
        self.logger = logger
        if self.logger is None:
            raise ValueError("Logger not provided. Please provide a logger instance.")
        self.logger.info("Initializing PPO Agent...")
        self.logger.info(f"device: {device}")
        self.input_dim = input_dim if PPO_INPUT_DIM is None else PPO_INPUT_DIM
        self.logger.info(f"Action input dimension: {self.input_dim}")
        self.action_dim = action_dim
        self.logger.info(f"Action output dimension: {self.action_dim}")

        self.summary_writer = summary_writer
        
        self.entropy_coef = ENTROPY_COEF
        
        self.actor = ActorNetwork(self.input_dim, action_dim).to(
            device
        )
        self.critic = CriticNetwork(self.input_dim).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=ACTOR_LEARNING_RATE
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=CRITIC_LEARNING_RATE
        )
        
        # Experience storage
        (
            self.states,
            self.actions,
            self.log_probs,
            self.vals,
            self.rewards,
            self.dones,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        self.learn_step_counter = 0  # Track PPO updates

        if not os.path.exists(PPO_CHECKPOINT_DIR):
            os.makedirs(PPO_CHECKPOINT_DIR)

        self.logger.info("PPO Agent initialized.")

    ##################################################################################################
    #                                        SELECT ACTION
    ##################################################################################################

    def select_action(self, state):
        """Select an action and return its log probability."""
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

        if state.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, but got {state.shape[1]}"
            )

        if torch.isnan(state).any():
            raise ValueError(f"NaN detected in input state: {state}")

        with torch.no_grad():
            action, log_prob = self.actor.sample_action(state)

        return action.cpu().numpy()[0], log_prob

    ##################################################################################################
    #                                     STORE LAST EXPERIENCE
    ##################################################################################################

    def store_transition(self, state, action, log_prob, val, reward, done):
        """Store experience for PPO updates. (every stored data will be used in the next update)"""
        self.states.append(state)
        self.actions.append(action)
        if not isinstance(log_prob, torch.Tensor):
            lp = torch.tensor([log_prob], dtype=torch.float32, device=device)
        else:
            lp = log_prob.detach()   # shape [1]
        self.log_probs.append(lp)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    ##################################################################################################
    #                                       SAVE AND LOAD MODELS
    ##################################################################################################

    def save_model_and_optimizers(self, directory):
        self.logger.info("Saving model + optimizer...")
        try:

            torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
            torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

            torch.save(
                self.actor_optimizer.state_dict(),
                os.path.join(directory, "actor_optim.pth"),
            )
            torch.save(
                self.critic_optimizer.state_dict(),
                os.path.join(directory, "critic_optim.pth"),
            )

            self.logger.info(f"Model + optimizer saved to {directory}")
        except Exception as e:
            self.logger.info(f"❌ Error saving model + optimizer: {e}")

    def load_model_and_optimizers(self, directory):
        self.logger.info(f"Loading model + optimizer from: {directory}")
        try:
            self.actor.load_state_dict(
                torch.load(os.path.join(directory, "actor.pth"), weights_only=False)
            )
            self.critic.load_state_dict(
                torch.load(os.path.join(directory, "critic.pth"), weights_only=False)
            )

            self.actor_optimizer.load_state_dict(
                torch.load(
                    os.path.join(directory, "actor_optim.pth"), weights_only=False
                )
            )
            self.critic_optimizer.load_state_dict(
                torch.load(
                    os.path.join(directory, "critic_optim.pth"), weights_only=False
                )
            )

            self.logger.info("Model and optimizer states loaded successfully.")
        except Exception as e:
            self.logger.info(f"❌ Error loading model and optimizer: {e}")

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

    def normalize_rewards(self, rewards):
        """Normalize rewards before training."""
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

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
        values = np.array(self.vals, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Convert lists to tensors
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        values = torch.tensor(values).to(device).detach()
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)
        old_log_probs = torch.cat(self.log_probs, dim=0).detach()
        if old_log_probs.device != device:
            old_log_probs = old_log_probs.to(device)
        if old_log_probs.dim() == 1:
            old_log_probs = old_log_probs.unsqueeze(1)
        
        rewards = self.normalize_rewards(rewards)

        advantages = self.compute_gae(values, rewards, dones)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        decay_base = 0.999
        initial_entropy_coef = 0.05
        min_entropy_coef = 0.001
        self.entropy_coef = max(initial_entropy_coef * (decay_base ** self.learn_step_counter), min_entropy_coef)

        
        # Perform PPO optimization steps
        for _ in range(NUM_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(indices), MINIBATCH_SIZE):
                batch = indices[start : start + MINIBATCH_SIZE]

                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch].unsqueeze(1)
                batch_values = values[batch].unsqueeze(1)
                batch_returns = batch_advantages + batch_values

                # Get new action probabilities and entropy
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = self.critic(batch_states).squeeze()
                
                new_log_probs = new_log_probs.squeeze()
                if new_log_probs.dim() == 1:
                    new_log_probs = new_log_probs.unsqueeze(1)
                    
                kl = (batch_old_log_probs - new_log_probs).mean()
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                print("PPO ratio:", ratio.mean().item(), "std:", ratio.std().item())
                print("   ratio:", ratio.shape)

                # PPO loss terms
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - POLICY_CLIP, 1 + POLICY_CLIP)
                    * batch_advantages
                )
                
                # Actor loss
                actor_loss = (
                    -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                )

                # A new suggestion to calculate the critic loss
                old_values_batch = batch_values.detach().view(-1)  # use stored values
                new_values = state_values.view(-1)
                clipped_values = old_values_batch + (
                    new_values - old_values_batch
                ).clamp(-0.2, 0.2)

                value_loss_unclipped = F.mse_loss(
                    new_values, batch_returns.detach().view(-1)
                )
                value_loss_clipped = F.mse_loss(
                    clipped_values, batch_returns.detach().view(-1)
                )
                critic_loss = VF_COEF * torch.max(
                    value_loss_unclipped, value_loss_clipped
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                with torch.no_grad():
                    # steer  std ∈ [0.05, 0.6]
                    self.actor.log_std.data[0].clamp_(np.log(0.05), np.log(0.6))
                    # throttle std ∈ [0.05, 1.0]
                    self.actor.log_std.data[1].clamp_(np.log(0.05), np.log(0.8))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        self.learn_step_counter += 1

        if self.summary_writer is not None:
            with torch.no_grad():
                current_log_std = self.actor.log_std.detach().cpu().numpy()
                current_action_std = torch.exp(self.actor.log_std.detach()).cpu().numpy()

                self.logger.info(
                    f"[Learn Step {self.learn_step_counter}] log_std: {current_log_std}, action_std: {current_action_std}"
                )
                # Per-dimension logging
                for i, (ls, std) in enumerate(zip(current_log_std, current_action_std)):
                    self.summary_writer.add_scalar(f"Exploration/log_std_dim_{i}", ls, self.learn_step_counter)
                    self.summary_writer.add_scalar(f"Exploration/std_dim_{i}", std, self.learn_step_counter)
                # Overall logging
                self.summary_writer.add_scalar("Exploration/mean log std", current_log_std.mean(), self.learn_step_counter)
                self.summary_writer.add_scalar("Exploration/mean std", current_action_std.mean(), self.learn_step_counter)
                self.summary_writer.add_scalar("KL/mean_kl_div", kl.item(), self.learn_step_counter)
                self.summary_writer.add_scalar("Exploration/entropy_coef", self.entropy_coef, self.learn_step_counter)


        # Clear stored experiences
        (
            self.states,
            self.actions,
            self.log_probs,
            self.vals,
            self.rewards,
            self.dones,
        ) = (
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
            total_entropy / num_batches,
        )
