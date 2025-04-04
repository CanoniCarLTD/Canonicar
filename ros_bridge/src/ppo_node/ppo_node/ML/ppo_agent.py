import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
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
        
        # Steering ~0.2, throttle/brake slightly less to avoid overexploration
        log_std_init = torch.tensor([np.log(0.2), np.log(0.2), np.log(0.2)], dtype=torch.float32)

        self.log_std = nn.Parameter(log_std_init.clone())  # learnable per-dimension

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

        self.init_weights()

    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc1 → tanh: {x}")

        x = self.fc2(x)
        x = torch.tanh(x)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc2 → tanh: {x}")

        x = self.fc3(x)
        x = torch.tanh(x)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"NaN/Inf after fc3 → tanh: {x}")

        action_mean = self.fc4(x)
        if not torch.isfinite(action_mean).all():
            raise RuntimeError(
                f"NaN/Inf in action_mean (before squashing): {action_mean}"
            )

        # Split and squash
        steering_mean = torch.tanh(action_mean[:, 0:1])
        throttle_mean = 0.5 * (torch.tanh(action_mean[:, 1:2]) + 1.0)
        brake_mean = 0.5 * (torch.tanh(action_mean[:, 2:3]) + 1.0)
        action_mean = torch.cat([steering_mean, throttle_mean, brake_mean], dim=1)

        if not torch.isfinite(action_mean).all():
            raise RuntimeError(
                f"NaN/Inf in action_mean (after squashing): {action_mean}"
            )

        return action_mean

    def get_dist(self, state):
        action_mean = self.forward(state)

        print(f"[DEBUG] action_mean shape: {action_mean.shape}")  # Expect: (batch_size, action_dim)
        print(f"[DEBUG] log_std shape: {self.log_std.shape}")     # Expect: (action_dim,)

        if not torch.isfinite(self.log_std).all():
            raise RuntimeError(f"NaN/Inf in log_std: {self.log_std}")

        # Check log_std with CPU operation first
        log_std_cpu = self.log_std.detach().cpu()
        if not torch.isfinite(log_std_cpu).all():
            self.logger.info(f"Warning: Non-finite values in log_std: {log_std_cpu}")

        action_std = torch.exp(self.log_std)

        if not torch.isfinite(action_std).all():
            raise RuntimeError(f"NaN/Inf in exp(log_std): {action_std}")

        if (action_std < 1e-6).any():
            raise RuntimeError(f"Action std too small: {action_std}")

        cov_mat = torch.diag_embed(action_std.expand_as(action_mean))

        if not torch.isfinite(cov_mat).all():
            raise RuntimeError(f"NaN/Inf in cov_mat: {cov_mat}")

        dist = MultivariateNormal(action_mean, cov_mat)
        return dist

    def sample_action(self, state):
        dist = self.get_dist(state)
        action = dist.sample()

        if not torch.isfinite(action).all():
            raise RuntimeError(f"NaN/Inf in sampled action: {action}")

        action_logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        if not torch.isfinite(action_logprob).all():
            raise RuntimeError(f"NaN/Inf in log_prob: {action_logprob}")

        # Clamp and return
        action[:, 0] = torch.clamp(action[:, 0], -1.0, 1.0)  # steering
        action[:, 1] = torch.clamp(action[:, 1], 0.0, 1.0)  # throttle
        action[:, 2] = torch.clamp(action[:, 2], 0.0, 1.0)  # brake

        return (
            action.detach(),
            action_logprob,
        )  # detach if you're not backproping through

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
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = self.fc1(state)
        x = torch.tanh(x)

        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc1")

        x = self.fc2(x)
        x = torch.tanh(x)

        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc2")

        x = self.fc3(x)
        x = torch.tanh(x)

        if not torch.isfinite(x).all():
            raise RuntimeError("NaN/Inf after critic fc3")

        value = self.fc4(x)

        if not torch.isfinite(value).all():
            raise RuntimeError("NaN/Inf in critic output")
        return value


##################################################################################################
#                                       PPO AGENT
##################################################################################################


class PPOAgent:
    def __init__(self, input_dim=198, action_dim=3, summary_writer=None, logger=None):
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
        self.log_probs.append(log_prob)
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
        old_log_probs = np.array(self.log_probs, dtype=np.float32)
        values = np.array(self.vals, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Convert lists to tensors
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        # old_log_probs = torch.tensor(old_log_probs).to(device).detach()
        old_log_probs = (
            torch.tensor(np.array(self.log_probs), dtype=torch.float32)
            .to(device)
            .detach()
            .unsqueeze(1)
        )  # To match the shape of new_probs
        values = torch.tensor(values).to(device).detach()
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)

        rewards = self.normalize_rewards(rewards)

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
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = batch_advantages + values[batch]  # Proper critic target

                # Get new action probabilities and entropy
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = self.critic(batch_states).squeeze()

                # PPO ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

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
                    self.actor.log_std.data[0].clamp_(np.log(0.05), np.log(0.3))
                    self.actor.log_std.data[1:].clamp_(np.log(0.02), np.log(0.25))

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
            current_action_std = torch.exp(self.actor.log_std).mean().item()
            current_log_std = self.actor.log_std.data.cpu().numpy()

            self.logger.info(
                f"[Learn Step {self.learn_step_counter}] log_std: {current_log_std}, action_std: {current_action_std}"
            )

            self.summary_writer.add_scalar(
                "Exploration/learned action std",
                current_action_std,
                self.learn_step_counter,
            )
            self.summary_writer.add_scalar(
                "Exploration/mean log std",
                current_log_std.mean().item(),
                self.learn_step_counter,
            )

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
