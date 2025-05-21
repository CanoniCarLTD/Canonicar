
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

        # self.log_std = nn.Parameter(
        #     torch.ones(action_dim) * -0.5
        # )
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.0)  # std ~1.0 initially
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # self.fc[-1].bias.data = torch.tensor([0.0, 0.25], dtype=torch.float32)  # steer 0, throttle 0.25
        self.fc[-1].bias.data.zero_()  # unbiased initialization
        
    def forward(self, state):
        return self.fc(state)

    def get_dist(self, state):
        mean = self.forward(state)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist

    def sample_action(self, state):
        dist = self.get_dist(state)
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action)
        steer = action[:, 0:1]  # [-1, 1]
        throttle = (action[:, 1:2] + 1) / 2  # map [-1,1] to [0,1]
        final_action = torch.cat([steer, throttle], dim=1)
        raw_log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = raw_log_prob - squash_correction
        return final_action.detach(), log_prob
    
    def evaluate_actions(self, state, action):
        steer = action[:, 0:1]
        throttle = action[:, 1:2] * 2 - 1  # map [0,1] to [-1,1]
        squashed_action = torch.cat([steer, throttle], dim=1).clamp(-0.999, 0.999)
        raw_action = 0.5 * torch.log((1 + squashed_action) / (1 - squashed_action))
        dist = self.get_dist(state)
        raw_log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        squash_correction = torch.log(1 - squashed_action.pow(2) + 1e-6).sum(
        dim=-1, keepdim=True)
        log_prob = raw_log_prob - squash_correction
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def act_deterministic(self, state: torch.Tensor) -> np.ndarray:
        """
        Given a single state [1xinput_dim], return the *mean* action
        (no noise) as a numpy array [action_dim].
        """
        dist = self.get_dist(state)  # MultivariateNormal
        raw_mean = dist.mean  # shape [1, action_dim]
        normal = Normal(0.0, 1.0)
        cdf_mean = normal.cdf(raw_mean)  # map Gaussian to [0,1]

        # steering in [-1,1], throttle in [0,1]
        steer = 2.0 * cdf_mean[..., 0:1] - 1.0
        throttle = cdf_mean[..., 1:2]

        action = torch.cat([steer, throttle], dim=1)
        return action.detach().cpu().numpy()[0]


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.fc(state)


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

        self.actor = ActorNetwork(self.input_dim, action_dim).to(device)
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
            lp = log_prob.detach()  # shape [1]
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
                torch.load(os.path.join(directory, "actor.pth"), map_location=device)
            )
            self.critic.load_state_dict(
                torch.load(os.path.join(directory, "critic.pth"), map_location=device)
            )
    
            self.actor_optimizer.load_state_dict(
                torch.load(os.path.join(directory, "actor_optim.pth"), map_location=device)
            )
            self.critic_optimizer.load_state_dict(
                torch.load(os.path.join(directory, "critic_optim.pth"), map_location=device)
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
            return (advantages - advantages.mean()) / (advantages.std() + 1e-7)
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
        return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

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

        advantages = self.compute_gae(values, rewards, dones)
        returns = advantages + values

        assert torch.isfinite(advantages).all(), "\nNon-finite advantages!\n"
        assert torch.isfinite(values).all(), "\nNon-finite values!\n"
        assert torch.isfinite(rewards).all(), "\nNon-finite rewards!\n"
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        # Perform PPO optimization steps
        for _ in range(NUM_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(indices), MINIBATCH_SIZE):
                batch = indices[start : start + MINIBATCH_SIZE]

                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch].unsqueeze(1)  # [B,1]
                batch_returns = returns[batch].detach().unsqueeze(1)  # [B,1]

                # ---- Actor update ----
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_states, batch_actions
                )
                new_log_probs = new_log_probs.view(-1, 1)  # ensure shape [B,1]
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
                actor_loss = -(
                    torch.min(surr1, surr2).mean() + self.entropy_coef * entropy.mean()
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                with torch.no_grad():
                    # steer  std ∈ [0.05, 0.7]
                    self.actor.log_std.data[0].clamp_(np.log(0.1), np.log(0.7))
                    # throttle std ∈ [0.05, 1.0]
                    self.actor.log_std.data[1].clamp_(np.log(0.1), np.log(0.9))

                # ---- Critic update ----
                values_pred = self.critic(batch_states).view(-1, 1)
                old_value = (batch_returns - batch_advantages).detach()
                unclipped = F.mse_loss(values_pred, batch_returns)
                clipped = F.mse_loss(
                    old_value + (values_pred - old_value).clamp(-0.2, 0.2),
                    batch_returns,
                )
                critic_loss = VF_COEF * torch.max(unclipped, clipped)

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
                current_action_std = (
                    torch.exp(self.actor.log_std.detach()).cpu().numpy()
                )

                self.logger.info(
                    f"[Learn Step {self.learn_step_counter}] log_std: {current_log_std}, action_std: {current_action_std}"
                )
                # Per-dimension logging
                for i, (ls, std) in enumerate(zip(current_log_std, current_action_std)):
                    self.summary_writer.add_scalar(
                        f"Exploration/log_std_dim_{i}", ls, self.learn_step_counter
                    )
                    self.summary_writer.add_scalar(
                        f"Exploration/std_dim_{i}", std, self.learn_step_counter
                    )
                # Overall logging
                self.summary_writer.add_scalar(
                    "Exploration/mean log std",
                    current_log_std.mean(),
                    self.learn_step_counter,
                )
                self.summary_writer.add_scalar(
                    "Exploration/mean std",
                    current_action_std.mean(),
                    self.learn_step_counter,
                )
                # self.summary_writer.add_scalar("KL/mean_kl_div", kl.item(), self.learn_step_counter)
                self.summary_writer.add_scalar(
                    "Exploration/entropy_coef",
                    self.entropy_coef,
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
