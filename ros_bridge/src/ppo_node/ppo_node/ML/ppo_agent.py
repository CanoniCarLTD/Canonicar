# PPO agent:

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

# NOTE:
# - Idrees-style unified ActorCritic:
#     * actor: MLP with Tanh head -> mean in [-1,1]^action_dim
#     * critic: MLP -> V(s)
#     * fixed diagonal covariance (no learnable log_std)
#     * MultivariateNormal for sampling / logprob
# - Explicit sampling/eval methods; forward() left unimplemented on purpose.

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init: float):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # in ActorCritic.__init__(...), replace the two register_buffer lines:
        self.cov_var = torch.full((self.action_dim,), action_std_init, device=device)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).to(device)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, self.action_dim),
            nn.Tanh(),  # outputs in [-1,1]
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self):
        raise NotImplementedError  # defensive: use explicit methods below

    def set_action_std(self, new_action_std: float):
        self.cov_var = torch.full((self.action_dim,), new_action_std, device=device)

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=device)
        return self.critic(obs)

    def get_action_and_log_prob(self, obs):
        # mean in [-1,1] due to tanh head
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=device)
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach()

    def evaluate(self, obs, action):
        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        return logprobs, values, dist_entropy


##################################################################################################
#                                       PPO AGENT
##################################################################################################

class PPOAgent:
    def __init__(self, input_dim=100, action_dim=2, summary_writer=None, logger=None):
        self.logger = logger
        if self.logger is None:
            raise ValueError("Logger not provided. Please provide a logger instance.")
        self.logger.info("Initializing PPO Agent (Idrees-style policy/logprob/cov)...")
        self.logger.info(f"device: {device}")

        self.input_dim = input_dim if PPO_INPUT_DIM is None else PPO_INPUT_DIM
        self.logger.info(f"Action input dimension: {self.input_dim}")
        self.action_dim = action_dim
        if self.action_dim != 2:
            # FATAL: this code assumes [steer, throttle] exactly
            raise ValueError("action_dim must be 2 (steer[-1,1], throttle[0,1]).")
        self.logger.info(f"Action output dimension: {self.action_dim}")

        self.summary_writer = summary_writer
        self.entropy_coef = ENTROPY_COEF
        self.lr = PPO_LEARNING_RATE

        # === unified Idrees-style module + frozen copy for sampling (old policy) ===
        action_std_init = ACTION_STD_INIT
        self.policy = ActorCritic(self.input_dim, self.action_dim, action_std_init)
        self.policy.to(device)

        self.action_std = action_std_init
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr}])
        
        self.old_policy = ActorCritic(self.input_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.to(device)
        self.MseLoss = nn.MSELoss()
        
        self.set_action_std(ACTION_STD_INIT)
        
        # Experience storage (unchanged)
        (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
        ) = ([], [], [], [], [])

        self.learn_step_counter = 0  # Track PPO updates

        if not os.path.exists(PPO_CHECKPOINT_DIR):
            os.makedirs(PPO_CHECKPOINT_DIR)

        # WARNINGS about metrics differences vs old implementation
        self.logger.warning(
            "Switched to Idrees-style log-probabilities (no tanh squash correction). "
            "Logged log_probs/entropy are NOT numerically comparable with the previous actor that used squash-correction."
        )
        self.logger.warning(
            "Exploration std is FIXED unless you call set_action_std/decay_action_std. "
            "Entropy will not naturally anneal from a learnable log_std anymore."
        )

        self.logger.info("PPO Agent initialized.")

    ##################################################################################################
    #                                        SELECT ACTION
    ##################################################################################################

    def select_action(self, obs):
        """Select an action and return its log probability (from frozen old policy)."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))

        return action.detach().cpu().numpy().flatten(), logprob.detach().cpu().numpy().flatten()

    ##################################################################################################
    #                                     STORE LAST EXPERIENCE
    ##################################################################################################

    def store_transition(self, state, action, log_prob, reward, done):
        self.states.append(state)

        # action arrives as [steer in -1..1, throttle in 0..1]; convert to model domain
        a = np.asarray(action, dtype=np.float32).copy()
        self.actions.append(a)

        if not isinstance(log_prob, torch.Tensor):
            lp = torch.tensor([log_prob], dtype=torch.float32, device=device)
        else:
            lp = log_prob.detach()
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.dones.append(done)

    ##################################################################################################
    #                                       SAVE AND LOAD MODELS
    ##################################################################################################

    def save_model_and_optimizers(self, directory):
        # Simplified save: only save the policy (ActorCritic state) to a single file
        # named `ppo_policy_12_.pth`. Do NOT save optimizer state.
        self.logger.info("Saving policy only (ppo_policy_12_.pth) - no optimizers...")
        try:
            os.makedirs(directory, exist_ok=True)
            out_path = os.path.join(directory, "ppo_policy_12_.pth")
            # Save a dict with a clear key for forward compatibility
            torch.save({"policy": self.policy.state_dict()}, out_path)
            self.logger.info(f"Policy saved to {out_path}")
        except Exception as e:
            self.logger.info(f"❌ Error saving policy: {e}")

    def load_model_and_optimizers(self, directory):
        """Load policy saved by `save_model_and_optimizers`.

        Expect a file named `ppo_policy_12_.pth` inside `directory`. The file
        should contain either a raw state_dict or a dict with key 'ac'.
        This function only loads the model weights (no optimizers).
        """
        self.logger.info(f"Loading policy from: {directory}")
        try:
            p = os.path.join(directory, "ppo_policy_12_.pth")
            checkpoint = torch.load(p, map_location=device)

            # Support either {'ac': ...}, {'policy': ...}, or a raw state_dict
            if isinstance(checkpoint, dict) and ("ac" in checkpoint or "policy" in checkpoint):
                model_state = checkpoint.get("policy", checkpoint.get("ac"))
            else:
                model_state = checkpoint

            try:
                self.policy.load_state_dict(model_state, strict=False)
            except Exception as e_load:
                self.logger.warning(f"Strict load failed: {e_load}; retrying with strict=False")
                self.policy.load_state_dict(model_state, strict=False)
            # Ensure models are on the correct device after loading
            self.policy.to(device)

            # Sync frozen policy
            self.old_policy.load_state_dict(self.policy.state_dict())
            self.old_policy.to(device)
            self.logger.info("Policy loaded into ActorCritic.")
        except Exception as e:
            self.logger.info(f"❌ Error loading policy: {e}")

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
    #                                           COMPUTE monte-carlo returns
    ##################################################################################################

    def compute_mc_returns(self, rewards, dones, gamma=GAMMA):
        T = rewards.size(0)
        returns = torch.zeros(T, device=device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G * (1.0 - dones[t])
            returns[t] = G
        advantages = returns - self.policy.get_value(self.states_tensor)[...,0].detach()  # or recompute values_now
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    ##################################################################################################
    #                                       NORMALIZE REWARDS
    ##################################################################################################

    def normalize_rewards(self, rewards):
        """Normalize rewards before training."""
        return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    ##################################################################################################
    #                                       MAIN PPO ALGORITHM
    ##################################################################################################

    def learn(self): # **FIX LOOP - REMOVE MINIBATCHES AND FIX MONTE CARLO RETURNS**
        """
        Perform PPO training using stored experiences (from latest batch).
        returns: actor_loss, critic_loss, entropy
        """
        # Convert lists to numpy arrays first
        states = torch.as_tensor(
            np.array(self.states), dtype=torch.float32, device=device
        )
        # expose states as an attribute for compatibility with compute_mc_returns
        # compute_mc_returns expects self.states_tensor to exist when using Monte-Carlo returns
        self.states_tensor = states
        model_actions = torch.as_tensor(
            np.array(self.actions), dtype=torch.float32, device=device
        )
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=device)

        advantages, returns = self.compute_mc_returns(rewards, dones)

        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        old_log_probs = torch.cat(self.log_probs, dim=0).detach().view(-1, 1).to(device)


        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        # Perform PPO optimization steps
        N = len(states)
        for _ in range(NUM_EPOCHS):
            indices = np.arange(N)
            np.random.shuffle(indices)

            for start in range(0, N, MINIBATCH_SIZE):
                b = indices[start:start + MINIBATCH_SIZE]

                # Slice and fix shapes to [B,1] where needed
                b_states      = states[b]                                  # [B, S]
                b_actions     = model_actions[b]                           # [B, A]  (MODEL domain: steer[-1,1], throttle[-1,1])
                b_old_logprob = old_log_probs[b].detach().view(-1, 1)      # [B, 1]
                b_adv         = advantages[b].detach().view(-1, 1)         # [B, 1]  (already normalized; keep as-is)
                b_returns     = returns[b].detach().view(-1, 1)            # [B, 1]  (DO NOT normalize)

                # ---- Forward new policy ----
                # evaluate() must return: logprobs [B], values [B,1], entropy [B] (or broadcastable)
                new_logprob, values, dist_entropy = self.policy.evaluate(b_states, b_actions)
                new_logprob = new_logprob.view(-1, 1)                      # [B,1]

                # ---- PPO ratio ----
                ratio = torch.exp(new_logprob - b_old_logprob)             # [B,1]

                # ---- Clipped surrogate ----
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - POLICY_CLIP, 1 + POLICY_CLIP) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # ---- Critic loss (no return normalization) ----
                critic_loss = F.mse_loss(values, b_returns) * VF_COEF

                # ---- Entropy bonus as a separate term ----
                entropy_loss = -self.entropy_coef * dist_entropy.mean()

                # ---- Joint backward/update ----
                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
                num_batches += 1


        # Sync frozen policy with current (Idrees-style)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.learn_step_counter += 1

        if self.summary_writer is not None:
            with torch.no_grad():
                # NOTE: no per-dim log_std anymore; we expose fixed std from cov_var.
                current_action_std = self.policy.cov_var.detach().cpu().numpy()
                self.logger.info(
                    f"[Learn Step {self.learn_step_counter}] fixed action_std per dim: {current_action_std}"
                )
                for i, std in enumerate(current_action_std):
                    self.summary_writer.add_scalar(
                        f"Exploration/std_dim_{i}", std, self.learn_step_counter
                    )
                self.summary_writer.add_scalar(
                    "Exploration/mean std",
                    current_action_std.mean(),
                    self.learn_step_counter,
                )
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
            self.rewards,
            self.dones,
        ) = ([], [], [], [], [])

        # Return averaged metrics
        return (
            total_actor_loss / max(1, num_batches),
            total_critic_loss / max(1, num_batches),
            total_entropy / max(1, num_batches),
        )

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std
