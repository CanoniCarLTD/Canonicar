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

        # Fixed diagonal covariance; registered so .to(device) moves it.
        self.register_buffer("cov_var", torch.full((action_dim,), float(action_std_init)))
        self.register_buffer("cov_mat", torch.diag(self.cov_var).unsqueeze(dim=0))  # [1,A,A]

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
        # updates registered buffers
        with torch.no_grad():
            self.cov_var.fill_(float(new_action_std))
            self.cov_mat.copy_(torch.diag(self.cov_var).unsqueeze(0))

    @torch.no_grad()
    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        return self.critic(obs)

    @torch.no_grad()
    def get_action_and_log_prob(self, obs):
        # mean in [-1,1] due to tanh head
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=device)
        mean = self.actor(obs)
        cov = self.cov_mat.expand(mean.shape[0], self.action_dim, self.action_dim)
        dist = MultivariateNormal(mean, cov)
        action = dist.sample()           # NOT squashed; can exceed [-1,1]
        log_prob = dist.log_prob(action) # scalar per batch element
        return action, log_prob

    def evaluate(self, obs, action):
        # action is expected in model domain [-1,1]
        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        logprobs = dist.log_prob(action)     # [B]
        dist_entropy = dist.entropy()        # [B]
        values = self.critic(obs)            # [B,1]
        return logprobs, values, dist_entropy


##################################################################################################
#                                       PPO AGENT
##################################################################################################

class PPOAgent:
    def __init__(self, input_dim=197, action_dim=2, summary_writer=None, logger=None):
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

        # === unified Idrees-style module + frozen copy for sampling (old policy) ===
        action_std_init = float(globals().get("ACTION_STD_INIT", 0.2))
        self.ac = ActorCritic(self.input_dim, self.action_dim, action_std_init).to(device)
        self.old_ac = ActorCritic(self.input_dim, self.action_dim, action_std_init).to(device)
        self.old_ac.load_state_dict(self.ac.state_dict())

        # Keep separate optimizers (actor/critic) like before
        self.actor_optimizer = optim.Adam(self.ac.actor.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.ac.critic.parameters(), lr=CRITIC_LEARNING_RATE)

        # Experience storage (unchanged)
        (
            self.states,
            self.actions,
            self.log_probs,
            self.vals,
            self.rewards,
            self.dones,
        ) = ([], [], [], [], [], [])

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

    def select_action(self, state):
        """Select an action and return its log probability (from frozen old policy)."""
        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if st.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, but got {st.shape[1]}"
            )

        if torch.isnan(st).any():
            raise ValueError(f"NaN detected in input state: {st}")

        with torch.no_grad():
            # Sample in model domain (can exceed [-1,1] because MVN is unbounded)
            model_action, log_prob = self.old_ac.get_action_and_log_prob(st)
            # ✅ Clamp to keep PPO math consistent with env execution
            model_action = model_action.clamp(-1.0, 1.0)

            # Map to environment semantics: steer∈[-1,1], throttle∈[0,1]
            steer = model_action[:, 0:1]
            throttle01 = (model_action[:, 1:2] + 1.0) / 2.0
            env_action = torch.cat([steer, throttle01], dim=1)

        return env_action.cpu().numpy()[0], log_prob.unsqueeze(0)  # keep [1]-like before

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
        self.logger.info("Saving model + optimizer (ActorCritic unified)...")
        try:
            os.makedirs(directory, exist_ok=True)

            # Unified AC file instead of separate actor/critic files.
            torch.save(self.ac.state_dict(), os.path.join(directory, "ac.pth"))

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

    def old_load_model_and_optimizers(self, directory):
        # WARNING: legacy API retained, but we now use a single ac.pth
        self.logger.info(f"Loading model + optimizer from: {directory}")
        try:
            self.ac.load_state_dict(
                torch.load(os.path.join(directory, "ac.pth"), map_location=device)
            )
            self.old_ac.load_state_dict(self.ac.state_dict())

            self.actor_optimizer.load_state_dict(
                torch.load(
                    os.path.join(directory, "actor_optim.pth"), map_location=device
                )
            )
            self.critic_optimizer.load_state_dict(
                torch.load(
                    os.path.join(directory, "critic_optim.pth"), map_location=device
                )
            )

            self.logger.info("Model and optimizer states loaded successfully.")
        except Exception as e:
            self.logger.info(f"❌ Error loading model and optimizer: {e}")

    def load_model_and_optimizers(self, directory):
        """Load model weights/optimizers.

        NOTE: Previously supported flat checkpoints with 'actor.'/'critic.' prefixes.
        Now we expect a single 'ac.pth'. If you need backward compatibility,
        add a translator here—do NOT guess formats.
        """
        return self.old_load_model_and_optimizers(directory)

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

    def compute_gae(self, rewards, values, dones, gamma=GAMMA, lam=LAMBDA_GAE):
        """
        rewards, values, dones are 1-D tensors of length T
        values already contains V(s_t) for t=0..T  (last extra element is V(s_T) or 0)
        returns both advantage and return tensors of length T
        """
        T = rewards.size(0)
        advantages = torch.zeros(T, device=device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_nonterminal - values[t]
            lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values[:-1]  # drop the bootstrap value
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

    def learn(self):
        """
        Perform PPO training using stored experiences (from latest batch).
        returns: actor_loss, critic_loss, entropy
        """
        # Convert lists to numpy arrays first
        states = torch.as_tensor(
            np.array(self.states), dtype=torch.float32, device=device
        )
        env_actions = torch.as_tensor(
            np.array(self.actions), dtype=torch.float32, device=device
        )  # steer[-1,1], throttle[0,1]
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=device)

        # fresh value estimates (no need to store them in memory)
        with torch.no_grad():
            values_now = self.ac.get_value(states).squeeze(-1)  # [T]
            last_value = (
                torch.zeros((), device=device)
                if dones[-1]
                else self.ac.get_value(states[-1:]).squeeze()
            )

        values = torch.cat([values_now, last_value.view(1)], dim=0)  # [T+1]

        advantages, returns = self.compute_gae(rewards, values, dones)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        old_log_probs = torch.cat(self.log_probs, dim=0).detach().view(-1, 1).to(device)

        # Map env actions back to model domain for likelihood evaluation
        steer = env_actions[:, 0:1]                     # already in [-1,1]
        throttle_m1_1 = env_actions[:, 1:2] * 2.0 - 1.0 # [0,1] -> [-1,1]
        model_actions = torch.cat([steer, throttle_m1_1], dim=1)
        # ✅ Clamp to keep training eval consistent with acting & env execution
        model_actions = model_actions.clamp(-1.0, 1.0)

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
                batch = indices[start : start + MINIBATCH_SIZE]

                batch_states = states[batch]
                batch_actions_model = model_actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_advantages = advantages[batch].unsqueeze(1).detach()  # [B,1]
                batch_returns = returns[batch].detach().unsqueeze(1)        # [B,1]

                # ---- Actor update (new policy) ----
                new_log_probs, _, entropy = self.ac.evaluate(
                    batch_states, batch_actions_model
                )
                new_log_probs = new_log_probs.view(-1, 1)  # [B,1]

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
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # ---- Critic update ----
                values_pred = self.ac.critic(batch_states).view(-1, 1)
                critic_loss = F.mse_loss(values_pred, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        # Sync frozen policy with current (Idrees-style)
        self.old_ac.load_state_dict(self.ac.state_dict())
        self.learn_step_counter += 1

        if self.summary_writer is not None:
            with torch.no_grad():
                # NOTE: no per-dim log_std anymore; we expose fixed std from cov_var.
                current_action_std = self.ac.cov_var.detach().cpu().numpy()
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
            self.vals,
            self.rewards,
            self.dones,
        ) = ([], [], [], [], [], [])

        # Return averaged metrics
        return (
            total_actor_loss / max(1, num_batches),
            total_critic_loss / max(1, num_batches),
            total_entropy / max(1, num_batches),
        )

    # === Optional: keep API symmetry and expose std control like Idrees ===
    def set_action_std(self, new_action_std: float):
        """Manually set fixed exploration std (Idrees-style)."""
        self.ac.set_action_std(new_action_std)
        self.old_ac.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        """Linearly decay fixed std. No guessing beyond linear clip."""
        cur = float(self.ac.cov_var[0].item())
        cur = max(min_action_std, cur - action_std_decay_rate)
        self.set_action_std(cur)
        return cur
