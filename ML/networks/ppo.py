import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal

"""
action_dim = Specifies the number of possible actions or the dimension of the 
    action space. In continuous action spaces, action_dim represents the dimensionality of the action vector. For a continuous action space with 2 dimensions (e.g., [steering angle acceleration]): action_dim = 2.
action_std_init = Represents the initial standard deviation of the action 
    distribution, typically used in continuous action spaces. This controls the 
    exploration-exploitation trade-off by adding noise to the action during 
    training. Higher values of action_std_init encourage more exploration by generating more diverse actions.
    (ETAI)
"""


class CustomActivation(nn.Module):
    """Normalize the output values of the actor network. steering [-1, 1], gas and brake [0, 1] (ETAI)"""

    def forward(self, x):
        """Apply tanh for steering (first output), sigmoid for gas and brake (remaining outputs)
        return: torch.tensor: Normalized action values  (ETAI)"""

        print("in CustomActivation forward")

        x[:, 0] = torch.tanh(x[:, 0])  # Steer: [-1, 1]
        x[:, 1] = torch.sigmoid(x[:, 1])  # gas: [0, 1]
        x[:, 2] = torch.sigmoid(x[:, 2])  # Brake: [0, 1]
        return x.detach().numpy().flatten()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create our variable for the matrix.
        # Note that I chose 0.2 for stdev arbitrarily.
        self.cov_var = torch.full((self.action_dim,), action_std_init)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, self.action_dim),
            CustomActivation(),  # Custom activation function for steering, gas, and brake (ETAI)
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

    def forward(self, obs):
        """ABSTRACT - THE METHOD IS INTENDED TO BE OVERRIDDEN BY SUBCLASSES (ETAI)"""
        raise NotImplementedError("Subclasses must implement the forward method.")

    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std)

    def get_value(self, obs):
        """This method returns the value estimate for a given observation.
        The value is computed using the critic network,
        which predicts the expected return from the given state. (ETAI)"""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.critic(
            obs
        )  # THIS LINE IS LITERALLY PASSING THE OBSERVATION IN THE CRITIC NETWORK (ETAI)

    def get_action_and_log_prob(self, obs):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        """This method returns an action sampled from the actor network
        and the log probability of that action. (ETAI)"""

        print("in get_action_and_log_prob")
        print("obs type: ", type(obs))

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        print("going into actor nn")
        # THIS LINE IS LITERALLY PASSING THE OBSERVATION IN THE ACTOR NETWORK (ETAI)
        mean = self.actor(obs)
        print("obs went through actor nn")
        self.cov_mat = torch.diag_embed(self.cov_var.expand_as(mean))

        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        # This introduces an element of randomness,
        # which is crucial for exploration in reinforcement learning. (ETAI)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach(), log_prob.detach()

    def evaluate(self, obs, action):
        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)

        logprobs = torch.clamp(dist.log_prob(action), min=-1e6)
        dist_entropy = dist.entropy()
        values = self.critic(obs)

        return logprobs, values, dist_entropy
