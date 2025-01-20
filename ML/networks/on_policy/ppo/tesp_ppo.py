import torch
import unittest
import random
from ppo import ActorCritic


class TestActorCritic(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 10
        self.action_dim = 3
        self.action_std_init = 0.2
        self.model = ActorCritic(self.obs_dim, self.action_dim, self.action_std_init)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(
                    m.weight, a=0, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def test_actor_critic_init(self):
        self.assertEqual(self.model.obs_dim, self.obs_dim)
        self.assertEqual(self.model.action_dim, self.action_dim)
        self.assertTrue(
            torch.equal(
                self.model.cov_var, torch.full((self.action_dim,), self.action_std_init)
            )
        )
        self.assertTrue(
            torch.equal(
                self.model.cov_mat, torch.diag(self.model.cov_var).unsqueeze(dim=0)
            )
        )

    def test_actor_output_multiple(self):
        for _ in range(5):
            obs = torch.tensor([[random.uniform(-1, 1) for _ in range(self.obs_dim)]])
            action = self.model.actor(obs)
            print("Steer:", action[0, 0].item())
            print("Gas:", action[0, 1].item())
            print("Brake:", action[0, 2].item())
            self.assertTrue(
                -1 <= action[0, 0].item() <= 1
            )  # Steer should be in [-1, 1]
            self.assertTrue(0 <= action[0, 1].item() <= 1)  # Gas should be in [0, 1]
            self.assertTrue(0 <= action[0, 2].item() <= 1)  # Brake should be in [0, 1]


if __name__ == "__main__":
    unittest.main()
