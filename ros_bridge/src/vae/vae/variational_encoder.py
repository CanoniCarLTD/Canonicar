import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We use this file in vision model file

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=95):
        super().__init__()
        # Path where export_encoder.py will write the encoder-only weights
        self.model_file = "model/vae_encoder_only.pth"
        # conv stack exactly as in train.py
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.LeakyReLU()  # → 80×40
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 40×20
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU()  # → 20×10
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # → 10×5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.linear = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 10 * 5, 1024), nn.LeakyReLU()
        )

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        # standard normal for reparameterization
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.linear(x)

        mu     = self.mu(x)
        logvar = self.sigma(x)

        std = torch.exp(0.5 * logvar)
        z = mu + std * self.N.sample(mu.shape).to(device)
        self.kl = (std**2 + mu**2 - logvar - 1).sum() * 0.5  # ← CORRECT KL

        return z

    def save(self):
        # ensure model folder exists
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
