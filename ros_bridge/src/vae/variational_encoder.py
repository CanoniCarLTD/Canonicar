# ros_bridge/src/vae/variational_encoder.py
import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=95):
        super().__init__()
        self.model_file = os.path.join('ros_bridge/src/vae/model', 'var_encoder_model.pth')
        # Conv layers from Idrees’s repo
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # → 80×40
            nn.LeakyReLU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 40×20
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # → 20×10
            nn.LeakyReLU()
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # → 10×5
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10 * 5, 1024),
            nn.LeakyReLU()
        )
        self.mu_layer    = nn.Linear(1024, latent_dims)
        self.logvar_layer= nn.Linear(1024, latent_dims)
        # for true variational sampling if you want:
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
        mu     = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar  # ✅ return both

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
