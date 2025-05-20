# ros_bridge/src/vae/train.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from variational_encoder import VariationalEncoder
from decoder import Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logvar = self.encoder(x)  # must return both
        std = torch.exp(0.5 * logvar)
        z = mu + std * self.N.sample(std.shape).to(x.device)
        recon = self.decoder(z)
        return recon, mu, logvar

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/ros_bridge/src/data_collector_node/data_collector_node/VAE/images")
LATENT_DIM  = 128
BETA        = 0.05    # lower = blurrier recon; higher = latent closer to N(0,1)
LR          = 3e-4
BATCH_SIZE  = 64
EPOCHS      = 20
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = (80, 160)  # height, width
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# ─── DATA ──────────────────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),        # [0,1]
])

# Load all images from the folder
full_dataset = datasets.ImageFolder(DATA_DIR, transform=tf)

# Split dataset into train and validation sets
train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ─── MODEL & OPT ───────────────────────────────────────────────────────────────
model = VAE(LATENT_DIM).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
mse   = nn.MSELoss(reduction="mean")

def loss_fn(recon, x, mu, logvar):
    # reconstruction + β·KL
    r_loss = mse(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return r_loss + BETA * kl, r_loss, kl

# ─── TRAIN/VAL LOOP ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(DEVICE)
            recon, mu, logvar = model(imgs)
            loss, r_l, kl_l = loss_fn(recon, imgs, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                recon, mu, logvar = model(imgs)
                loss, _, _ = loss_fn(recon, imgs, mu, logvar)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train L={train_loss:.6f} | Val L={val_loss:.6f}")
        # brutal checkpoint: only save if it actually improves
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "model/encoder_decoder.pth")
            print(f"  → saved best model (val {best_val:.6f})")

    # ─── AFTER TRAINING ─────────────────────────────────────────────────────────────
    print("\n🔥 Training complete.")
    print("  • Final model: vae/encoder_decoder.pth")
    print("  • Next: run export_encoder.py to peel off the encoder for PPO.")