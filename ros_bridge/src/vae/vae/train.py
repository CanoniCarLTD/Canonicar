import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
import torchvision.transforms as transforms
from PIL import Image

from encoder import VariationalEncoder
from vae.decoder import Decoder

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VAE_DATA_DIR = Path(
    "/ros_bridge/src/data_collector_node/data_collector_node/VAE/images"
)
VAE_LATENT_DIM = 128
VAE_BETA = 0.05  # lower = blurrier recon; higher = latent closer to N(0,1)
VAE_LR = 1e-4
VAE_BATCH_SIZE = 64
VAE_EPOCHS = 20  # bump up once it's stable
VAE_TRAIN_SPLIT = 0.8  # 80% train / 20% val
IMG_SIZE = (80, 160)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "model"
MODEL_FILE = "model/vae_full.pth"


# ─── VARIATIONAL AUTOENCODER ─────────────────────────────────────────────────
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = MODEL_FILE
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(DEVICE)
        z = self.encoder(x)  # must set self.encoder.kl internally
        return self.decoder(z)

    def save(self):
        # Ensure the parent directory exists
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()


# ─── TRAIN & VALIDATION EPOCHS ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(DEVICE)
        x_hat = model(x)
        # ── align spatial dims ────────────────────────
        if x_hat.size(-2) != x.size(-2) or x_hat.size(-1) != x.size(-1):
            x_hat = F.interpolate(
                x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        # ── compute loss ───────────────────────────────
        recon = ((x - x_hat) ** 2).sum()
        kl = model.encoder.kl
        loss = recon + VAE_BETA * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / len(loader.dataset)


def eval_epoch(model, loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            x_hat = model(x)
            if x_hat.size(-2) != x.size(-2) or x_hat.size(-1) != x.size(-1):
                x_hat = F.interpolate(
                    x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            recon = ((x - x_hat) ** 2).sum()
            kl = model.encoder.kl
            loss = recon + VAE_BETA * kl
            total += loss.item()
    return total / len(loader.dataset)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # ── transforms ─────────────────────────────────────────────────────────────
    train_tf = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )

    # ── loader for flat folder ──────────────────────────────────────────────────
    def load_img(path):
        return Image.open(path).convert("RGB")

    full = DatasetFolder(
        root=VAE_DATA_DIR,
        loader=load_img,
        extensions=("png", "jpg", "jpeg"),
        transform=train_tf,
    )

    n = len(full)
    train_n = int(n * VAE_TRAIN_SPLIT)
    val_n = n - train_n
    train_ds, val_ds = random_split(full, [train_n, val_n])

    # override validation transform
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds,
        batch_size=VAE_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAE_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Loaded {n} images → {train_n} train / {val_n} val")
    print(f"Training on device: {DEVICE}")

    model = VariationalAutoencoder(VAE_LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)

    for epoch in range(1, VAE_EPOCHS + 1):
        tl = train_epoch(model, train_loader, optimizer)
        vl = eval_epoch(model, val_loader)
        print(f"Epoch {epoch:02d}/{VAE_EPOCHS}  Train: {tl:.4f}  Val: {vl:.4f}")

    model.save()
    print(f"\n🔥 Done. Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Terminating...")
