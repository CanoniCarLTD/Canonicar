import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from encoder import VariationalEncoder
from decoder import Decoder

# ─── Config ─────────────────────────────────────────────────────────────
IMG_PATH = "004926.png"
MODEL_FILE = "model/vae_full.pth"
LATENT_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load image ─────────────────────────────────────────────────────────
tf = transforms.Compose([transforms.Resize((80, 160)), transforms.ToTensor()])
img = Image.open(IMG_PATH).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(DEVICE)


# ─── Define VAE (reconstruction only) ───────────────────────────────────
class VAE(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ─── Load full model ────────────────────────────────────────────────────
model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
state = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ─── Reconstruct ────────────────────────────────────────────────────────
with torch.no_grad():
    recon = model(img_tensor)

# ─── Convert to PIL and save ────────────────────────────────────────────
recon_tensor = recon.squeeze(0).clamp(0, 1).cpu()
recon_img = transforms.ToPILImage()(recon_tensor)

original_name = os.path.basename(IMG_PATH)
output_path = f"reconstructed_{original_name}"
recon_img.save(output_path)
print(f"✅ Saved reconstructed image to: {output_path}")

# ─── Visualize ──────────────────────────────────────────────────────────
print("Showing original and reconstructed images...")
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img)
axs[0].set_title("Original")
axs[0].axis("off")
axs[1].imshow(recon_img)
axs[1].set_title("Reconstruction")
axs[1].axis("off")
plt.tight_layout()
plt.show()
