# ros_bridge/src/vae/reconstructor.py
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from variational_encoder import VariationalEncoder
from decoder import Decoder  # You’ll need to move Idrees's decoder here
import os

# ─── Config ─────────────────────────────────────────────────────────────
IMG_PATH = "005521.png"

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load image ─────────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((80, 160)),  # match training resolution
    transforms.ToTensor()
])
img = Image.open(IMG_PATH).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(DEVICE)

# Replace your encoder + decoder loads with:
from train import VAE  # or re-define VAE inline here
model = VAE(latent_dim=128).to(DEVICE)
state = torch.load("model/encoder_decoder.pth", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ─── Reconstruct ────────────────────────────────────────────────────────
with torch.no_grad():
    recon, mu, logvar = model(img_tensor)
    
# ─── Convert to PIL and save ────────────────────────────────────────────
recon_tensor = recon.squeeze(0).clamp(0, 1).cpu()
recon_img = transforms.ToPILImage()(recon_tensor)

# Derive save path from original image name
original_name = os.path.basename(IMG_PATH)
output_path = f"reconstructed_{original_name}"
recon_img.save(output_path)
print(f"✅ Saved reconstructed image to: {output_path}")

# ─── Visualize ──────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(img);         axs[0].set_title("Original");   axs[0].axis("off")
axs[1].imshow(recon_img);   axs[1].set_title("Reconstruction"); axs[1].axis("off")
plt.tight_layout(); plt.show()