# export_encoder.py
from encoder import VariationalEncoder
from decoder import Decoder
import torch
import os
from pathlib import Path

LATENT_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define file paths
full_model_path = "model/vae_full.pth"
encoder_out_path = "model/vae_encoder_only.pth"

# Load full model state dict
full_state = torch.load(full_model_path, map_location=device)

# Extract encoder weights only
encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in full_state.items()
    if k.startswith("encoder.")
}

# Load into fresh encoder
encoder = VariationalEncoder(latent_dims=LATENT_DIM).to(device)
encoder.load_state_dict(encoder_state)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

# Save encoder only
torch.save(encoder.state_dict(), encoder_out_path)
print(f"✅ Exported encoder to: {encoder_out_path}")
