import torch
from variational_encoder import VariationalEncoder
from decoder import Decoder
import os

# Paths
full_model_path = "model/encoder_decoder.pth"  # use relative path
vae_encoder_out = "model/var_encoder_model.pth"  # destination
LATENT_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the full VAE state_dict
full_state = torch.load(full_model_path, map_location=device)

# Extract only the encoder weights (and remove 'encoder.' prefix)
encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in full_state.items()
    if k.startswith("encoder.")
}

# Init clean encoder
encoder = VariationalEncoder(latent_dims=LATENT_DIM).to(device)
encoder.load_state_dict(encoder_state)
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

# Save only the encoder
os.makedirs("model", exist_ok=True)
torch.save(encoder.state_dict(), vae_encoder_out)
print(f"✅ Exported encoder to: {vae_encoder_out}")