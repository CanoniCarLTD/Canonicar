import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from client_node.rgb_track_dataset import RGBTrackDataset

# 1) model definition
class RGBTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # same slice you use in vision_model
        self.encoder = nn.Sequential(*list(base.features[:10]))
        self.pool    = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(64, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds    = RGBTrackDataset()
    dl    = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    model = RGBTrackModel().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()

    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(5):
        total = 0
        for imgs, steers in dl:
            imgs = imgs.to(device)
            steers = steers.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, steers)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {epoch} ⏿ avg L1 = {total/len(dl):.4f}")
    # 4) save only the encoder part
    torch.save(model.encoder.state_dict(), "checkpoints/mobilenet_trackslice10.pth")

if __name__=="__main__":
    main()
