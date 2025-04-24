import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class RGBTrackDataset(Dataset):
    def __init__(self, root_dir="data/rgb_finetune/train", split="train"):
        self.root = os.path.join(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        # load filenames + labels
        with open(f"{self.root}/labels.csv") as f:
            lines = f.read().strip().split("\n")[1:]
        self.items = [line.split(",") for line in lines]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, steer = self.items[idx]
        img = Image.open(os.path.join(self.root, fn)).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor([float(steer)], dtype=torch.float32)
