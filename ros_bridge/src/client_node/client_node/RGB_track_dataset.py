import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
from sklearn.model_selection import train_test_split


class RGBTrackDataset(Dataset):
    def __init__(
        self,
        root_dir="/ros_bridge/src/client_node/client_node/data/rgb_finetune/train",
        split="train",
    ):
        self.root = os.path.join(root_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # RGB mean
                ),  # RGB std
            ]
        )
        with open(
            "/ros_bridge/src/client_node/client_node/data/rgb_finetune/train/labels.csv"
        ) as f:
            lines = f.read().strip().split("\n")[1:]
        all_data = [line.split(",") for line in lines]

        # randomize the data
        random.shuffle(all_data)

        # Split once, then save if needed
        trainval, test = train_test_split(all_data, test_size=0.1, random_state=42)
        train, val = train_test_split(trainval, test_size=0.15, random_state=42)

        if split == "train":
            self.items = train
        elif split == "val":
            self.items = val
        else:
            self.items = test

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, steer = self.items[idx]
        img = Image.open(os.path.join(self.root, fn)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([float(steer)], dtype=torch.float32)
