import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torchvision.models.resnet import BasicBlock
import cv2
import os
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LiDAREncoder(nn.Module):
    def __init__(self, in_ch=3, output_features=64):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            BasicBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BasicBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, output_features)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()

        self.model_file = 'model/var_encoder_model.pth'

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79, 39
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # 19, 9
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9*4*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def save(self):
        # ensure model folder exists
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except:
            print('Encoder could not be initialized.')
            sys.exit()
    
    def process(self, observation):
        image_obs = torch.tensor(observation, dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        
        # navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        # observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        image_obs = image_obs.view(-1)
        return image_obs


class VisionProcessor:
    """Helper class to process raw sensor data before feeding to the model."""

    def __init__(
        self, height_range=(-2.0, 4.0), device="cpu"
    ):
        """
        Initialize the vision processor

        Args:
            lidar_grid_size (tuple): Size of the BEV grid for LiDAR (H, W)
            height_range (tuple): Min and max height for LiDAR points
            device: Torch device to run on
        """
        # self.lidar_grid_size = lidar_grid_size
        # self.height_range = height_range
        self.device = device

        # No need for RGB normalization since semantic segmentation is already in a standardized format
        # But we'll still preprocess to normalize to [0, 1] range
        self.semantic_tensor = torch.zeros((1, 3, 160, 80), device=self.device)

        # Create and load the model
        self.model = EncodeState(
            latent_dim=95
        ).to(self.device)


    def lidar_to_bev(self, lidar_points):
        """
        Convert LiDAR points to bird's eye view depth map with vectorized operations
        """
        if not isinstance(lidar_points, np.ndarray):
            lidar_points = np.array(lidar_points)

        # Extract x, y, z, intensity coordinates
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        intensities = (
            lidar_points[:, 3] if lidar_points.shape[1] > 3 else np.ones_like(x)
        )

        # Filter points within height range
        mask = (z >= self.height_range[0]) & (z <= self.height_range[1])
        x, y, z, intensities = x[mask], y[mask], z[mask], intensities[mask]

        # Define grid boundaries
        grid_range = 50.0
        grid_size = self.lidar_grid_size

        # Calculate grid indices
        x_indices = ((x + grid_range) * (grid_size[1] / (2 * grid_range))).astype(
            np.int32
        )
        y_indices = ((y + grid_range) * (grid_size[0] / (2 * grid_range))).astype(
            np.int32
        )

        # Filter indices within grid
        mask = (
            (x_indices >= 0)
            & (x_indices < grid_size[1])
            & (y_indices >= 0)
            & (y_indices < grid_size[0])
        )
        x_indices, y_indices = x_indices[mask], y_indices[mask]
        x, y, z, intensities = x[mask], y[mask], z[mask], intensities[mask]

        # Pre-compute all distances once (faster than computing in loop)
        distances = np.sqrt(x**2 + y**2)

        # Initialize grids - combined approach with vectorized operations where possible
        grid_occupancy = np.zeros(grid_size, dtype=np.float32)
        grid_distance = np.full(grid_size, 50.0, dtype=np.float32)

        # Create flat indices for faster operations
        flat_indices = y_indices * grid_size[1] + x_indices

        # For density, use numpy's bincount for faster counting
        grid_density_flat = np.bincount(
            flat_indices, minlength=grid_size[0] * grid_size[1]
        )
        grid_density = grid_density_flat.reshape(grid_size)

        # Create a weighted occupancy that highlights closer barriers
        distance_weights = 1.0 - (distances / 50.0)
        distance_weights = np.clip(distance_weights, 0.1, 1.0)

        # Use a combination of vectorized operations and a loop for occupancy and distance
        for i in range(len(flat_indices)):
            idx = flat_indices[i]
            y_idx, x_idx = y_indices[i], x_indices[i]

            # Set weighted occupancy based on distance (closer = higher weight)
            if grid_occupancy[y_idx, x_idx] < distance_weights[i]:
                grid_occupancy[y_idx, x_idx] = distance_weights[i]

            # Update minimum distance
            if grid_distance[y_idx, x_idx] > distances[i]:
                grid_distance[y_idx, x_idx] = distances[i]

        # Normalize and invert distance
        grid_distance = 1.0 - (grid_distance / 50.0)

        # Normalize density
        max_density = np.max(grid_density)
        if max_density > 0:
            grid_density = grid_density / max_density

        # Stack channels
        grid_combined = np.stack([grid_occupancy, grid_distance, grid_density], axis=0)

        # Convert to tensor [1, 3, H, W]
        grid_tensor = torch.from_numpy(grid_combined).float().unsqueeze(0)
        return grid_tensor.to(self.device)

