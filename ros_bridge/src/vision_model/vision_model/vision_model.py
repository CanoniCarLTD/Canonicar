import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torchvision.models.resnet import BasicBlock
import cv2
from vae.variational_encoder import VariationalEncoder

class LiDAREncoder(nn.Module):
    def __init__(self, in_ch=3, output_features=64):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            BasicBlock(32, 32)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            BasicBlock(64, 64)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            BasicBlock(128, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, output_features)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SemanticEncoder(nn.Module):
    def __init__(self, output_features=128):
        super(SemanticEncoder, self).__init__()
        # Input is 3-channel BGR from the semantic segmentation
        # Use a more efficient network for semantic images since they're already preprocessed
        self.encoder = nn.Sequential(
            # Initial block
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 1
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_features)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SensorFusionModel(nn.Module):
    """Model that fuses Semantic and LiDAR data for autonomous driving."""
    def __init__(self, latent_dim=128, lidar_features=64, final_features=192):
        super(SensorFusionModel, self).__init__()
        # self.semantic_encoder = SemanticEncoder(output_features=semantic_features)
        self.vae_encoder = VariationalEncoder(latent_dims=latent_dim)
        # freeze weights in case someone accidentally trains it
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        self.lidar_encoder = LiDAREncoder(in_ch=3, output_features=lidar_features)
        # Fusion layer to combine the features
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim + lidar_features, final_features),
            nn.ReLU()
        )

    def forward(self, semantic_image, lidar_bev):
        """
        Forward pass through the model
        
        Args:
            semantic_image (tensor): Semantic segmentation image [B, 3, H, W]
            lidar_bev (tensor): LiDAR bird's eye view depth map [B, 3, H, W]
            
        Returns:
            tensor: Fused feature vector
        """
        semantic_features = self.vae_encoder(semantic_image)
        lidar_features = self.lidar_encoder(lidar_bev)
        
        # Concatenate features
        fused_features = torch.cat([semantic_features, lidar_features], dim=1)
        
        # Further compress features
        fused_features = self.fusion_layer(fused_features)
        
        return fused_features
 
    
    
class VisionProcessor:
    """Helper class to process raw sensor data before feeding to the model."""
    def __init__(self, lidar_grid_size=(64, 64), height_range=(-2.0, 4.0), device='cpu'):
        """
        Initialize the vision processor
        
        Args:
            lidar_grid_size (tuple): Size of the BEV grid for LiDAR (H, W)
            height_range (tuple): Min and max height for LiDAR points
            device: Torch device to run on
        """
        self.lidar_grid_size = lidar_grid_size
        self.height_range = height_range
        self.device = device
        
        # No need for RGB normalization since semantic segmentation is already in a standardized format
        # But we'll still preprocess to normalize to [0, 1] range
        self.semantic_tensor = torch.zeros((1, 3, 224, 224), device=device)

        # Create and load the model
        self.model = SensorFusionModel(semantic_features=128, lidar_features=64, final_features=192).to(device)
        self.model.eval()
        
        VAE_PATH = "ros_bridge/src/vae/model/var_encoder_model.pth"
        self.model.vae_encoder.load()      # uses our .load() helper
        
        # Model warmup for more consistent timing
        dummy_semantic = torch.zeros((1, 3, 224, 224), device=device)
        dummy_lidar = torch.zeros((1, 3, 64, 64), device=device)
        with torch.no_grad():
            self.model(dummy_semantic, dummy_lidar)

        
    def process_semantic(self, semantic_image):
        """Process semantic segmentation image"""
        if isinstance(semantic_image, np.ndarray):
            # Resize using OpenCV for better performance
            if semantic_image.shape[0] != 224 or semantic_image.shape[1] != 224:
                resized = cv2.resize(semantic_image, (224, 224), interpolation=cv2.INTER_NEAREST)
            else:
                resized = semantic_image
                
            # Convert to tensor [0, 1] range
            semantic_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            semantic_tensor = semantic_tensor.to(self.device)
            
            return semantic_tensor
        return None
    
    def lidar_to_bev(self, lidar_points):
        """
        Convert LiDAR points to bird's eye view depth map with vectorized operations
        """
        if not isinstance(lidar_points, np.ndarray):
            lidar_points = np.array(lidar_points)
            
        # Extract x, y, z, intensity coordinates
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        intensities = lidar_points[:, 3] if lidar_points.shape[1] > 3 else np.ones_like(x)
        
        # Filter points within height range
        mask = (z >= self.height_range[0]) & (z <= self.height_range[1])
        x, y, z, intensities = x[mask], y[mask], z[mask], intensities[mask]
        
        # Define grid boundaries
        grid_range = 50.0
        grid_size = self.lidar_grid_size
        
        # Calculate grid indices
        x_indices = ((x + grid_range) * (grid_size[1] / (2 * grid_range))).astype(np.int32)
        y_indices = ((y + grid_range) * (grid_size[0] / (2 * grid_range))).astype(np.int32)
        
        # Filter indices within grid
        mask = (x_indices >= 0) & (x_indices < grid_size[1]) & (y_indices >= 0) & (y_indices < grid_size[0])
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
        grid_density_flat = np.bincount(flat_indices, minlength=grid_size[0]*grid_size[1])
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
    
    def process_sensor_data(self, semantic_image, lidar_points):
        """
        Process both semantic segmentation and LiDAR data and run inference
        
        Args:
            semantic_image: Semantic segmentation image as np.array [H, W, 3]
            lidar_points: List of LiDAR points [x, y, z, intensity]
            
        Returns:
            np.array: Fused feature vector
        """
        # Process semantic image
        semantic_tensor = self.process_semantic(semantic_image)
        
        # Process LiDAR
        lidar_tensor = self.lidar_to_bev(lidar_points)
        
        # Run inference
        with torch.no_grad():
            features = self.model(semantic_tensor, lidar_tensor)
            
        
        return features.cpu().numpy()[0]  # Return as numpy array
