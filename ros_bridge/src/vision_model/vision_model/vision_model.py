import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torchvision.models.resnet import BasicBlock

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


class RGBEncoder(nn.Module):
    def __init__(self, output_features=128):
        super(RGBEncoder, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.mobilenet.features[:10]))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_features)  # Channel count at layer 10
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SensorFusionModel(nn.Module):
    """Model that fuses RGB and LiDAR data for autonomous driving."""
    def __init__(self, rgb_features=128, lidar_features=64, final_features=192):
        super(SensorFusionModel, self).__init__()
        self.rgb_encoder = RGBEncoder(output_features=rgb_features)
        self.lidar_encoder = LiDAREncoder(in_ch=3,output_features=lidar_features)
        
        # Optional: fusion layer to further compress the concatenated features
        self.fusion_layer = nn.Sequential(
            nn.Linear(rgb_features + lidar_features, final_features),
            nn.ReLU()
        )

    def forward(self, rgb_image, lidar_bev):
        """
        Forward pass through the model
        
        Args:
            rgb_image (tensor): RGB image [B, 3, H, W]
            lidar_bev (tensor): LiDAR bird's eye view depth map [B, 2, H, W]
            
        Returns:
            tensor: Fused feature vector
        """
        rgb_features = self.rgb_encoder(rgb_image)
        lidar_features = self.lidar_encoder(lidar_bev)
        
        ####################################################################
        # print(f"[DEBUG] RGB features: mean={rgb_features.mean().item():.4f}, std={rgb_features.std().item():.4f}, min={rgb_features.min().item():.4f}, max={rgb_features.max().item():.4f}")
        # print(f"[DEBUG] LiDAR features: mean={lidar_features.mean().item():.4f}, std={lidar_features.std().item():.4f}, min={lidar_features.min().item():.4f}, max={lidar_features.max().item():.4f}")
        ####################################################################

        # Concatenate features
        fused_features = torch.cat([rgb_features, lidar_features], dim=1)
        
        # print(f"[DEBUG] Fused pre-ReLU: mean={fused_features.mean().item():.4f}, std={fused_features.std().item():.4f}")

        # Optional: further compress features
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
        
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.rgb_tensor = torch.zeros((1, 3, 224, 224), device=device)

        # Create and load the model
        self.model = SensorFusionModel(rgb_features=128, lidar_features=64, final_features=192).to(device)
        self.model.eval()

        # Model warmup for more consistent timing
        dummy_rgb = torch.zeros((1, 3, 224, 224), device=device)
        dummy_lidar = torch.zeros((1, 3, 64, 64), device=device)
        with torch.no_grad():
            self.model(dummy_rgb, dummy_lidar)

        
    def process_rgb(self, rgb_image):
        """Optimized RGB processing with pre-allocated tensors"""
        if isinstance(rgb_image, np.ndarray):
            # Use a more direct conversion path with fewer operations
            if rgb_image.shape[0] == 224 and rgb_image.shape[1] == 224:
                # Already correct size
                rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            else:
                # Resize using OpenCV instead of torch (much faster)
                import cv2
                resized = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_LINEAR)
                rgb_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
                
            # Fast normalization 
            rgb_tensor = rgb_tensor.to(self.device)
            rgb_tensor = (rgb_tensor - self.rgb_mean) / self.rgb_std
            
            return rgb_tensor
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
    
    def process_sensor_data(self, rgb_image, lidar_points):
        """
        Process both RGB and LiDAR data and run inference
        
        Args:
            rgb_image: RGB image as np.array [H, W, 3]
            lidar_points: List of LiDAR points [x, y, z, intensity]
            
        Returns:
            np.array: Fused feature vector
        """
        # Process RGB
        rgb_tensor = self.process_rgb(rgb_image)
        
        # Process LiDAR
        lidar_tensor = self.lidar_to_bev(lidar_points)
        
        # Run inference
        with torch.no_grad():
            features = self.model(rgb_tensor, lidar_tensor)
            
        if not torch.isfinite(features).all():
            raise RuntimeError("NaN/Inf in output fused features")
        
        return features.cpu().numpy()[0]  # Return as numpy array
