import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision import transforms


class LiDAREncoder(nn.Module):
    """Lightweight encoder for LiDAR point cloud data converted to a BEV depth map."""
    def __init__(self, input_channels=1, output_features=64):
        super(LiDAREncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_features)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RGBEncoder(nn.Module):
    """Efficient RGB encoder using MobileNetV2 pretrained on ImageNet."""
    def __init__(self, output_features=128):
        super(RGBEncoder, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove classifier, keep only features
        self.encoder = self.mobilenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Last layer of MobileNetV2 features outputs 1280 channels
        self.fc = nn.Linear(1280, output_features)
        
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
        self.lidar_encoder = LiDAREncoder(output_features=lidar_features)
        
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
            lidar_bev (tensor): LiDAR bird's eye view depth map [B, 1, H, W]
            
        Returns:
            tensor: Fused feature vector
        """
        rgb_features = self.rgb_encoder(rgb_image)
        lidar_features = self.lidar_encoder(lidar_bev)
        
        # Concatenate features
        fused_features = torch.cat([rgb_features, lidar_features], dim=1)
        
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
        
        # Create and load the model
        self.model = SensorFusionModel().to(device)
        self.model.eval()
        
    def process_rgb(self, rgb_image):
        """
        Process RGB image to tensor format
        
        Args:
            rgb_image: np.array of shape [H, W, 3] in RGB format
            
        Returns:
            tensor: Processed RGB tensor ready for the model
        """
        # Convert to torch tensor and normalize
        if isinstance(rgb_image, np.ndarray):
            # Convert from [H, W, 3] to [3, H, W]
            # Make sure the array is C-contiguous before converting to tensor
            rgb_image = np.ascontiguousarray(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            
            # Resize to expected input size (224x224 for MobileNetV2)
            rgb_tensor = F.interpolate(rgb_tensor.unsqueeze(0), size=(224, 224), 
                                       mode='bilinear', align_corners=False).squeeze(0)
            
            # Normalize with ImageNet stats
            rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
            rgb_tensor = normalize(rgb_tensor)
            
            return rgb_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        return None
    
    def lidar_to_bev(self, lidar_points):
        """
        Convert LiDAR points to bird's eye view depth map
        
        Args:
            lidar_points: List of [x, y, z] points or np.array of shape [N, 3+]
            
        Returns:
            tensor: BEV representation as [1, 1, H, W] tensor
        """
        if not isinstance(lidar_points, np.ndarray):
            lidar_points = np.array(lidar_points)
            
        # Extract x, y, z coordinates (assuming points are [x, y, z, ...])
        x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]
        
        # Filter points within height range
        mask = (z >= self.height_range[0]) & (z <= self.height_range[1])
        x, y, z = x[mask], y[mask], z[mask]
        
        # Define grid boundaries (assuming points are centered around vehicle)
        grid_range = 50.0  # meters in each direction
        grid_size = self.lidar_grid_size
        
        # Calculate grid indices
        x_indices = ((x + grid_range) * (grid_size[1] / (2 * grid_range))).astype(np.int32)
        y_indices = ((y + grid_range) * (grid_size[0] / (2 * grid_range))).astype(np.int32)
        
        # Filter indices within grid
        mask = (x_indices >= 0) & (x_indices < grid_size[1]) & (y_indices >= 0) & (y_indices < grid_size[0])
        x_indices, y_indices, z = x_indices[mask], y_indices[mask], z[mask]
        
        # Create empty grid
        grid = np.zeros(grid_size, dtype=np.float32)
        
        # Fill grid with max height values
        for i in range(len(x_indices)):
            if grid[y_indices[i], x_indices[i]] < z[i]:
                grid[y_indices[i], x_indices[i]] = z[i]
        
        # Normalize grid values to [0, 1]
        if np.max(grid) > np.min(grid):
            grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
        
        # Convert to tensor
        grid_tensor = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
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
            
        return features.cpu().numpy()[0]  # Return as numpy array
