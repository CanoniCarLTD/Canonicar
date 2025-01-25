import open3d as o3d
import torch
import torch.nn as nn
import numpy as np


class PointCloudFeatureExtractor(nn.Module):
    def __init__(self, output_size=120):
        super(PointCloudFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, point_cloud):
        # Point-wise features
        point_features = self.mlp(point_cloud)  # Shape: (N, 256)
        # Global pooling
        global_features = self.global_pooling(point_features.T.unsqueeze(0)).squeeze()  # Shape: (256,)
        # Reduce to fixed output size
        output_vector = self.fc(global_features)  # Shape: (output_size,)
        return output_vector


# Load point cloud using open3d
def load_point_cloud(file_path):
    # Check file extension to decide format
    file_extension = file_path.split('.')[-1].lower()

    # Read the point cloud
    if file_extension == 'pcd':
        pcd = o3d.io.read_point_cloud(file_path, format="pcd")
    elif file_extension == 'ply':
        pcd = o3d.io.read_point_cloud(file_path, format="ply")
    else:
        raise ValueError("Unsupported file format: {}".format(file_extension))

    # Convert the point cloud data to numpy array
    points = np.asarray(pcd.points)  # Shape: (N, 3) -> x, y, z

    # Add intensity (I) if not present (for PCD files, intensity may be absent)
    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1))  # Default intensity as 0
        points = np.hstack((points, intensity))  # Shape: (N, 4)

    return points


# Example usage
if __name__ == "__main__":
    # Path to your .ply or .pcd file
    file_path = "418.ply"  # Replace with your file path

    # Load point cloud
    point_cloud_data = load_point_cloud(file_path)
    print("Point cloud shape:", point_cloud_data.shape)  # Should be (N, 4)

    # Convert to PyTorch tensor
    point_cloud_tensor = torch.tensor(point_cloud_data, dtype=torch.float32)

    # Create the model
    model = PointCloudFeatureExtractor(output_size=120)

    # Feed the point cloud to the model
    output_vector = model(point_cloud_tensor)
    print("Output vector shape:", output_vector.shape)  # Should be (120,)
