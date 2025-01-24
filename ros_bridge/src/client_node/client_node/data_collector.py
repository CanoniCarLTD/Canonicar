import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import carla
import json
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from std_msgs.msg import Header
import numpy as np
import struct
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

class DataCollector(Node):
    def __init__(self, carla_client=None, carla_world=None):
        super().__init__('data_collector')

        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 2000)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value

        # Use provided CARLA client and world, or connect if not provided
        if carla_client and carla_world:
            self.client = carla_client
            self.world = carla_world
        else:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

        self.data_buffer = []  # List of dictionaries to store synchronized data
        self.setup_subscribers()
        self.vision_model = self.initialize_vision_model()

    def setup_subscribers(self):
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT  # Prioritize speed over reliability
        )

        self.image_sub = Subscriber(self, Image, "/carla/ego_vehicle/rgb_front/image_raw", qos)
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/ego_vehicle/lidar/points", qos)
        self.imu_sub = Subscriber(self, Imu, "/carla/ego_vehicle/imu", qos)
        self.gnss_sub = Subscriber(self, NavSatFix, "/carla/ego_vehicle/gnss", qos)

        self.ats = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.imu_sub, self.gnss_sub],
            queue_size=10,
            slop=0.01  # Reduce slop to improve synchronization speed
        )
        self.ats.registerCallback(self.sync_callback)

    def initialize_vision_model(self):
        """Initialize a convolutional model (e.g., MobileNetV2) to process images using PyTorch."""
        model = mobilenet_v2(pretrained=True)
        model = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return model

    def sync_callback(self, image_msg, lidar_msg, imu_msg, gnss_msg):
        processed_data = self.process_data(image_msg, lidar_msg, imu_msg, gnss_msg)
        self.data_buffer.append(processed_data)

    def process_data(self, image_msg, lidar_msg, imu_msg, gnss_msg):
        return self.aggregate_state_vector(
            self.process_image(image_msg),
            self.process_lidar(lidar_msg),
            self.process_imu(imu_msg),
            self.process_gnss(gnss_msg)
        )

    def process_image(self, image_msg):
        """Process camera image data and extract convolutional features using PyTorch."""
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((image_msg.height, image_msg.width, -1))
        input_image = self.transform(raw_image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.vision_model(input_image).squeeze(0).numpy()  # Extract features
        return features[:20]  # Return 20 feature cells

    def process_lidar(self, lidar_msg):
        """Process LiDAR data."""
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack('ffff', lidar_msg.data)
        ]
        mean_height = np.mean([p[2] for p in points]) if points else 0
        density = len(points) / 100  # Normalize for density
        return [mean_height, density] + points[:13]  # Up to 15 features

    def process_imu(self, imu_msg):
        """Process IMU data."""
        return [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ]

    def process_gnss(self, gnss_msg):
        """Process GNSS data."""
        latitude, longitude, altitude = (
            gnss_msg.latitude, gnss_msg.longitude, gnss_msg.altitude
        )
        velocity = 0.0  # Placeholder: Compute from GNSS deltas
        heading = 0.0  # Placeholder: Compute heading
        return [latitude, longitude, altitude, velocity, heading]

    def aggregate_state_vector(self, image_features, lidar_features, imu_features, gnss_features):
        """Aggregate features into a single state vector."""
        state_vector = np.zeros(50)  # Define a fixed size state vector
        state_vector[:20] = image_features
        state_vector[20:35] = lidar_features[:15]
        state_vector[35:41] = imu_features
        state_vector[41:] = gnss_features[:5]
        return state_vector

    def get_latest_data(self):
        """Retrieve the most recent synchronized data."""
        return self.data_buffer[-1] if self.data_buffer else None

    def clear_buffer(self):
        """Clear the stored data buffer."""
        self.data_buffer.clear()

def main(args=None):
    rclpy.init(args=args)

    # Update to integrate DataCollector with spawn_vehicle.py
    from client_node.spawn_vehicle import SpawnVehicleNode

    # Initialize SpawnVehicleNode to manage vehicle and sensors
    spawn_node = SpawnVehicleNode()
    if spawn_node.vehicle:
        carla_client = spawn_node.client
        carla_world = spawn_node.world
        
        # Use the vehicle and CARLA connection in DataCollector
        node = DataCollector(carla_client=carla_client, carla_world=carla_world)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            spawn_node.destroy_node()
    else:
        print("No vehicle found. Ensure the vehicle is spawned correctly.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
