import os
import rclpy
import math
import struct

from carla import Client
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch
import torch.nn as nn
from ament_index_python.packages import get_package_share_directory
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from std_msgs.msg import Float32MultiArray, String


class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")
        try:
            self.prev_gnss = (
                None  # Store previous GNSS position for velocity calculation
            )
            self.prev_time = None  # Store timestamp
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server: {e}")
            return

        self.data_buffer = []  # List of dictionaries to store synchronized data
        self.setup_subscribers()
        self.vision_model = self.initialize_vision_model()
        self.get_logger().info("DataCollector Node initialized.")
        self.start_vehicle_manager = self.create_publisher(
            String, "/start_vehicle_manager", 10
        )
        self.lap_subscription = self.create_subscription(
            String,
            "/lap_completed",  # Topic name for lap completion
            self.lap_ending_callback,  # Callback function
            10,  # QoS
        )
        self.publish_to_PPO = self.create_publisher(
            Float32MultiArray, "/data_to_ppo", 10
        )

    def lap_ending_callback(self, msg):
        """Callback function for lap completion."""
        self.get_logger().info("Lap completed.")
        request_msg = String()
        request_msg.data = "DataCollector is ready"
        self.start_vehicle_manager.publish(request_msg)
        self.get_logger().info("Data collector is ready to start the next lap.")

    def setup_subscribers(self):
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.get_logger().info("Setting up subscribers...")

        self.get_logger().info("Waiting for image data...")
        self.image_sub = Subscriber(self, Image, "/carla/rgb_front/image_raw")
        self.get_logger().info("Waiting for LiDAR data...")
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/lidar/points")
        self.get_logger().info("Waiting for IMU data...")
        self.imu_sub = Subscriber(self, Imu, "/carla/imu/imu")
        self.get_logger().info("Waiting for GNSS data...")
        self.gnss_sub = Subscriber(self, NavSatFix, "/carla/gnss/gnss")

        # self.lidar_sub, self.imu_sub, self.gnss_sub
        self.ats = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.imu_sub, self.gnss_sub ],
            queue_size=100,
            slop=0.1,  # Adjusted for better synchronization
        )

        # Print information about the synchronizer
        self.ats.registerCallback(self.sync_callback)
        self.get_logger().info(f"Synchronizer slop: {self.ats.__dict__}")
        self.get_logger().info("Subscribers set up successfully.")

    def initialize_vision_model(self):
        """Initialize a convolutional model (e.g., MobileNetV2) to process images using PyTorch."""
        weights = MobileNet_V2_Weights.IMAGENET1K_V1  # Use latest weight enum
        model = mobilenet_v2(weights=weights)

        model = nn.Sequential(
            model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=weights.transforms().mean, std=weights.transforms().std
                ),
            ]
        )
        self.get_logger().info("Vision model initialized successfully.")
        return model

    #   lidar_msg, imu_msg, gnss_msg
    def sync_callback(self, image_msg, lidar_msg, imu_msg,  gnss_msg):
        self.get_logger().info("Synchronized callback triggered.")
        processed_data = self.process_data(image_msg, lidar_msg, imu_msg, gnss_msg)
        self.data_buffer.append(processed_data)
        self.get_logger().info("Data appended to buffer.")

    def process_data(self,image_msg, lidar_msg, imu_msg, gnss_msg):
        self.get_logger().info("Processing data...")
        return self.aggregate_state_vector(
            self.process_image(image_msg),
            self.process_lidar(lidar_msg),
            self.process_imu(imu_msg),
            self.process_gnss(gnss_msg),
        )

    def process_image(self, image_msg):
        """Process camera image data and extract convolutional features using PyTorch."""
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, -1)
        )
        if raw_image.shape[2] == 4:
            raw_image = raw_image[:, :, :3]  # Remove alpha channel if present

        # Convert NumPy array to tensor (but do not use `ToTensor()` here)
        input_image = (
            torch.tensor(raw_image).permute(2, 0, 1).float() / 255.0
        )  # Normalize to [0,1]

        # Apply transformations properly (avoid calling `ToTensor()` on a tensor)
        if isinstance(input_image, torch.Tensor):
            input_image = transforms.Resize((224, 224))(input_image)  # Resize only
            input_image = transforms.Normalize(
                mean=self.transform.transforms[-1].mean,
                std=self.transform.transforms[-1].std,
            )(input_image)

        input_image = input_image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = (
                self.vision_model(input_image).squeeze(0).numpy()
            )  # Extract features
        return features[:20]  # Return 20 feature cells

    def process_lidar(self, lidar_msg):
        """Process LiDAR data."""
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack("ffff", lidar_msg.data)
        ]
        self.get_logger().info(f"Received {len(points)} LiDAR points.")
        mean_height = np.mean([p[2] for p in points]) if points else 0
        density = len(points) / 100  # Normalize for density
        self.get_logger().info(
            f"Processing image data...\n Mean height: {mean_height}\n Density: {density}"
        )
        return [mean_height, density] + points[:13]  # Up to 15 features

    def process_imu(self, imu_msg):
        """Process IMU data."""
        self.get_logger().info("Processing IMU data...")
        imu_data = [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        ]
        self.get_logger().info(f"IMU data: {imu_data}")
        return imu_data

    # use geopy.distance for safer calculations.
    def process_gnss(self, gnss_msg):
        """Process GNSS data and compute velocity and heading."""
        latitude, longitude, altitude = (
            gnss_msg.latitude,
            gnss_msg.longitude,
            gnss_msg.altitude,
        )

        # Compute velocity using previous GNSS readings
        velocity = 0.0
        heading = 0.0

        if self.prev_gnss is not None and self.prev_time is not None:
            delta_time = (
                self.get_clock().now() - self.prev_time
            ).nanoseconds / 1e9  # Convert to seconds
            delta_lat = latitude - self.prev_gnss[0]
            delta_lon = longitude - self.prev_gnss[1]

            # Approximate distance (assuming small delta lat/lon)
            delta_x = delta_lon * 111320  # Convert lon to meters
            delta_y = delta_lat * 110540  # Convert lat to meters
            distance = math.sqrt(delta_x**2 + delta_y**2)
#           if delta_time > 0:
            velocity = distance / delta_time  # Speed in meters per second

            # Compute heading (angle of movement)
            heading = (
                math.degrees(math.atan2(delta_y, delta_x)) if distance > 0 else 0.0
            )

        # Update previous GNSS data
        self.prev_gnss = (latitude, longitude)
        self.prev_time = self.get_clock().now()

        return np.array([latitude, longitude, altitude, velocity, heading])

    # , lidar_features, imu_features, gnss_features
    def aggregate_state_vector(self,image_features, lidar_features, imu_featurs, gnss_features):
        """Aggregate features into a single state vector.""" 

        state_vector = np.zeros(46, dtype=np.float32)  # Adjusted to correct size
        state_vector[:20] = image_features
        state_vector[20:35] = (
            np.array(lidar_features[:15]).flatten()
            if isinstance(lidar_features, np.ndarray)
            else np.zeros(15)
        )
        state_vector[35:41] = imu_featurs
        state_vector[41:46] = gnss_features[:5]  # Now includes velocity and heading
        self.get_logger().info(f"State Vector: {state_vector}")
        self.get_logger().info("State vector aggregated successfully.")

        response = Float32MultiArray()
        response.data = state_vector.tolist()
        self.publish_to_PPO.publish(response)

        return state_vector

    def get_latest_data(self):
        """Retrieve the most recent synchronized data."""
        self.get_logger().info("Retrieving latest data...")
        return self.data_buffer[-1] if self.data_buffer else None

    def clear_buffer(self):
        """Clear the stored data buffer."""
        self.get_logger().info("Clearing data buffer...")
        self.data_buffer.clear()


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
