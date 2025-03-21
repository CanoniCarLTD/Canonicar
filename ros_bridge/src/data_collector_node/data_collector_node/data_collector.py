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
from vision_model import VisionProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.vision_processor = VisionProcessor(device= device)
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

    

    #   lidar_msg, imu_msg, gnss_msg
    def sync_callback(self, image_msg, lidar_msg, imu_msg,  gnss_msg):

        self.get_logger().info("Synchronized callback triggered.")
        processed_data = self.process_data(image_msg, lidar_msg, imu_msg, gnss_msg)
        self.data_buffer.append(processed_data)
        self.get_logger().info("Data appended to buffer.")

    def process_data(self, image_msg, lidar_msg, imu_msg, gnss_msg):

        self.get_logger().info("Processing data...")

        # Use the combined vision processor instead of separate processing
        vision_features = self.process_vision_data(image_msg, lidar_msg)

        return self.aggregate_state_vector(
            vision_features,

            self.process_imu(imu_msg),
            self.process_gnss(gnss_msg),
        )

    def process_vision_data(self, image_msg, lidar_msg):
        """Process both RGB and LiDAR data using our fusion model."""
        # Convert image_msg to numpy array
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, -1)
        )
        if raw_image.shape[2] == 4:  # BGRA format
            raw_image = raw_image[:, :, :3]  # Remove alpha channel
            raw_image = raw_image[:, :, ::-1].copy()  # Convert BGR to RGB
        # Convert lidar_msg to point list
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack("ffff", lidar_msg.data)
        ]

        # Process using our vision model
        fused_features = self.vision_processor.process_sensor_data(raw_image, points)
        return fused_features


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
    def aggregate_state_vector(self, vision_features, imu_features, gnss_features):

        """Aggregate features into a single state vector.""" 

        # Assuming vision_features is 192 dimensions from the SensorFusionModel
        # Total vector size: 192 (vision) + 6 (IMU) + 5 (GNSS) = 203
        state_vector = np.zeros(203, dtype=np.float32)
        
        # Fill with vision features (fused RGB + LiDAR)
        state_vector[:192] = vision_features

        # Add IMU data
        state_vector[192:198] = imu_features

        # Add GNSS data
        state_vector[198:203] = gnss_features[:5]  # Includes velocity and heading

        self.get_logger().info(f"State Vector shape: {state_vector.shape}")

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