import os
import rclpy
import math
import struct
import time

from carla import Client
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch
import torch.nn as nn
from std_msgs.msg import Float32MultiArray, String
from vision_model import VisionProcessor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Uncomment for debugging CUDA errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")
        
        # torch.autograd.set_detect_anomaly(True) # slows things down, so only enable it for debugging.

        # Flag to track collector readiness
        self.ready_to_collect = False
        self.vehicle_id = None
        self.last_sensor_timestamp = time.time()
        self.setup_done = False
        
        self.data_buffer = []  # List of dictionaries to store synchronized data
        
        # Create state subscriber first to handle simulation status
        self.state_subscription = self.create_subscription(
            String,
            '/simulation/state',
            self.handle_system_state,
            10
        )
        
        # Vehicle ready subscription
        self.vehicle_ready_subscription = self.create_subscription(
            String,
            '/vehicle_ready',
            self.handle_vehicle_ready,
            10
        )
        
        
        self.publish_to_PPO = self.create_publisher(
            Float32MultiArray, "/data_to_ppo", 10
        )
        
        # Setup vision processing
        self.vision_processor = VisionProcessor(device=device)
        
        self.image_sub = Subscriber(self, Image, "/carla/rgb_front/image_raw")
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/lidar/points")
        self.imu_sub = Subscriber(self, Imu, "/carla/imu/imu")
        
        self.get_logger().info("DataCollector Node initialized. Waiting for vehicle...")

    def setup_subscribers(self):
        """Set up subscribers once we verify the camera is publishing""" 

        # Create synchronizer with more relaxed settings
        self.ats = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.imu_sub],
            queue_size=60,
            slop=0.2,  # More relaxed time synchronization
        )
        
        # Register the callback immediately
        self.ats.registerCallback(self.sync_callback)
        self.ready_to_collect = True
        
    def handle_system_state(self, msg):
        """Handle changes in simulation state"""
        state_msg = msg.data
        
        # Parse state
        if ':' in state_msg:
            state_name, details = state_msg.split(':', 1)
        else:
            state_name = state_msg
        
        # Handle different states
        if state_name in ["RESPAWNING", "MAP_SWAPPING"]:
            self.ready_to_collect = False
            
            if state_name == "RESPAWNING":
                if "collision" in details.lower():
                    self.get_logger().info("Respawning due to collision")
                elif "episode_complete" in details.lower():
                    self.get_logger().info("Respawning due to episode completion")
                else:
                    self.get_logger().info(f"Respawning for other reason: {details}")
        
        elif state_name == "RUNNING":
            # Don't immediately set ready_to_collect to True
            # We need to wait for vehicle_ready notification first
            self.get_logger().info("System returned to RUNNING state, waiting for vehicle ready")
    
    def handle_vehicle_ready(self, msg):
        """Handle notification that a vehicle is ready"""
        try:
            self.vehicle_id = int(msg.data)
            self.last_sensor_timestamp = time.time()
            self.setup_subscribers()
        except Exception as e:
            self.get_logger().error(f"Error handling vehicle ready: {e}")

    
    def sync_callback(self, image_msg, lidar_msg, imu_msg):
        """Process synchronized data"""
        if not self.ready_to_collect:
            return
        try:
            # Update timestamp to know sensors are active
            self.last_sensor_timestamp = time.time()
            
            processed_data = self.process_data(image_msg, lidar_msg, imu_msg)
            
            # Publish to PPO node for training/inference
            response = Float32MultiArray()
            response.data = processed_data.tolist()
            
            if not np.isnan(processed_data).any():
                self.publish_to_PPO.publish(response)
            else:
                self.get_logger().warn("State vector contains NaN values. Skipping...")
                
        except Exception as e:
            self.get_logger().error(f"Error in sync callback: {e}")

    def process_data(self, image_msg, lidar_msg, imu_msg):
        """Process sensor data into state vector"""
        # start_time = time.time()
        vision_features = self.process_vision_data(image_msg, lidar_msg)
        # end_time = time.time()
        # self.get_logger().info(f"Vision processing time: {end_time - start_time:.4f} seconds")
        return self.aggregate_state_vector(
            vision_features,
            self.process_imu(imu_msg)
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
        imu_data = [
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        ]
        return imu_data

    def aggregate_state_vector(self, vision_features, imu_features):
        """Aggregate features into a single state vector.""" 
        # Total vector size: 192 (vision) + 6 (IMU) = 198
        state_vector = np.zeros(198, dtype=np.float32)
        
        # Fill with vision features (fused RGB + LiDAR)
        state_vector[:192] = vision_features

        # # Add IMU data
        state_vector[192:198] = imu_features
        
        return state_vector

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