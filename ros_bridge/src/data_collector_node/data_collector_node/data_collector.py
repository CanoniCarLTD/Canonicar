import os

import cv2
import rclpy #type: ignore
import math
import struct
import time

from carla import Client
from rclpy.node import Node #type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy #type: ignore
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix #type: ignore
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber #type: ignore
import torch
import torch.nn as nn
from std_msgs.msg import Float32MultiArray, String #type: ignore
from vision_model import VisionProcessor

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # might reduce performance time! Uncomment for debugging CUDA errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")
        
        # torch.autograd.set_detect_anomaly(True) # slows things down, so only enable it for debugging.
        
        self.record_rgb = False
        self.record_buffer = []
        os.makedirs("/ros_bridge/src/client_node/client_node/data/rgb_finetune/train", exist_ok=True)
        
        # Initialize image index from labels.csv
        labels_path = "/ros_bridge/src/client_node/client_node/data/rgb_finetune/train/labels.csv"
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:  # Check if there are entries beyond the header
                    last_line = lines[-1]
                    last_filename = last_line.split(",")[0]
                    self.image_index = int(last_filename.split(".")[0]) + 1
                else:
                    self.image_index = 0
        else:
            # Create the file with a header if it doesn't exist
            with open(labels_path, "w") as f:
                f.write("filename,steer\n")
            self.image_index = 0
        
        # Flag to track collector readiness
        self.ready_to_collect = False
        self.vehicle_id = None
        self.last_sensor_timestamp = time.time()
        self.latest_image_msg = None
        self.latest_steering = 0.0

        self.data_buffer = []  # List of dictionaries to store synchronized data
        # Count how many PNGs already exist — start from there
        existing_files = os.listdir("/ros_bridge/src/client_node/client_node/data/rgb_finetune/train")
        existing_images = [f for f in existing_files if f.endswith(".png")]
        self.image_index = len(existing_images)

        self.imu_mean = np.zeros(6, dtype=np.float32)
        self.imu_var = np.ones(6, dtype=np.float32)
        self.imu_count = 1e-4  # avoid div by zero
        
        # Create state subscriber first to handle simulation status
        self.state_subscription = self.create_subscription(
            String,
            '/simulation/state',
            self.handle_system_state,
            10
        )
        
        # Add a regular subscriber for RGB images (not using message_filters)
        self.rgb_image_subscription = self.create_subscription(
            Image,
            "/carla/rgb_front/image_raw",
            self.handle_rgb_image,
            10
        )
        self.action_subscription = self.create_subscription(
            Float32MultiArray,
            '/carla/vehicle/steer',
            self.handle_save_rgb_steering,
            10
        )

    
        self.publish_to_PPO = self.create_publisher(
            Float32MultiArray, "/data_to_ppo", 10
        )
        
        # Setup vision processing
        self.vision_processor = VisionProcessor(device = device, pretrained_rgb_encoder = "/ros_bridge/src/client_node/client_node/train/checkpoints/mobilenet_trackslice14.pth")
        
        self.image_sub = Subscriber(self, Image, "/carla/segmentation_front/image")
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/lidar/points")
        self.imu_sub = Subscriber(self, Imu, "/carla/imu/imu")
        
        self.get_logger().info("DataCollector Node initialized. Waiting for vehicle...")

    def handle_rgb_image(self, msg):
        """Store the latest RGB image"""
        self.latest_image_msg = msg

    def setup_subscribers(self):
        """Set up subscribers once we verify the camera is publishing""" 
        # Create health check timer
        self.sensor_health_timer = self.create_timer(5.0, self.check_sensor_health)
        
        # Create synchronizer with more relaxed settings
        self.ats = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub, self.imu_sub],
            queue_size=60,
            slop=0.2,  # More relaxed time synchronization
        )
        
        # Register the callback immediately
        self.ats.registerCallback(self.sync_callback)
        self.ready_to_collect = True
    
    def check_sensor_health(self):
        """Check if sensors are still active"""
        current_time = time.time()
        if self.ready_to_collect and current_time - self.last_sensor_timestamp > 2.0:
            self.get_logger().warn("Sensors aren't sending data. Last update was " + 
                                f"{current_time - self.last_sensor_timestamp:.1f}s ago")
        
    def handle_system_state(self, msg):
        """Handle changes in simulation state"""
        state_msg = msg.data
        
        # Parse state
        if ':' in state_msg:
            state_name, details = state_msg.split(':', 1)
        else:
            state_name = state_msg
            details = ""
        
        # Handle different states
        if state_name in ["RESPAWNING", "MAP_SWAPPING"]:
            if "vehicle_relocated" in details and self.ready_to_collect:
                self.last_sensor_timestamp = time.time()
                self.get_logger().info(f"Reset sensor timestamp after vehicle relocation")
            else:
                self.ready_to_collect = False
                self.get_logger().info(f"Pausing data collection during {state_name}: {details}")
            
        elif state_name == "RUNNING":
            # Check if we have details about vehicle readiness
            if "vehicle_" in details and "ready" in details:
                try:
                    # Extract vehicle_id from "vehicle_{id}_ready"
                    vehicle_id_str = details.split("vehicle_")[1].split("_ready")[0]
                    self.vehicle_id = int(vehicle_id_str)
                    self.last_sensor_timestamp = time.time()
                    self.setup_subscribers()
                    self.get_logger().info(f"Data collection started for vehicle {self.vehicle_id}")
                except Exception as e:
                    self.get_logger().error(f"Error parsing vehicle ID from state: {e}")
    
    def handle_save_rgb_steering(self, msg):
        self.latest_steering = msg.data[0]  # Update with the steering value            
        if self.record_rgb and self.latest_steering is not None:
            # Use the latest stored image message
            if self.latest_image_msg is None:
                self.get_logger().warn("No image message received yet.")
                return

            # Convert image_msg to numpy array
            raw_image = np.frombuffer(self.latest_image_msg.data, dtype=np.uint8).reshape(
                (self.latest_image_msg.height, self.latest_image_msg.width, -1)
            )
            if raw_image.shape[2] == 4:  # BGRA format
                bgr_img = raw_image[:, :, :3]  # Remove alpha channel
                
            steer = self.latest_steering
            fn = f"{self.image_index:05d}.png"
            cv2.imwrite(f"/ros_bridge/src/client_node/client_node/data/rgb_finetune/train/{fn}", bgr_img)
            with open("/ros_bridge/src/client_node/client_node/data/rgb_finetune/train/labels.csv", "a") as f:
                f.write(f"{fn},{steer:.4f}\n")
            self.record_buffer.append(fn)
            self.image_index += 1
            
    def sync_callback(self, image_msg, lidar_msg, imu_msg):
        """Process synchronized data"""
        if not self.ready_to_collect:
            return
        try:
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
        # Extract semantic segmentation image directly (already processed by carla_semantic_image_to_ros_image)
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, 3)  # BGR format from semantic converter
        )
        
        # Convert lidar_msg to point list
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack("ffff", lidar_msg.data)
        ]

        # Process using vision model - note that we're passing raw_image directly
        # which is now a semantic segmentation image
        vision_features = self.vision_processor.process_sensor_data(raw_image, points)
        
        return self.aggregate_state_vector(
            vision_features, self.process_imu(imu_msg)
        )

    def process_vision_data(self, image_msg, lidar_msg):
        """Process both RGB and LiDAR data using our fusion model."""
        # Convert image_msg to numpy array
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, -1)
        )
        if raw_image.shape[2] == 4:  # BGRA format
            raw_image = raw_image[:, :, :3]  # Remove alpha channel
            raw_rgb_image = raw_image[:, :, ::-1]  # Convert to RGB format
        # Convert lidar_msg to point list
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack("ffff", lidar_msg.data)
        ]

        # Process using our vision model
        fused_features = self.vision_processor.process_sensor_data(raw_rgb_image, points)
        return fused_features

    def process_imu(self, imu_msg):
        imu_raw = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z,
        ], dtype=np.float32)

        # Update running stats
        self.update_imu_stats(imu_raw)

        # Normalize and clip
        imu_scaled = self.normalize_imu(imu_raw)
        return imu_scaled.tolist()
    
    def update_imu_stats(self, imu_sample):
        delta = imu_sample - self.imu_mean
        self.imu_count += 1
        self.imu_mean += delta / self.imu_count
        self.imu_var += delta * (imu_sample - self.imu_mean)

    def normalize_imu(self, imu_sample):
        std = np.sqrt(self.imu_var / self.imu_count + 1e-6)
        normalized = (imu_sample - self.imu_mean) / std
        return np.clip(normalized, -3.0, 3.0)

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