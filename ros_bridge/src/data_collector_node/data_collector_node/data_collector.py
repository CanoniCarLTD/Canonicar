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

import cv2
from pathlib import Path
import threading
from queue import Queue

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # might reduce performance time! Uncomment for debugging CUDA errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RECORD_SS_IMAGES = False  # flip to False when you just want PPO
SAVE_EVERY_N_FRAMES = 5
DATA_ROOT = Path(
    "/ros_bridge/src/data_collector_node/data_collector_node/VAE/images"
)  # will create Train/ Val/ inside
NUMBER_OF_IMAGES = 14000


class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")

        # torch.autograd.set_detect_anomaly(True) # slows things down, so only enable it for debugging.

        # Flag to track collector readiness
        self.ready_to_collect = False
        self.vehicle_id = None
        self.last_sensor_timestamp = time.time()

        self.steps_counter = 0

        self.data_buffer = []  # List of dictionaries to store synchronized data

        self.imu_mean = np.zeros(6, dtype=np.float32)
        self.imu_var = np.ones(6, dtype=np.float32)
        self.imu_count = 1e-4  # avoid div by zero
        self.prev_time = None
        # concat 5 more values: velocity, throttle, previous steer, dev from center and angle
        
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0

        # Create state subscriber first to handle simulation status
        self.state_subscription = self.create_subscription(
            String, "/simulation/state", self.handle_system_state, 10
        )

        self.publish_to_PPO = self.create_publisher(
            Float32MultiArray, "/data_to_ppo", 10
        )

        # Setup vision processing
        self.vision_processor = VisionProcessor(device=device)

        # # ─── SMOKE-TEST (DEBUG) ───────────────────────────────────────────
        # dummy_img = np.zeros((80, 160, 3), dtype=np.uint8)
        # dummy_lidar = np.zeros((0, 4), dtype=np.float32)
        # state = self.vision_processor.process_sensor_data(dummy_img, dummy_lidar)
        # assert state.shape == (192,), f"State vector wrong shape {state.shape}"
        # self.get_logger().info(
        #     f"VisionProcessor smoke-test passed: state.shape = {state.shape}"
        # )
        # # ───────────────────────────────────────────────────────────

        self.image_sub = Subscriber(self, Image, "/carla/segmentation_front/image")
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/lidar/points")
        self.imu_sub = Subscriber(self, Imu, "/carla/imu/imu")

        # Semantic segmentation and VAE training
        self.frame_id = 0
        self.saved_image_index = 1
        self.save_queue = Queue(maxsize=128)
        # if RECORD_SS_IMAGES:
        #     ts = time.strftime("%Y%m%d")
        #     self.run_dir = DATA_ROOT / f"raw_{ts}"
        #     self.run_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        #     existing_files = list(self.run_dir.glob("*.png"))
        #     if existing_files:
        #         max_idx = max(
        #             [int(f.stem) for f in existing_files if f.stem.isdigit()], default=0
        #         )
        #         self.saved_image_index = max_idx + 1
        #     self.writer_thread = threading.Thread(target=self._disk_writer, daemon=True)
        #     self.writer_thread.start()

        self.get_logger().info("DataCollector Node initialized. Waiting for vehicle...")

    # Helper functions for VAE training
    def _disk_writer(self):
        """Runs in background; receives (img, path) tuples from queue."""
        while True:
            if self.saved_image_index > NUMBER_OF_IMAGES:
                self.get_logger().info(
                    f"Reached the maximum number of images ({NUMBER_OF_IMAGES}). Stopping image saving."
                )
                break  # Exit the loop to stop saving images
            img, out_path = self.save_queue.get()
            try:
                cv2.imwrite(str(out_path), img)
            except Exception as e:
                self.get_logger().error(f"[VAE-rec] Failed to save {out_path}: {e}")
            self.save_queue.task_done()

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
            self.get_logger().warn(
                "Sensors aren't sending data. Last update was "
                + f"{current_time - self.last_sensor_timestamp:.1f}s ago"
            )

    def handle_system_state(self, msg):
        """Handle changes in simulation state"""
        state_msg = msg.data

        # Parse state
        if ":" in state_msg:
            state_name, details = state_msg.split(":", 1)
        else:
            state_name = state_msg
            details = ""

        # Handle different states
        if state_name in ["RESPAWNING", "MAP_SWAPPING"]:
            if "vehicle_relocated" in details and self.ready_to_collect:
                self.last_sensor_timestamp = time.time()
                self.get_logger().info(
                    f"Reset sensor timestamp after vehicle relocation"
                )
            else:
                self.ready_to_collect = False
                self.get_logger().info(
                    f"Pausing data collection during {state_name}: {details}"
                )

        elif state_name == "RUNNING":
            # Check if we have details about vehicle readiness
            if "vehicle_" in details and "ready" in details:
                try:
                    # Extract vehicle_id from "vehicle_{id}_ready"
                    vehicle_id_str = details.split("vehicle_")[1].split("_ready")[0]
                    self.vehicle_id = int(vehicle_id_str)
                    self.last_sensor_timestamp = time.time()
                    self.setup_subscribers()
                    self.get_logger().info(
                        f"Data collection started for vehicle {self.vehicle_id}"
                    )
                except Exception as e:
                    self.get_logger().error(f"Error parsing vehicle ID from state: {e}")

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
                if self.steps_counter % 5 == 0:
                    self.publish_to_PPO.publish(response)
                self.steps_counter += 1
                if self.steps_counter == 100000:
                    # reset the counter to avoid overflow
                    self.steps_counter = 0
            else:
                self.get_logger().warn("State vector contains NaN values. Skipping...")

        except Exception as e:
            self.get_logger().error(f"Error in sync callback: {e}")

    def process_data(self, image_msg, lidar_msg, imu_msg):
        """Process sensor data into state vector"""
        self.process_imu(imu_msg)
        # Extract semantic segmentation image directly (already processed by carla_semantic_image_to_ros_image)
        raw_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            (image_msg.height, image_msg.width, 3)  # BGR format from semantic converter
        )

        # ─── NEW: queue saving ───────────────────────────────────────────────
        if RECORD_SS_IMAGES and (self.frame_id % SAVE_EVERY_N_FRAMES == 0):
            run_dir = self.run_dir
            out_path = run_dir / f"{self.saved_image_index:06}.png"
            self.saved_image_index += 1
            try:
                self.save_queue.put_nowait((raw_image.copy(), out_path))
            except Queue.Full:
                self.get_logger().warn("[VAE-rec] Save queue full, dropping frame")
        self.frame_id += 1
        # --------------------------------------------------------------------

        # Convert lidar_msg to point list
        points = [
            [point[0], point[1], point[2]]  # Extract x, y, z
            for point in struct.iter_unpack("ffff", lidar_msg.data)
        ]

        # Process using vision model - note that we're passing raw_image directly
        # which is now a semantic segmentation image
        vision_features = self.vision_processor.process_sensor_data(raw_image, points)
        return self.aggregate_state_vector(vision_features)

    def process_vision_data(self, image_msg, lidar_msg):
        """Process both segmentation and LiDAR data using our fusion model."""
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


    def update_velocity_from_imu(self, imu_msg):
        # Get current time
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            return 0.0  # No velocity update on first call

        # Calculate time difference
        dt = current_time - self.prev_time
        self.prev_time = current_time

        # Integrate acceleration to estimate velocity
        self.velocity_x += imu_msg[0] * dt
        self.velocity_y += imu_msg[1] * dt
        self.velocity_z += imu_msg[2] * dt

        # Compute velocity magnitude (m/s)
        velocity = np.sqrt(
            self.velocity_x**2 +
            self.velocity_y**2 +
            self.velocity_z**2
        )*3.6  # Convert to km/h
        return velocity
    
    def process_imu(self, imu_msg):
        imu_raw = np.array(
            [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z,
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )

        # Update running stats
        self.update_imu_stats(imu_raw)

        # Normalize and clip
        imu_scaled = self.normalize_imu(imu_raw)
        self.velocity = self.update_velocity_from_imu(imu_scaled)
    

    def update_imu_stats(self, imu_sample):
        delta = imu_sample - self.imu_mean
        self.imu_count += 1
        self.imu_mean += delta / self.imu_count
        self.imu_var += delta * (imu_sample - self.imu_mean)

    def normalize_imu(self, imu_sample):
        std = np.sqrt(self.imu_var / self.imu_count + 1e-6)
        normalized = (imu_sample - self.imu_mean) / std
        return np.clip(normalized, -3.0, 3.0)

    def aggregate_state_vector(self, vision_features):
        """Aggregate features into a single state vector."""
        # Total vector size: 95 (vision) + 5 (speed etc.) = 100
        state_vector = np.zeros(100, dtype=np.float32)

        # Fill with vision features (fused Segmentation + LiDAR)
        state_vector[:95] = vision_features

        # # Add IMU data
        state_vector[96:97] = self.velocity
        state_vector[97:98] = self.velocity / 22 # velocity/target speed

        return state_vector


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if RECORD_SS_IMAGES:
            node.save_queue.join()  # flush pending writes
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
