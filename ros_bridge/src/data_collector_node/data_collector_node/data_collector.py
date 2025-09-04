import os
import rclpy
import math
import struct
import time
import weakref

from carla import Client
import carla
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

RECORD_SS_IMAGES = True  # flip to False when you just want PPO
SAVE_EVERY_N_FRAMES = 1
DATA_ROOT = Path(
    "/ros_bridge/src/data_collector_node/data_collector_node/VAE/images"
)  # will create Train/ Val/ inside
NUMBER_OF_IMAGES = 14000


class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")

        # Connect to CARLA client
        self.carla_client = None
        self.carla_world = None
        self.camera_sensor = None
        self.vehicle_actor = None
        self.latest_camera_data = None
        
        # Flag to track collector readiness
        self.ready_to_collect = False
        self.vehicle_id = None
        self.last_sensor_timestamp = time.time()

        self.front_camera = list()
        self.nav_data = [0.0] * 5  # Initialize navigation data

        # Connect to CARLA
        self._connect_to_carla()

        # Create state subscriber first to handle simulation status
        self.state_subscription = self.create_subscription(
            String, "/simulation/state", self.handle_system_state, 10
        )

        self.navigation_subscription = self.create_subscription(
            Float32MultiArray, "/carla/vehicle/navigation", self.handle_navigation_data, 10
        )

        self.publish_to_PPO = self.create_publisher(
            Float32MultiArray, "/data_to_ppo", 10
        )

        # Setup vision processing
        self.vision_processor = VisionProcessor(device=device)

        # Semantic segmentation and VAE training
        self.frame_id = 0
        self.saved_image_index = 1
        self.save_queue = Queue(maxsize=128)
        
        if RECORD_SS_IMAGES:
            ts = time.strftime("%Y%m%d")
            self.run_dir = DATA_ROOT / f"raw_{ts}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            existing_files = list(self.run_dir.glob("*.png"))
            if existing_files:
                max_idx = max(
                    [int(f.stem) for f in existing_files if f.stem.isdigit()], default=0
                )
                self.saved_image_index = max_idx + 1
            self.writer_thread = threading.Thread(target=self._disk_writer, daemon=True)
            self.writer_thread.start()

        # Timer to process camera data and publish to PPO
        self.processing_timer = None

        self.get_logger().info("DataCollector Node initialized. Waiting for vehicle...")

    def _connect_to_carla(self):
        """Connect to CARLA client"""
        try:
            self.carla_client = Client('5.29.227.167', 2000)
            self.carla_client.set_timeout(10.0)
            self.carla_world = self.carla_client.get_world()
            self.get_logger().info("Connected to CARLA server")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA: {e}")
            self.carla_client = None
            self.carla_world = None

    def _find_vehicle_by_id(self, vehicle_id):
        """Find vehicle actor by ID in CARLA world"""
        if not self.carla_world:
            return None
        
        try:
            # Get all actors and find the vehicle with matching ID
            actors = self.carla_world.get_actors()
            for actor in actors:
                if actor.type_id.startswith('vehicle.') and actor.id == vehicle_id:
                    return actor
            
            # If not found by exact ID, try to find the ego vehicle
            for actor in actors:
                if actor.type_id.startswith('vehicle.') and hasattr(actor, 'attributes'):
                    if actor.attributes.get('role_name') == 'ego_vehicle':
                        return actor
            
            return None
        except Exception as e:
            self.get_logger().error(f"Error finding vehicle: {e}")
            return None

    def _setup_camera_sensor(self, vehicle_actor):
        """Setup camera sensor attached to the vehicle"""
        try:
            if not self.carla_world or not vehicle_actor:
                self.get_logger().error("Cannot setup camera: missing world or vehicle")
                return None

            # Clean up existing sensor
            if self.camera_sensor:
                self.camera_sensor.destroy()
                self.camera_sensor = None

            # Get semantic segmentation camera blueprint
            camera_bp = self.carla_world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', '160')
            camera_bp.set_attribute('image_size_y', '80')
            camera_bp.set_attribute('fov', '125')
            camera_bp.set_attribute('sensor_tick', '0.1')
            # Setup camera transform (front of vehicle)
            camera_transform = carla.Transform(
                carla.Location(x=5.0, z=3.0), 
                carla.Rotation(pitch=-12)
            )

            # Spawn camera sensor
            camera_sensor = self.carla_world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=vehicle_actor
            )

            # Setup camera callback
            weak_self = weakref.ref(self)
            camera_sensor.listen(
                lambda image: DataCollector._get_front_camera_data(weak_self, image)
            )

            self.camera_sensor = camera_sensor
            self.get_logger().info(f"Camera sensor spawned successfully for vehicle {vehicle_actor.id}")
            return camera_sensor

        except Exception as e:
            self.get_logger().error(f"Failed to setup camera sensor: {e}")
            return None
    
    def _has_dark_top_edge(self, image_array, top_rows=3, dark_threshold=30, dark_pixel_ratio=0.5):
        """
        Check if the top edge of the image has dark/unsegmented pixels.
        
        Args:
            image_array: numpy array of shape (height, width, 3)
            top_rows: number of top rows to check
            dark_threshold: threshold below which a pixel is considered dark (0-255)
            dark_pixel_ratio: minimum ratio of dark pixels to consider the edge "dark"
        
        Returns:
            True if the top edge has too many dark pixels (incomplete frame)
        """
        if image_array is None or image_array.size == 0:
            return True
            
        try:
            height, width, channels = image_array.shape
            
            # Ensure we don't exceed image dimensions
            check_rows = min(top_rows, height)
            
            if check_rows == 0:
                return True
            
            # Extract top edge rows
            top_edge = image_array[:check_rows, :]
            
            # Convert to grayscale for easier dark pixel detection
            gray_edge = np.mean(top_edge, axis=2)
            
            # Count dark pixels in the top edge
            dark_pixels = np.sum(gray_edge < dark_threshold)
            total_pixels = check_rows * width
            
            # Calculate ratio of dark pixels
            dark_ratio = dark_pixels / total_pixels
            
            is_dark = dark_ratio > dark_pixel_ratio
            
            if is_dark:
                self.get_logger().debug(
                    f"Dark top edge detected: {dark_pixels}/{total_pixels} "
                    f"pixels are dark ({dark_ratio:.2%}) in top {check_rows} rows"
                )
            
            return is_dark
            
        except Exception as e:
            self.get_logger().error(f"Error checking dark top edge: {e}")
            return True  # Assume dark/invalid if we can't check

    def _is_frame_complete(self, image_array):
        """
        Check if the frame is complete and properly segmented.
        
        Args:
            image_array: numpy array of shape (height, width, 3)
            
        Returns:
            True if the frame appears complete, False if incomplete/corrupted
        """
        if image_array is None or image_array.size == 0:
            return False
            
        try:
            # Check for dark pixels in top edge (main indicator of incomplete segmentation)
            if self._has_dark_top_edge(image_array):
                return False
            
            # # Additional checks for frame completeness
            # height, width, channels = image_array.shape
            
            # # Check if image has reasonable dimensions
            # if height < 10 or width < 10:
            #     return False
            
            # # Check for completely black image
            # mean_intensity = np.mean(image_array)
            # if mean_intensity < 5:  # Almost completely black
            #     return False
            
            # # Check for reasonable color diversity (segmented images should have various colors)
            # unique_colors = len(np.unique(image_array.reshape(-1, channels), axis=0))
            # if unique_colors < 3:  # Too few colors suggests incomplete segmentation
            #     return False
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error validating frame completeness: {e}")
            return False
    
    @staticmethod
    def _get_front_camera_data(weak_self, image):
        """Static callback method for camera data with top edge validation"""
        self = weak_self()
        if not self:
            return
        
        try:
            
            # Convert to semantic segmentation with CityScapes palette
            image.convert(carla.ColorConverter.CityScapesPalette)
            placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            placeholder1 = placeholder.reshape((image.height, image.width, 4))
            target = placeholder1[:, :, :3]
            
            # Add slight delay to ensure data integrity
            time.sleep(0.005)
            
            # Check if frame is complete and doesn't have dark top edge
            if self._is_frame_complete(target):
                # Frame is good - update data
                self.latest_camera_data = target.copy()  # Make a copy to avoid reference issues
                self.last_sensor_timestamp = time.time()
                    
            
        except Exception as e:
            if hasattr(self, 'get_logger'):
                self.get_logger().error(f"Error processing camera data: {e}")
            
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

    def start_processing(self):
        """Start the processing timer to regularly process camera data"""
        if self.processing_timer is None:
            # Process at ~20 Hz (0.05 seconds)
            self.processing_timer = self.create_timer(0.05, self.process_and_publish)
            self.ready_to_collect = True
            self.get_logger().info("Started camera data processing")

    def stop_processing(self):
        """Stop the processing timer"""
        if self.processing_timer:
            self.processing_timer.cancel()
            self.processing_timer = None

            self.ready_to_collect = False
            self.get_logger().info("Stopped camera data processing")

    def process_and_publish(self):
        """Process camera data and publish to PPO"""
        if not self.ready_to_collect or self.latest_camera_data is None:
            return
        
        try:
            # Process the camera data through vision model
            raw_image = self.latest_camera_data.copy()  # <— critical
            processed_data = self.process_data(raw_image)
                        
            # Publish to PPO node for training/inference
            response = Float32MultiArray()
            response.data = [float(self.frame_id)] + processed_data.tolist()
            self.publish_to_PPO.publish(response)
            
        except Exception as e:
            self.get_logger().error(f"Error in process_and_publish: {e}")

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
            self.latest_camera_data = None
            if "vehicle_relocated" in details and self.ready_to_collect:
                self.last_sensor_timestamp = time.time()
                self.get_logger().info(
                    f"Reset sensor timestamp after vehicle relocation"
                )
            else:
                self.stop_processing()
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
                    
                    # Find the vehicle in CARLA world
                    self.vehicle_actor = self._find_vehicle_by_id(self.vehicle_id)
                    if self.vehicle_actor:
                        # Setup camera sensor for this vehicle
                        camera_sensor = self._setup_camera_sensor(self.vehicle_actor)
                        if camera_sensor:
                            self.start_processing()
                            self.get_logger().info(
                                f"Data collection started for vehicle {self.vehicle_id}"
                            )
                        else:
                            self.get_logger().error(f"Failed to setup camera for vehicle {self.vehicle_id}")
                    else:
                        self.get_logger().error(f"Vehicle {self.vehicle_id} not found in CARLA world")

                except Exception as e:
                    self.get_logger().error(f"Error parsing vehicle ID from state: {e}")

    def handle_navigation_data(self, msg):
        """Handle incoming navigation data"""
        self.nav_data = msg.data
        if len(self.nav_data) != 5:
            self.get_logger().warn(
                f"Navigation data length mismatch. Expected 5, got {len(self.nav_data)}"
            )
            return

    def process_data(self, camera_image):
        """Process camera data into state vector"""
        # Use camera image directly (it's already processed by the callback)
        raw_image = camera_image

        # Save images if recording is enabled
        if RECORD_SS_IMAGES and (self.frame_id % SAVE_EVERY_N_FRAMES == 0):
            run_dir = self.run_dir
            out_path = run_dir / f"{self.saved_image_index:06}.png"
            self.saved_image_index += 1
            try:
                self.save_queue.put_nowait((raw_image.copy(), out_path))
            except:
                self.get_logger().warn("[VAE-rec] Save queue full, dropping frame")
        self.frame_id += 1

        # Process using vision model
        vision_features = self.vision_processor.model.process(raw_image, self.nav_data)
        return vision_features

    def __del__(self):
        """Cleanup when node is destroyed"""
        if self.camera_sensor:
            try:
                self.camera_sensor.destroy()
            except:
                pass


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
