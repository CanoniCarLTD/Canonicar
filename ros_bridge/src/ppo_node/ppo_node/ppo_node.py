from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from client_node import spawn_vehicle


class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        """
        These are not the right topics.
        Need to subscribe to the right topics after processing the data
        """
        
        # Need a topic to subscribe to convolution
        self.image_sub = self.create_subscription(
            Image, "/carla/rgb_front/image_raw", self.image_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2, "/carla/lidar/points", self.lidar_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/carla/imu/imu", self.imu_callback, 10
        )
        self.gnss_sub = self.create_subscription(
            NavSatFix, "/carla/gnss/gnss", self.gnss_callback, 10
        )

        # Action publisher (Steering, Throttle, Brake)
        # Publish the action data in another node?
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 10
        )
        
        self.get_logger().info("PPOModelNode initialized and subscribed to topics.")

    def image_callback(self, msg):
        self.get_logger().info(
            f"Received image with resolution {msg.width}x{msg.height}"
        )

    def lidar_callback(self, msg):
        self.get_logger().info(f"Received LiDAR data with {msg.width} points")

    def imu_callback(self, msg):
        self.get_logger().info(
            f"Received IMU data: orientation [{msg.orientation.x}, {msg.orientation.y}, {msg.orientation.z}]"
        )

    def gnss_callback(self, msg):
        self.get_logger().info(
            f"Received GNSS data: Latitude {msg.latitude}, Longitude {msg.longitude}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
