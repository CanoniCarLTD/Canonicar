from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np

class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        
        # CHANGE THE TOPIC PATH
        self.data_sub = self.create_subscription(
            Float32MultiArray, "/carla/###/###", self.data_callback, 10
        )

        # Action publisher (Steering, Throttle, Brake)
        # CHANGE THE TOPIC PATH
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 10
        )
        
        self.get_logger().info("PPOModelNode initialized and subscribed to data topic.")

    def data_callback(self, msg):
        self.get_logger().info(
            f"Received image with resolution {msg.width}x{msg.height}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
