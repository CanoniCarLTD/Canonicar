from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
import sys
import os

# from ML.ppo_agent import PPOAgent
# from ML.parameters import PPO_CHECKPOINT_DIR, VERSION

from .ML import ppo_agent
from .ML import parameters


class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")

        self.data_sub = self.create_subscription(
            Float32MultiArray, "/data_to_ppo", self.data_callback, 10
        )

        # Action publisher (Steering, Throttle, Brake)
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 10
        )

        # Initialize PPO Agent (loads from checkpoint if available)
        self.ppo_agent = ppo_agent.PPOAgent()
        self.get_logger().info(
            "PPOModelNode initialized,subscribed to data topic and PPO model loaded."
        )
        self.get_logger().info(f"Model version: {parameters.VERSION}")
        self.get_logger().info(f"Checkpoint directory: {parameters.PPO_CHECKPOINT_DIR}")

    def data_callback(self, msg):
        self.get_logger().info(f"Received data in PPO node: {msg.data}")
        self.get_action(msg.data)
        self.publish_action()

    def get_action(self, data):
        self.action = self.ppo_agent.select_action(data)

    def publish_action(self):
        action_msg = Float32MultiArray()
        action_msg.data = self.action.tolist()
        self.action_publisher.publish(action_msg)
        self.get_logger().info(f"Published action: {action_msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
