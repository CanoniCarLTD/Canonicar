from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
import sys
import os

#Fix the path to the ML folder, can't import the agent and ppo files

#import ML.networks.agent as agent
#import ML.networks.ppo as ppo

class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        
        # CHANGE THE TOPIC PATH
        self.data_sub = self.create_subscription(
            Float32MultiArray, "carla_demo_topic", self.data_callback, 100
            #Float32MultiArray, "/carla/###/###", self.data_callback, 10
        )

        # Action publisher (Steering, Throttle, Brake)
        # CHANGE THE TOPIC PATH
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 100
        )
        
        self.get_logger().info("PPOModelNode initialized and subscribed to data topic.")

    def data_callback(self, msg):
        self.get_logger().info(
            #f"Received image with resolution {msg.width}x{msg.height}"
            f"Received data: {msg.data}"
        )
        self.send_to_ppo(msg.data)
        
    def send_to_ppo(self, data):
        # Send data to PPO model
        ppoModel=agent.PPOAgent(town="Town10HD",action_std_init=0.4)
        if ppoModel is not None:
            print("PPO model is not None")


def main(args=None):
    #print(torch.__version__)
    #print(torch.cuda.is_available())
    #print(torch.cuda.get_device_name(0))
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
