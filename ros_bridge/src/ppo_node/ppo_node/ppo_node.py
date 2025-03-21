from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
import sys
import os
from datetime import datetime

from .ML import ppo_agent
from .ML import parameters
from .ML.parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        train = TRAIN
        self.current_episode = 0 
        self.data_sub = self.create_subscription(
            Float32MultiArray, "/data_to_ppo", self.training if train else self.testing, 10
        )

        # Action publisher (Steering, Throttle, Brake)
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 10
        )

        # Initialize PPO Agent (loads from checkpoint if available)
        self.ppo_agent = ppo_agent.PPOAgent()
        self.get_logger().info(f"Model version: {VERSION}")
        self.get_logger().info(f"Checkpoint directory: {PPO_CHECKPOINT_DIR}")
        
        # Initialize state, action, reward, etc.
        self.state = None
        self.action = None
        self.reward = 0
        self.done = False
        
        # Initialize episode counter and timestep counter
        self.episode_counter = 0
        self.timestep_counter = 0
        self.total_timesteps = TOTAL_TIMESTEPS
        self.episode_length = EPISODE_LENGTH
        self.current_ep_reward = 0
        
        self.get_logger().info(
            "PPOModelNode initialized,subscribed to data topic and PPO model loaded.")

    def get_action(self, data):
        self.action, _ = self.ppo_agent.select_action(data)
        self.get_logger().info(f"Retured action: {self.action}")

    def publish_action(self):
        action_msg = Float32MultiArray()
        action_msg.data = self.action.tolist()
        self.action_publisher.publish(action_msg)
        self.get_logger().info(f"Published action: {action_msg.data}")
        
    def calculate_reward(self):
        # Implement reward calculation logic here
        # For example:
        if self.state[0] < 0:  # Example condition for punishment
            self.reward = -1
        else:
            self.reward = 1  # Example condition for reward

    def store_transition(self):
        # Assuming you have access to the value and done flag
        value = self.ppo_agent.critic(torch.tensor(self.state, dtype=torch.float32).to(device).unsqueeze(0)).item()
        self.ppo_agent.store_transition(self.state, self.action, 0, value, self.reward, self.done)

    def reset_environment(self):
        # Implement your environment reset logic here
        self.state = None
        self.action = None
        self.reward = 0
        self.done = False
        self.current_ep_reward = 0
        self.get_logger().info(f"Episode {self.episode_counter} finished. Resetting environment.")

    def training(self, msg):
        if self.timestep_counter < self.total_timesteps:
            self.state = np.array(msg.data, dtype=np.float32)
            self.current_ep_reward = 0
            t1 = datetime.now()
            if self.current_episode < self.episode_length:
                self.current_episode+=1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()
                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                if self.timestep_counter % BATCH_SIZE == 0:
                    self.ppo_agent.learn()
                if self.done:
                    self.episode_counter += 1
                    t2 = datetime.now()
                    t3 = t2 - t1
                    print(f"Episode duration: {t3}")
                    return 1
            self.reset_environment()
            print(
                f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}"
            )
        # decide what to do in the end of an episode
        else:
            pass

    def testing(self, msg):
        if self.timestep_counter < TEST_TIMESTEPS:
            self.state = np.array(msg.data, dtype=np.float32)
            self.current_ep_reward = 0
            t1 = datetime.now()
            if self.current_episode < self.episode_length:
                self.current_episode+=1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()
                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                if self.done:
                    self.episode_counter += 1
                    t2 = datetime.now()
                    t3 = t2 - t1
                    print(f"Episode duration: {t3}")
                    return 1
            self.reset_environment()
            print(
                f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}"
            )
        # decide what to do in the end of an episode
        else:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
