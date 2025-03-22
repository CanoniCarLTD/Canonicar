from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
import sys
import os
import json
from datetime import datetime
import random
import csv
from torch.utils.tensorboard import SummaryWriter

from .ML import ppo_agent
from .ML import parameters
from .ML.parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")

        train = TRAIN
        self.current_step_in_episode = 0

        self.data_sub = self.create_subscription(
            Float32MultiArray,
            "/data_to_ppo",
            self.training if train else self.testing,
            10,
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

        if DETERMINISTIC_CUDNN:
            self.deterministic_cuda()
            # Seeding to reproduce the results
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

        self.create_new_run_dir()

        self.get_logger().info(
            "PPOModelNode initialized,subscribed to data topic and PPO model loaded."
        )

    ##################################################################################################
    #                                       ACTION SELECTION
    ##################################################################################################

    def get_action(self, data):
        self.action, _ = self.ppo_agent.select_action(data)
        self.get_logger().info(f"Retured action: {self.action}")

    ##################################################################################################
    #                                       ACTION PUBLISHING
    ##################################################################################################

    def publish_action(self):
        action_msg = Float32MultiArray()
        action_msg.data = self.action.tolist()
        self.action_publisher.publish(action_msg)
        self.get_logger().info(f"Published action: {action_msg.data}")

    ##################################################################################################
    #                                       REWARD FUNCTION
    ##################################################################################################

    def calculate_reward(self):
        # Implement reward calculation logic here
        self.reward = 1

    def store_transition(self):
        # Assuming you have access to the value and done flag
        value = self.ppo_agent.critic(
            torch.tensor(self.state, dtype=torch.float32).to(device).unsqueeze(0)
        ).item()
        self.ppo_agent.store_transition(
            self.state, self.action, 0, value, self.reward, self.done
        )

    ##################################################################################################
    #                                       RUN RESET
    ##################################################################################################

    def reset_run(self):
        ''' Reset the state, action, reward, done, current_ep_reward, and current_step_in_episode variables after an episode ends.
        '''
        # Implement your environment reset logic here
        self.state = None
        self.action = None
        self.reward = 0
        self.done = False
        self.current_ep_reward = 0
        self.current_step_in_episode = 0    
        self.get_logger().info(
            f"Episode {self.episode_counter} finished. Resetting."
        )

    ##################################################################################################
    #                                       TRAINING AND TESTING
    ##################################################################################################

    def training(self, msg):
        if self.timestep_counter < self.total_timesteps:
            self.state = np.array(msg.data, dtype=np.float32)
            self.current_ep_reward = 0
            t1 = datetime.now()
            
            if self.current_step_in_episode < self.episode_length:
                self.current_step_in_episode += 1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()
                self.store_transition()
                
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                
                if self.timestep_counter % BATCH_SIZE == 0:
                    actor_loss, critic_loss, entropy = self.ppo_agent.learn()
                    self.log_step_metrics(actor_loss, critic_loss, entropy)
                    
                if self.done:
                    t2 = datetime.now()
                    t3 = t2 - t1
                    print(f"Episode duration: {t3}")
                    self.log_episode_metrics()
                    self.save_checkpoint(self.run_dir)
                    print(f"Checkpoint saved at {self.run_dir}")
                    self.reset_run()
                    self.episode_counter += 1
                    return 1
            print(
                f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}"
            )
        else:
            self.log_episode_metrics()
            self.save_checkpoint(self.run_dir)
            self.reset_run()
            self.episode_counter += 1
            # decide what to do in the end of an episode
            # exit(2)

    def testing(self, msg):
        if self.timestep_counter < TEST_TIMESTEPS:
            self.state = np.array(msg.data, dtype=np.float32)
            self.current_ep_reward = 0
            t1 = datetime.now()
            
            if self.current_step_in_episode < self.episode_length:
                self.current_step_in_episode += 1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()
                self.store_transition()
                
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                
                if self.done:
                    t2 = datetime.now()
                    t3 = t2 - t1
                    print(f"Episode duration: {t3}")
                    self.log_episode_metrics()
                    self.save_checkpoint(self.run_dir)
                    print(f"Checkpoint saved at {self.run_dir}")
                    self.reset_run()
                    self.episode_counter += 1
                return 1
            print(
                f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}"
            )

        else:
            self.log_episode_metrics()
            self.save_checkpoint(self.run_dir)
            self.reset_run()
            self.episode_counter += 1
            # decide what to do in the end of an episode
            # exit(2)

    ##################################################################################################
    #                                   CHECKPOINTING AND LOGGING
    ##################################################################################################

    def save_checkpoint(self, run_dir):
        self.ppo_agent.save_checkpoint(run_dir)
        # continue to save other stuff...
        # Save metadata
        meta = {
            "episode_counter": self.episode_counter,
            "timestep_counter": self.timestep_counter,
            "action_std": self.ppo_agent.action_std,
            "current_ep_reward": self.current_ep_reward,
            "current_step_in_episode": self.current_step_in_episode,
        }
        try:
            meta_path = os.path.join(run_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            self.get_logger().info(f"✅ Metadata saved: {meta}")
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not saved: {e}")
        
    def load_checkpoint(self, run_dir):
        self.ppo_agent.load_checkpoint(run_dir)
        # continue to load other stuff...
        # Load metadata
        try:
            meta_path = os.path.join(run_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.episode_counter = meta.get("episode_counter", 0)
                self.timestep_counter = meta.get("timestep_counter", 0)
                self.ppo_agent.action_std = meta.get("action_std", ACTION_STD_INIT)
                self.current_ep_reward = meta.get("current_ep_reward", 0)
                self.current_step_in_episode = meta.get("current_step_in_episode", 0)
                self.get_logger().info(f"✅ Metadata loaded: {meta}")
            else:
                self.get_logger().warn(
                    f"No meta.json found in {run_dir}. No metadata loaded."
                )
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not loaded: {e}")

    def log_step_metrics(self, actor_loss, critic_loss, entropy):
        '''
        Logs the metrics into both a csv file and a TensorBoard after each step - for training only.
        The csv file is for easy access to the metrics and debugging.
        The TensorBoard logs are for visualization and monitoring the training process.
        '''
        log_file = os.path.join(self.run_dir, "training_log.csv")
        row = {
            "step": self.ppo_agent.learn_step_counter,
            "timestep": self.timestep_counter,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "action_std": self.ppo_agent.action_std
        }
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        
        # TensorBoard logging
        self.writer.add_scalar("Loss/actor", actor_loss, self.timestep_counter)
        self.writer.add_scalar("Loss/critic", critic_loss, self.timestep_counter)
        self.writer.add_scalar("Entropy", entropy, self.timestep_counter)
        self.writer.add_scalar("Exploration/action_std", self.ppo_agent.action_std, self.timestep_counter)
    
    def log_episode_metrics(self):
        '''
        Logs the metrics into a csv file after each episode - for both training and testing.
        '''
        log_file = os.path.join(self.run_dir, "training_log.csv" if TRAIN else "testing_log.csv")
        row = {
            "episode": self.episode_counter,
            "timestep": self.timestep_counter,
            "episode_reward": self.current_ep_reward,
            "episode_length": self.current_step_in_episode,
            "action_std": self.ppo_agent.action_std
        }
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        
        # TensorBoard logging
        prefix = "Train" if TRAIN else "Test"
        self.writer.add_scalar(f"{prefix}/Episode Reward", self.current_ep_reward, self.episode_counter)
        self.writer.add_scalar(f"{prefix}/Episode Length", self.current_step_in_episode, self.episode_counter)
        self.writer.add_scalar(f"{prefix}/Action Std", self.ppo_agent.action_std, self.episode_counter)
        
    ##################################################################################################
    #                                           UTILITIES
    ##################################################################################################

    def deterministic_cuda(self):
        """
        Set CuDNN to deterministic mode for reproducibility.
        """
        if torch.cuda.is_available():
            # Ensure CuDNN is enabled and set deterministic mode based on the user's preference
            if torch.backends.cudnn.enabled:
                print("CuDNN is enabled. Setting CuDNN to deterministic mode.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = (
                    False  # Disable auto-tuner for deterministic behavior
                )
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = (
                    True  # Enable auto-tuner for faster performance
                )
                
    def create_new_run_dir(self):
        """
        Creates for the current run - both a visually readable directory and a tensorboard log directory 
        """
        version_dir = os.path.join(PPO_CHECKPOINT_DIR, VERSION)
        os.makedirs(version_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"run_{timestamp}"
        self.run_dir = os.path.join(version_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.get_logger().info(f"Run directory created: {self.run_dir}")

        # SummaryWriter setup
        tb_base = os.path.join("tensorboard_logs", VERSION)
        os.makedirs(tb_base, exist_ok=True)

        mode = "train" if TRAIN else "test"
        self.tb_log_dir = os.path.join(tb_base, f"{self.run_name}_{mode}")
        self.summary_writer = SummaryWriter(log_dir=self.tb_log_dir)

        # Log hyperparameters
        hparams = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "lambda_gae": LAMBDA_GAE,
            "entropy_coef": ENTROPY_COEF,
            "policy_clip": POLICY_CLIP,
            "input_dim": PPO_INPUT_DIM,
            "episode_length": EPISODE_LENGTH,
            "total_timesteps": TOTAL_TIMESTEPS,
        }
        self.summary_writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % "\n".join([f"|{k}|{v}|" for k, v in hparams.items()])
        )

def shutdown_writer(self):
    self.summary_writer.close()
    self.get_logger().info("SummaryWriter closed.")
    
def main(args=None):
    rclpy.init(args=args)
    node = PPOModelNode()
    rclpy.spin(node)
    node.destroy_node()
    node.shutdown_writer()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
