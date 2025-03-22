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

        self.state = None
        self.action = None
        self.reward = 0
        self.done = False

        self.termination_reason = "unknown"
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
            
        if MODEL_LOAD and CHECKPOINT_FILE:
            self.run_dir = CHECKPOINT_FILE
            self.get_logger().info(f"📂 Resuming from checkpoint: {self.run_dir}")
            self.load_training_state(self.run_dir)
        else:
            self.create_new_run_dir()
            self.get_logger().info(f"🆕 Starting a new training run in: {self.run_dir}")

        self.get_logger().info(
            "PPOModelNode initialized,subscribed to data topic and PPO model loaded."
        )

    ##################################################################################################
    #                                       ACTION SELECTION
    ##################################################################################################

    def get_action(self, data):
        self.action, _ = self.ppo_agent.select_action(data)
        self.get_logger().info(f"Returned action: {self.action}")

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
        self.state = None
        self.action = None
        self.reward = 0
        self.done = False
        self.current_ep_reward = 0
        self.current_step_in_episode = 0
        self.termination_reason = "unknown"
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
            self.t1 = datetime.now()

            if self.current_step_in_episode < self.episode_length:
                self.current_step_in_episode += 1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()

                # Mark episode as done if episode_length reached
                if self.current_step_in_episode >= self.episode_length:
                    self.done = True
                    self.termination_reason = "episode_length"
                    
                # Later, if we detect crashes or other done causes and want to respawn, update:
                # self.termination_reason = "timeout"
                # self.termination_reason = "collision"
                # self.termination_reason = "goal_reached"
                
                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward

                if self.timestep_counter % BATCH_SIZE == 0:
                    actor_loss, critic_loss, entropy = self.ppo_agent.learn()
                    self.log_step_metrics(actor_loss, critic_loss, entropy)

                if self.done:
                    self.t2 = datetime.now()
                    self.get_logger().info(f"Episode duration: {self.t2 - self.t1}")

                    # Save state, log data
                    self.save_training_state(self.run_dir)
                    self.save_episode_trajectory()
                    self.log_episode_metrics()

                    self.get_logger().info(f"Checkpoint saved at {self.run_dir}")
                    self.reset_run()
                    self.episode_counter += 1
                    return 1

            self.get_logger().info(f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}")

        else:
            # End of training
            self.log_episode_metrics()
            self.save_training_state(self.run_dir)
            self.save_episode_trajectory()
            self.reset_run()
            self.episode_counter += 1

    def testing(self, msg):
        if self.timestep_counter < TEST_TIMESTEPS:
            self.state = np.array(msg.data, dtype=np.float32)
            self.current_ep_reward = 0
            self.t1 = datetime.now()

            if self.current_step_in_episode < self.episode_length:
                self.current_step_in_episode += 1
                self.get_action(self.state)
                self.publish_action()
                self.calculate_reward()

                # Mark episode as done if episode_length reached
                if self.current_step_in_episode >= self.episode_length:
                    self.done = True

                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward

                if self.done:
                    self.t2 = datetime.now()
                    self.get_logger().info(f"Episode duration: {self.t2 - self.t1}")

                    self.save_training_state(self.run_dir)
                    self.save_episode_trajectory()
                    self.log_episode_metrics()
                    self.get_logger().info(f"Checkpoint saved at {self.run_dir}")

                    self.reset_run()
                    self.episode_counter += 1
                return 1

            print(f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}")

        else:
            self.log_episode_metrics()
            self.save_training_state(self.run_dir)
            self.save_episode_trajectory()
            self.reset_run()
            self.episode_counter += 1


    ##################################################################################################
    #                                   CHECKPOINTING AND LOGGING
    ##################################################################################################

    
    def save_training_state(self, run_dir):
        """
        Saves the model state dicts, optimizers, and metadata.
        """
        self.ppo_agent.save_model_and_optimizers(run_dir)
        self.save_training_metadata(run_dir)

    def load_training_state(self, run_dir):
        """
        Loads model weights, optimizers, and training metadata.
        """
        self.ppo_agent.load_model_and_optimizers(run_dir)
        self.load_training_metadata(run_dir)

    def save_training_metadata(self):
        meta = {
            "episode_counter": self.episode_counter,
            "timestep_counter": self.timestep_counter,
            "action_std": self.ppo_agent.action_std,
            "current_ep_reward": self.current_ep_reward,
            "current_step_in_episode": self.current_step_in_episode,
        }
        try:
            meta_path = os.path.join(self.log_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            self.get_logger().info(f"✅ Metadata saved: {meta}")
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not saved: {e}")

    def load_training_metadata(self):
        try:
            meta_path = os.path.join(self.log_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.episode_counter = meta.get("episode_counter", 0)
                self.timestep_counter = meta.get("timestep_counter", 0)
                action_std = meta.get("action_std", ACTION_STD_INIT)
                self.ppo_agent.actor.set_action_std(action_std)
                self.current_ep_reward = meta.get("current_ep_reward", 0)
                self.current_step_in_episode = meta.get("current_step_in_episode", 0)
                self.get_logger().info(f"✅ Metadata loaded: {meta}")
            else:
                self.get_logger().warn(f"No meta.json found in {self.log_dir}. No metadata loaded.")
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not loaded: {e}")

    def log_step_metrics(self, actor_loss, critic_loss, entropy):
        log_file = os.path.join(self.log_dir, "training_log.csv")
        row = {
            "episode": self.episode_counter,
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

        self.summary_writer.add_scalar("episode", self.episode_counter, self.timestep_counter)
        self.summary_writer.add_scalar("Step", self.ppo_agent.learn_step_counter, self.timestep_counter)
        self.summary_writer.add_scalar("Timestep", self.timestep_counter, self.timestep_counter)
        self.summary_writer.add_scalar("Loss/actor", actor_loss, self.timestep_counter)
        self.summary_writer.add_scalar("Loss/critic", critic_loss, self.timestep_counter)
        self.summary_writer.add_scalar("Entropy", entropy, self.timestep_counter)
        self.summary_writer.add_scalar("Exploration/action_std", self.ppo_agent.action_std, self.timestep_counter)

    def log_episode_metrics(self):
        log_file = os.path.join(self.log_dir, "training_log.csv" if TRAIN else "testing_log.csv")
        row = {
            "episode": self.episode_counter,
            "timestep": self.timestep_counter,
            "episode_reward": self.current_ep_reward,
            "episode_length": self.current_step_in_episode,
            "action_std": self.ppo_agent.action_std,
            "termination_reason": self.termination_reason
        }
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        prefix = "Train" if TRAIN else "Test"
        self.summary_writer.add_scalar(f"{prefix}/Episode Duration (s)", (self.t2 - self.t1).total_seconds(), self.episode_counter)
        self.summary_writer.add_scalar(f"{prefix}/Episode Reward", self.current_ep_reward, self.episode_counter)
        self.summary_writer.add_scalar(f"{prefix}/Episode Length", self.current_step_in_episode, self.episode_counter)
        self.summary_writer.add_scalar(f"{prefix}/Action Std", self.ppo_agent.action_std, self.episode_counter)
        
    def save_episode_trajectory(self):
        os.makedirs(self.trajectory_dir, exist_ok=True)
        episode_file = os.path.join(self.trajectory_dir, f"episode_{self.episode_counter:04d}.csv")
        fieldnames = ["step", "timestep", "state", "action", "reward", "done"]
        try:
            with open(episode_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.ppo_agent.states)):
                    writer.writerow({
                        "step": i,
                        "timestep": self.timestep_counter - len(self.ppo_agent.states) + i + 1,
                        "state": np.array2string(self.ppo_agent.states[i], separator=","),
                        "action": np.array2string(self.ppo_agent.actions[i], separator=","),
                        "reward": self.ppo_agent.rewards[i],
                        "done": self.ppo_agent.dones[i]
                    })
            self.get_logger().info(f"📊 Episode trajectory saved to {episode_file}")
        except Exception as e:
            self.get_logger().error(f"❌ Could not save episode CSV: {e}")

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
        version_dir = os.path.join(PPO_CHECKPOINT_DIR, VERSION)
        os.makedirs(version_dir, exist_ok=True)

        existing = [d for d in os.listdir(version_dir) if d.startswith("run_")]
        serial = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d")
        self.run_name = f"run_{timestamp}_{serial:04d}"
        self.run_dir = os.path.join(version_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_dir = os.path.join(version_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.get_logger().info(f"Run directory created: {self.run_dir}")

        # Subfolders
        self.state_dict_dir = os.path.join(self.run_dir, "state_dict")
        os.makedirs(self.state_dict_dir, exist_ok=True)

        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.trajectory_dir = os.path.join(self.log_dir, "trajectories")
        os.makedirs(self.trajectory_dir, exist_ok=True)

        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)

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
