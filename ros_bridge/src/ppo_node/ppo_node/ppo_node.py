from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from ros_interfaces.srv import VehicleReady, GetTrackWaypoints
import torch
import numpy as np
import sys
import os
import json
from datetime import datetime
import random
import csv
from torch.utils.tensorboard import SummaryWriter
import math


from .ML import ppo_agent
from .ML import parameters
from .ML.parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        
        print("Device: ", device)
        
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

        self.vehicle_ready_server = self.create_service(
            VehicleReady,
            'vehicle_ready',
            self.handle_vehicle_ready
        )
        
        self.summary_writer = None
        
        self.state = None
        self.action = None
        self.reward = 0.0
        self.done = False

        self.termination_reason = "unknown"
        self.collision = False
        self.track_progress = 0.0
        self.episode_counter = 0
        self.timestep_counter = 0
        self.total_timesteps = TOTAL_TIMESTEPS
        self.episode_length = EPISODE_LENGTH
        self.current_ep_reward = 0.0

        # Track waypoints related variables
        self.track_waypoints = []  # List of (x, y) tuples
        self.track_length = 0.0    # Total track length in meters
        self.closest_waypoint_idx = 0  # Index of closest waypoint
        self.start_point = None
        self.prev_progress_distance = 0.0
        self.lap_completed = False
        self.lap_count = 0
        self.vehicle_location = None
        self.lap_start_time = None
        self.lap_end_time = None
        self.lap_time = None
        
        # Initialize PPO Agent (loads from checkpoint if available)
        self.ppo_agent = ppo_agent.PPOAgent(summary_writer=self.summary_writer)
        self.get_logger().info(f"Model version: {VERSION}")
        self.get_logger().info(f"Checkpoint directory: {PPO_CHECKPOINT_DIR}")
        
        if MODEL_LOAD and CHECKPOINT_FILE:
            self.run_dir = CHECKPOINT_FILE
            if os.path.exists(self.run_dir):
                self.get_logger().info(f"📂 Resuming from checkpoint: {self.run_dir}")
                self.load_training_state(self.run_dir)
                self.log_dir = os.path.join(self.run_dir, "logs")
                # self.trajectory_dir = os.path.join(self.log_dir, "trajectories")
                self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
                self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)
                self.ppo_agent.summary_writer = self.summary_writer
            else:
                raise FileNotFoundError(f"❌ Checkpoint file not found: {self.run_dir}")
        else:
            self.create_new_run_dir()
            self.get_logger().info(f"🆕 Starting a new training run in: {self.run_dir}")
            


        self.collision_sub = self.create_subscription(
            String,
            "/collision_detected",
            self.collision_callback,
            10
        )

        # Create client for the waypoints service
        self.waypoints_client = self.create_client(
            GetTrackWaypoints, 'get_track_waypoints'
        )
        
        # Check if the service is available
        while not self.waypoints_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for track waypoints service...')
        
        # Request track waypoints
        self.request_track_waypoints()

        if DETERMINISTIC_CUDNN:
            self.deterministic_cuda()
            # Seeding to reproduce the results
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

        self.get_logger().info(
            "PPOModelNode initialized,subscribed to data topic and PPO model loaded."
        )

    ##################################################################################################
    #                                       ACTION SELECTION
    ##################################################################################################

    def get_action(self, data):
        self.state = data # might be redundant but just to make sure
        self.action, self.log_prob = self.ppo_agent.select_action(data)

    ##################################################################################################
    #                                       ACTION PUBLISHING
    ##################################################################################################

    def publish_action(self):
        action_msg = Float32MultiArray()
        action_msg.data = self.action.tolist()
        self.action_publisher.publish(action_msg)

    ##################################################################################################
    #                                       REWARD FUNCTION
    ##################################################################################################

    def calculate_reward(self):
        collision_penalty = -5.0       
        time_penalty = -0.1            
        progress_reward_factor = 300.0  
        finish_bonus = 50.0            
        
        if self.collision:
            self.reward = collision_penalty
            self.done = True # CHECK IF WE WANT TO DO THIS
        else:
            progress_reward = progress_reward_factor * self.track_progress

            if self.track_progress >= 1.0:
                progress_reward += finish_bonus

            self.reward = progress_reward + time_penalty
        
    ##################################################################################################
    #                                       STORE TRANSITION
    ##################################################################################################
    
    def store_transition(self):
        value = self.ppo_agent.critic(
            torch.tensor(self.state, dtype=torch.float32).to(device).unsqueeze(0)
        ).item()
        
        self.ppo_agent.store_transition(
            self.state, self.action, float(self.log_prob), value, self.reward, self.done
        )

    ##################################################################################################
    #                                           RUN RESET
    ##################################################################################################

    def reset_run(self):
        ''' Reset the state, action, reward, done, current_ep_reward, and current_step_in_episode variables after an episode ends.
        '''
        self.state = None
        self.action = None
        self.reward = 0.0
        self.done = False
        self.current_ep_reward = 0.0
        self.current_step_in_episode = 0.0
        self.collision = False
        self.lap_completed = False
        self.track_progress=0.0
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
                # self.termination_reason = "goal_reached"
                
                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                
                # Log reward at every step (add this line)
                self.summary_writer.add_scalar("Rewards/step_reward", self.reward, self.timestep_counter)
                self.summary_writer.add_scalar("Rewards/cumulative_reward", self.current_ep_reward, self.timestep_counter)
                
                if self.timestep_counter % LEARN_EVERY_N_STEPS == 0:
                    try:
                        # Add CUDA memory debug information
                        if torch.cuda.is_available():
                            self.get_logger().info(f"CUDA Memory: Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB, Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
                            # Force garbage collection
                            torch.cuda.empty_cache()
                        actor_loss, critic_loss, entropy = self.ppo_agent.learn()
                        self.get_logger().info(f"entropy: {entropy}")
                        self.log_step_metrics(actor_loss, critic_loss, entropy)
                    except RuntimeError as e:
                        # Handle CUDA errors more gracefully
                        self.get_logger().error(f"CUDA Error during learning: {e}")
                        # Try to recover - empty cache and continue
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if self.timestep_counter % SAVE_EVERY_N_TIMESTEPS == 0:
                    self.save_training_state(self.run_dir)
                if self.done:
                    self.t2 = datetime.now()
                    self.get_logger().info(f"Episode duration: {self.t2 - self.t1}")

                    # Save state, log data
                    self.save_training_state(self.run_dir)
                    # self.save_episode_trajectory()
                    self.log_episode_metrics()
                    self.get_logger().info(f"Checkpoint saved at {self.run_dir}")
                    self.reset_run()
                    self.episode_counter += 1
                    self.ppo_agent.decay_action_std(self.episode_counter)
                    return 1

            self.get_logger().info(f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}")

        else:
            # End of training
            self.ppo_agent.decay_action_std(self.episode_counter)
            self.log_episode_metrics()
            self.save_training_state(self.run_dir)
            # self.save_episode_trajectory()
            self.reset_run()
            self.episode_counter += 1

    def testing(self, msg):
        if self.timestep_counter < TEST_TIMESTEPS:
            self.state = np.array(msg.data, dtype=np.float32)
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
                # self.termination_reason = "goal_reached"
                
                self.store_transition()
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                
                # Log reward at every step (add this line)
                self.summary_writer.add_scalar("Rewards/step_reward", self.reward, self.timestep_counter)
                self.summary_writer.add_scalar("Rewards/cumulative_reward", self.current_ep_reward, self.timestep_counter)
            

                if self.done:
                    self.t2 = datetime.now()
                    self.get_logger().info(f"Episode duration: {self.t2 - self.t1}")

                    self.save_training_state(self.run_dir)
                    # self.save_episode_trajectory()
                    self.log_episode_metrics()
                    self.get_logger().info(f"Checkpoint saved at {self.run_dir}")

                    self.reset_run()
                    self.episode_counter += 1
                return 1

            print(f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}")

        else:
            self.log_episode_metrics()
            self.save_training_state(self.run_dir)
            # self.save_episode_trajectory()
            self.reset_run()
            self.episode_counter += 1


    ##################################################################################################
    #                                   CHECKPOINTING AND LOGGING
    ##################################################################################################

    
    def save_training_state(self, run_dir):
        """
        Saves the model state dicts, optimizers, and metadata.
        """
        state_dict_dir = os.path.join(run_dir, "state_dict")
        os.makedirs(state_dict_dir, exist_ok=True)
        self.ppo_agent.save_model_and_optimizers(state_dict_dir)
        self.save_training_metadata(state_dict_dir)

    def load_training_state(self, run_dir):
        """
        Loads model weights, optimizers, and training metadata.
        """
        state_dict_dir = os.path.join(run_dir, "state_dict")
        self.ppo_agent.load_model_and_optimizers(state_dict_dir)
        self.load_training_metadata(state_dict_dir)

    def save_training_metadata(self, state_dict_dir):
        meta = {
            "episode_counter": self.episode_counter,
            "timestep_counter": self.timestep_counter,
            "action_std": self.ppo_agent.action_std,
            "current_ep_reward": self.current_ep_reward,
            "current_step_in_episode": self.current_step_in_episode,
        }
        try:
            meta_path = os.path.join(state_dict_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            self.get_logger().info(f"✅ Metadata saved: {meta}")
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not saved: {e}")

    def load_training_metadata(self, state_dict_dir):
        try:
            meta_path = os.path.join(state_dict_dir, "meta.json")
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
                self.get_logger().warn(f"No meta.json found in {state_dict_dir}. No metadata loaded.")
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
            "action_std": self.ppo_agent.action_std,
            "step_reward": self.reward,  # Add step reward to log
            "cumulative_reward": self.current_ep_reward
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
        
    # def save_episode_trajectory(self):
    #     # IT THINK IT IS RUDUNDENT. PLEASE TELL ME WHAT YOU THINK.
    #     os.makedirs(self.trajectory_dir, exist_ok=True)
    #     episode_file = os.path.join(self.trajectory_dir, f"episode_{self.episode_counter:04d}.csv")
    #     fieldnames = ["step", "timestep", "state", "action", "reward", "done"]
    #     try:
    #         with open(episode_file, "w", newline="") as f:
    #             writer = csv.DictWriter(f, fieldnames=fieldnames)
    #             writer.writeheader()
    #             for i in range(len(self.ppo_agent.states)):
    #                 if i % 512 == 0:  # Save every 512 steps
    #                     writer.writerow({
    #                         "step": i,
    #                         "timestep": self.timestep_counter - len(self.ppo_agent.states) + i + 1,
    #                         "state": np.array2string(self.ppo_agent.states[i], separator=","),
    #                         "action": np.array2string(self.ppo_agent.actions[i], separator=","),
    #                         "reward": self.ppo_agent.rewards[i],
    #                         "done": self.ppo_agent.dones[i]
    #                     })
    #         self.get_logger().info(f"📊 Episode trajectory saved to {episode_file}")
    #     except Exception as e:
    #         self.get_logger().error(f"❌ Could not save episode CSV: {e}")

    ##################################################################################################
    #                                           UTILITIES
    ##################################################################################################

    def collision_callback(self, msg):
        """Handle collision notifications from topic"""
        if not self.collision:  # Prevent duplicate penalties
            self.collision = True
            self.get_logger().warn(f"Collision detected: {msg.data}")
            self.termination_reason = "collision"
            
            # Update reward immediately
            self.calculate_reward()
            
            # Log the collision penalty
            self.get_logger().info(f"Applied collision penalty: {self.reward}")

    def request_track_waypoints(self):
        """Request track waypoints from the map loader service"""
        self.get_logger().info("Requesting track waypoints...")
        
        request = GetTrackWaypoints.Request()
        future = self.waypoints_client.call_async(request)
        
        # Add a callback for when the service call completes
        future.add_done_callback(self.process_waypoints_response)
    
    def process_waypoints_response(self, future):
        """Process the waypoints received from the service"""
        try:
            response = future.result()
            
            # Extract waypoints from the response
            waypoints_x = response.waypoints_x
            waypoints_y = response.waypoints_y
            self.track_length = response.track_length
            
            # Combine into list of (x, y) tuples
            self.track_waypoints = list(zip(waypoints_x, waypoints_y))
            
            self.get_logger().info(f"Received {len(self.track_waypoints)} waypoints")
            self.get_logger().info(f"Track length: {self.track_length:.2f} meters")
        
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def location_callback(self, msg):
        """Process vehicle location data"""
        self.vehicle_location = msg.data
        
        if not self.track_waypoints:
            self.get_logger().warn("No track waypoints available yet")
            return
            
        if self.start_point is None:
            self.start_point = self.vehicle_location
            self.get_logger().info(f"Start position recorded: {self.start_point}")
            self.lap_start_time = self.get_clock().now()
            self.get_logger().info(f"Lap timer started at: {self.lap_start_time}")
            return
        
        # Calculate progress around the track
        self.update_track_progress()
        
        # Check if we've completed a lap (close to start point and made progress around track)
        distance_to_start = self.calculate_distance(self.start_point, self.vehicle_location)
        if distance_to_start < 5.0 and self.track_progress > 0.9:
            if not self.lap_completed:
                self.lap_completed = True
                self.lap_count += 1
                self.lap_end_time = self.get_clock().now()
                self.lap_time = self.lap_end_time - self.lap_start_time
                self.lap_start_time = self.get_clock().now()
                self.get_logger().info(f"Lap {self.lap_count} completed!")
                self.get_logger().info(f"Lap time: {self.lap_time} seconds")
                # Reset progress for next lap
                self.track_progress = 0.0
                self.prev_progress_distance = 0.0
        else:
            self.lap_completed = False

    def update_track_progress(self):
        """Update the track progress based on current position"""
        if not self.track_waypoints or self.vehicle_location is None:
            return
            
        # Find the closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (wx, wy) in enumerate(self.track_waypoints):
            dist = ((self.vehicle_location[0] - wx)**2 + 
                    (self.vehicle_location[1] - wy)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Calculate distance traveled along track since start
        if closest_idx > self.closest_waypoint_idx:
            # Moving forward along the track
            segments_traveled = closest_idx - self.closest_waypoint_idx
            # Calculate distance between these waypoints
            distance = 0.0
            for i in range(self.closest_waypoint_idx, closest_idx):
                next_i = i + 1
                x1, y1 = self.track_waypoints[i]
                x2, y2 = self.track_waypoints[next_i]
                segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                distance += segment_length
                
            self.prev_progress_distance += distance
            
        elif closest_idx < self.closest_waypoint_idx:
            # Crossed the finish line or moving backwards
            if self.closest_waypoint_idx > len(self.track_waypoints) * 0.8 and closest_idx < len(self.track_waypoints) * 0.2:
                # Likely crossed the finish line - add remaining distance
                distance = 0.0
                for i in range(self.closest_waypoint_idx, len(self.track_waypoints)):
                    next_i = (i + 1) % len(self.track_waypoints)
                    x1, y1 = self.track_waypoints[i]
                    x2, y2 = self.track_waypoints[next_i]
                    segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    distance += segment_length
                
                # Plus distance from start to current waypoint
                for i in range(0, closest_idx):
                    next_i = i + 1
                    x1, y1 = self.track_waypoints[i]
                    x2, y2 = self.track_waypoints[next_i]
                    segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
                    distance += segment_length
                    
                self.prev_progress_distance += distance
            # Otherwise, we're moving backwards or jumped positions
        
        # Update closest waypoint index
        self.closest_waypoint_idx = closest_idx
        
        # Calculate progress as a ratio of distance traveled to total track length
        self.track_progress = min(1.0, self.prev_progress_distance / self.track_length)
        
        # Every few seconds, log the progress
        if self.track_progress > 0 and int(self.get_clock().now().nanoseconds / 1e9) % 5 == 0:
            self.get_logger().info(f"Track progress: {self.track_progress:.2f} ({self.prev_progress_distance:.1f}m / {self.track_length:.1f}m)")

    def calculate_distance(self, point1, point2):
        if point1 is None or point2 is None:
            return float("inf")
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
        
    def handle_vehicle_ready(self, request, response):
        """Service handler when spawn_vehicle_node notifies that a vehicle is ready"""
        self.get_logger().info(f'Received vehicle_ready notification: {request.vehicle_id}')
        self.collision = False
        # Reset track progress variables for new vehicle
        self.track_progress = 0.0
        self.prev_progress_distance = 0.0
        self.closest_waypoint_idx = 0
        self.lap_completed = False
        
        return response

    def deterministic_cuda(self):
        """
        Set CuDNN to deterministic mode for reproducibility.
        """
        if torch.cuda.is_available():
            print ("✅ CUDA is available.")
            # Ensure CuDNN is enabled and set deterministic mode based on the user's preference
            if torch.backends.cudnn.enabled:
                print("✅ CuDNN is enabled. Setting CuDNN to deterministic mode.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = (
                    False  # Disable auto-tuner for deterministic behavior
                )
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = (
                    True  # Enable auto-tuner for faster performance
                )
        else:
            print("CUDA is not available.")
                 
    def create_new_run_dir(self, base_dir=None):
        version_dir = os.path.join(PPO_CHECKPOINT_DIR, VERSION)
        os.makedirs(version_dir, exist_ok=True)
    
        if base_dir:
            # Extract the serial number from the base_dir and increment it
            base_serial = int(base_dir.split('_')[-1])
            serial = base_serial + 1
        else:
            existing = [d for d in os.listdir(version_dir) if d.startswith("run_")]
            serial = len(existing) + 1
    
        timestamp = datetime.now().strftime("%Y%m%d")
        self.run_name = f"run_{timestamp}_{serial:04d}"
        self.run_dir = os.path.join(version_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
    
        # Subfolders
        self.state_dict_dir = os.path.join(self.run_dir, "state_dict")
        os.makedirs(self.state_dict_dir, exist_ok=True)
    
        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
    
        # self.trajectory_dir = os.path.join(self.log_dir, "trajectories")
        # os.makedirs(self.trajectory_dir, exist_ok=True)
    
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.ppo_agent.summary_writer = self.summary_writer
    
        # Log hyperparameters
        hparams = {
            "actor learning_rate": ACTOR_LEARNING_RATE,
            "critic learning_rate": CRITIC_LEARNING_RATE,
            "learn_every_N_steps": LEARN_EVERY_N_STEPS,
            "minibatch_size": MINIBATCH_SIZE,
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_training_state(node.run_dir)
        # node.save_episode_trajectory()
    finally:
        node.destroy_node()
        node.shutdown_writer()
        rclpy.shutdown()
    


if __name__ == "__main__":
    main()
