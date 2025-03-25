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

        self.collision_sub = self.create_subscription(
            String,
            "/collision_detected",
            self.collision_callback,
            10
        )

        self.location_sub = self.create_subscription(
            Float32MultiArray,
            "/carla/vehicle/location",
            self.location_callback,
            10
        )
        
        self.summary_writer = None
        if DETERMINISTIC_CUDNN:
            self.set_global_seed_and_determinism(SEED, DETERMINISTIC_CUDNN)
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
        self.needs_progress_reset = False

        self.stagnation_counter = 0
        
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
        
        self.get_logger().info(f"Initial action std: {self.ppo_agent.actor.log_std.exp().detach().cpu().numpy()}") # To check that log_std.exp() is around ACTION_STD_INIT:

        

        # Create client for the waypoints service
        self.waypoints_client = self.create_client(
            GetTrackWaypoints, 'get_track_waypoints'
        )
        
        # Check if the service is available
        while not self.waypoints_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for track waypoints service...')
        
        # Request track waypoints
        self.request_track_waypoints()


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
        """
        Improved reward function based on track progress delta:
        - Scaled positive reward for forward progress
        - Increasing stagnation penalty
        - Penalty for going backwards
        - Collision penalty
        - Lap completion bonus
        """
        # Core reward parameters
        progress_multiplier = 3000.0     # More balanced multiplier for progress
        base_time_penalty = -0.05      # Base penalty when not moving
        stagnation_factor = 0.02       # Increases penalty over time
        backwards_penalty = -1.0       # Penalty for going backwards
        collision_penalty = -5.0      # Stronger collision penalty
        lap_completion_bonus = 50.0    # Lap completion bonus
        
        # Initialize stagnation counter if not exists
        if not hasattr(self, 'stagnation_counter'):
            self.stagnation_counter = 0
        
        # Handle collision case first
        if self.collision:
            self.get_logger().info(f"Applied collision penalty: {collision_penalty}")
            self.reward = collision_penalty
            self.done = True
            self.stagnation_counter = 0  # Reset counter
            return
        
        # Calculate progress delta
        progress_delta = self.track_progress - self.prev_progress_distance
        
        # Handle wrap-around at 1.0 (lap completion)
        if progress_delta < -0.5:
            progress_delta = (1.0 - self.prev_progress_distance) + self.track_progress
            self.get_logger().info(f"Lap progress wrap-around detected: {progress_delta:.4f}")
        
        # Case 1: Moving backwards
        if progress_delta < -0.001:
            self.reward = backwards_penalty
            self.stagnation_counter = 0  # Reset counter
            self.get_logger().info(f"Moving backwards: {progress_delta:.6f}, reward = {backwards_penalty}")
            return
        
        # Case 2: Moving forward significantly
        elif progress_delta > 0.00001:
            # Reward directly proportional to progress made
            self.reward = progress_multiplier * progress_delta
            self.stagnation_counter = 0  # Reset counter
            self.get_logger().info(f"Moving forward: {progress_delta:.6f}, reward = {self.reward:.4f}")
        
        # Case 3: Not moving/minimal movement
        else:
            # Apply increasing stagnation penalty
            self.stagnation_counter += 1
            stagnation_penalty = base_time_penalty * (1 + self.stagnation_counter * stagnation_factor)
            self.reward = stagnation_penalty
            self.get_logger().info(f"Not moving: {progress_delta:.6f}, stagnation: {self.stagnation_counter}, reward = {stagnation_penalty:.4f}")
        
        # Add lap completion bonus if detected
        if self.lap_completed:
            self.reward += lap_completion_bonus
            self.lap_completed = False  # Reset flag
            self.stagnation_counter = 0  # Reset counter
            self.get_logger().info(f"Lap completion bonus applied: +{lap_completion_bonus}")
            
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
                    # self.ppo_agent.decay_action_std(self.episode_counter) # Trying a learnable log_std instead
                    return 1

            # self.get_logger().info(f"Episode: {self.episode_counter}, Timestep: {self.timestep_counter}, Reward: {self.current_ep_reward}")

        else:
            # End of training
            # self.ppo_agent.decay_action_std(self.episode_counter) # Trying a learnable log_std instead
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
        log_std = self.ppo_agent.actor.log_std.detach().cpu().tolist()  # convert tensor to list
        meta = {
            "episode_counter": self.episode_counter,
            "timestep_counter": self.timestep_counter,
            "log_std": log_std,
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
                self.current_ep_reward = meta.get("current_ep_reward", 0)
                self.current_step_in_episode = meta.get("current_step_in_episode", 0)
                log_std_list = meta.get("log_std")
                if log_std_list is not None:
                    log_std_tensor = torch.tensor(log_std_list, dtype=torch.float32, device=device)
                with torch.no_grad():
                    self.ppo_agent.actor.log_std.copy_(log_std_tensor)
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

    def location_callback(self, msg):
        """Process vehicle location updates and calculate track progress"""
        if len(self.track_waypoints) == 0:
            self.get_logger().warn("Track waypoints not available yet. Cannot calculate progress.")
            return
        
        # Store previous location for direct movement calculation
        previous_location = self.vehicle_location
        
        # Extract vehicle location
        self.vehicle_location = (msg.data[0], msg.data[1])
        
        # First call - just store the location
        if previous_location is None:
            self.get_logger().info(f"Initial vehicle position: ({self.vehicle_location[0]:.2f}, {self.vehicle_location[1]:.2f})")
            self.update_track_progress()
            return
            
        # Calculate direct movement vector for debugging
        movement_x = self.vehicle_location[0] - previous_location[0]
        movement_y = self.vehicle_location[1] - previous_location[1]
        distance_moved = math.sqrt(movement_x**2 + movement_y**2)
        
        # Log movement for debugging
        if distance_moved > 0.01:  # Only log significant movements
            self.get_logger().debug(f"Vehicle moved: {distance_moved:.3f}m")
        

        self.prev_progress_distance = self.track_progress

        # Calculate progress along the track
        self.update_track_progress()
        
        # Check for lap completion
        if self.track_progress < 0.1 and self.prev_progress_distance > 0.9:
            self.check_lap_completion()
    
    def update_track_progress(self):
        """Calculate vehicle's progress percentage along the track with high sensitivity"""
        if not self.vehicle_location or len(self.track_waypoints) < 2:
            return
        
        # STEP 1: Find position relative to track
        # Find the two closest waypoints to determine which segment we're on
        distances = []
        for i, waypoint in enumerate(self.track_waypoints):
            dist = self.calculate_distance(self.vehicle_location, waypoint)
            distances.append((dist, i))
        
        # Sort by distance
        distances.sort()
        closest_dist, closest_idx = distances[0]
        second_dist, second_idx = distances[1]
        
        # STEP 2: Determine our position between waypoints and track direction
        track_len = len(self.track_waypoints)
        
        # Let's identify the segment we're on (connecting the waypoints)
        # First check if these points are adjacent (handling wraparound)
        are_adjacent = (abs(closest_idx - second_idx) == 1) or (abs(closest_idx - second_idx) == track_len - 1)
        
        if not are_adjacent:
            # The closest points aren't adjacent - find best segment by checking all point pairs
            min_segment_dist = float('inf')
            segment_start_idx = 0
            
            for i in range(track_len):
                next_i = (i + 1) % track_len
                p1 = self.track_waypoints[i]
                p2 = self.track_waypoints[next_i]
                
                # Calculate distance from point to line segment
                segment_dist = self.point_to_segment_distance(self.vehicle_location, p1, p2)
                
                if segment_dist < min_segment_dist:
                    min_segment_dist = segment_dist
                    segment_start_idx = i
            
            # Now we have the closest segment
            closest_idx = segment_start_idx
            second_idx = (segment_start_idx + 1) % track_len
        
        # STEP 3: Calculate precise progress along track
        # If this is the first update, initialize the start point
        if self.start_point is None or self.needs_progress_reset:
            self.start_point = closest_idx
            self.prev_progress_distance = 0.0
            self.track_progress = 0.0
            self.needs_progress_reset = False
            self.get_logger().info(f"Progress tracking initialized at waypoint {closest_idx}/{track_len}")
            return
        
        # We need to know direction of track
        wp_current = self.track_waypoints[closest_idx]
        wp_next = self.track_waypoints[(closest_idx + 1) % track_len]
        
        # Project vehicle position onto the track segment to get precise location
        projection = self.project_point_to_segment(self.vehicle_location, wp_current, wp_next)
        segment_progress = self.calculate_distance(wp_current, projection) / self.calculate_distance(wp_current, wp_next)
        
        # Calculate overall progress (waypoint index + segment progress)
        if closest_idx >= self.start_point:
            base_progress = (closest_idx - self.start_point) / track_len
        else:
            # We've wrapped around the track
            base_progress = (track_len - self.start_point + closest_idx) / track_len
        
        # Add fractional progress within current segment
        segment_fraction = 1.0 / track_len
        raw_progress = base_progress + (segment_progress * segment_fraction)
        
        # Normalize to [0, 1] range
        raw_progress = raw_progress % 1.0
        
        # Check if this is a reasonable change from previous progress
        if self.prev_progress_distance is not None:
            progress_diff = raw_progress - self.prev_progress_distance
            
            # Handle wrap-around at 1.0
            if progress_diff < -0.5:  # We've wrapped from 0.99 to 0.01
                progress_diff = (1.0 - self.prev_progress_distance) + raw_progress
            elif progress_diff > 0.5:  # Unlikely large jump
                progress_diff = -((1.0 - raw_progress) + self.prev_progress_distance)
            
            # Debug log for significant progress changes
            if abs(progress_diff) > 0.01:
                self.get_logger().info(f"Progress change: {progress_diff:.4f} (from {self.prev_progress_distance:.4f} to {raw_progress:.4f})")
        
        # Update progress
        self.track_progress = raw_progress

        self.get_logger().info(f"Track progress: {self.track_progress:.8f}")
        
        # Log progress occasionally
        if int(self.track_progress * 100) % 10 == 0 and int(self.prev_progress_distance * 100) % 10 != 0:
            self.get_logger().info(f"Track progress: {self.track_progress:.4f}")

    def point_to_segment_distance(self, p, v, w):
        """Calculate the distance from point p to line segment vw"""
        # Length squared of segment
        l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
        if l2 == 0:  # v == w case
            return self.calculate_distance(p, v)
        
        # Consider the line extending the segment, parameterized as v + t (w - v)
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        
        # Projection falls on the segment
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        
        return self.calculate_distance(p, projection)
    

    def project_point_to_segment(self, p, v, w):
        """Project point p onto line segment vw and return the projection point"""
        # Length squared of segment
        l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
        if l2 == 0:  # v == w case
            return v
        
        # Consider the line extending the segment, parameterized as v + t (w - v)
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        
        # Projection falls on the segment
        return (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))

    def check_lap_completion(self):
        """Check if a lap was completed and handle the event"""
        if self.lap_completed:
            return
        
        # We've crossed back near the starting point after making progress
        if self.track_progress < 0.1 and self.prev_progress_distance > 0.9:
            self.lap_completed = True
            self.lap_count += 1
            
            # Calculate lap time if we're tracking it
            if self.lap_start_time:
                self.lap_end_time = datetime.now()
                self.lap_time = (self.lap_end_time - self.lap_start_time).total_seconds()
                self.get_logger().info(f"🏁 Lap {self.lap_count} completed in {self.lap_time:.2f} seconds!")
                
                # Add lap time to tensorboard
                if self.summary_writer:
                    self.summary_writer.add_scalar("Laps/time", self.lap_time, self.lap_count)
            
            # Reset for next lap
            self.lap_start_time = datetime.now()
            self.track_progress = 0.0
            self.prev_progress_distance = 0.0
            
            # Log the lap completion
            self.get_logger().info(f"🏁 Lap {self.lap_count} completed!")
            
            # Apply lap completion bonus in the next reward calculation
            self.termination_reason = "lap_completed"

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
        self.start_point = None 
        self.lap_completed = False
        self.lap_count = 0
        self.lap_start_time = None
        self.lap_end_time = None
        self.lap_time = None

        self.vehicle_location = None

        self.needs_progress_reset = True
        
        self.get_logger().info("Track progress variables fully reset for new vehicle")

        response.success = True
        response.message = "Vehicle control ready"
        return response

    def set_global_seed_and_determinism(self, seed=42, deterministic_cudnn=True):
        """
        Sets global seeds for full reproducibility and configures CuDNN for deterministic behavior.
        Call this at the start of training before any randomness or model initialization.
        """
        print(f"🔒 [SEEDING] Setting global seed to {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print("✅ CUDA is available.")

            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.enabled:
                if deterministic_cudnn:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    print("✅ CuDNN deterministic mode enabled.")
                else:
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = True
                    print("⚠️ CuDNN benchmarking mode enabled (faster but nondeterministic).")
            else:
                print("⚠️ CuDNN is not enabled or available.")
        else:
            print("⚠️ CUDA is not available. Running on CPU.")

        print("✅ Global seed and determinism setup complete.")
             
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
