from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix  # type: ignore
import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from std_msgs.msg import Float32MultiArray, String  # type: ignore
from ros_interfaces.srv import VehicleReady, GetTrackWaypoints
import torch
import numpy as np
import os
import json
from datetime import datetime, time
import random
import csv
from torch.utils.tensorboard import SummaryWriter
import math
import psutil
import json
from std_msgs.msg import String  # type: ignore
from datetime import datetime
import threading
from queue import Queue
import time

from .ML import ppo_agent
from .ML import parameters
from .ML.parameters import *

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # might reduce performance time! Uncomment for debugging CUDA errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOModelNode(Node):
    def __init__(self):
        super().__init__("ppo_model_node")
        self.logger = self.get_logger()
        self.get_logger().info(f"Device is:{device} ")

        self.current_sim_state = "INITIALIZING"

        self.task_queue = Queue()
        self.worker_thread = threading.Thread(
            target=self._process_background_tasks, daemon=True
        )
        self.worker_thread.start()
        self.get_logger().info("Background worker thread started")

        self.metrics_lock = threading.Lock()
        self.model_lock = threading.Lock()

        self.train = TRAIN

        self.current_step_in_episode = 0

        self.data_sub = self.create_subscription(
            Float32MultiArray,
            "/data_to_ppo",
            self.training if self.train else self.testing,
            10,
        )
        # Action publisher (Steering, Throttle)
        self.action_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle_control", 10
        )

        self.episode_metrics_pub = self.create_publisher(
            String, "/training/episode_metrics", 10
        )

        self.performance_metrics_pub = self.create_publisher(
            String, "/training/performance_metrics", 10
        )

        self.error_logs_pub = self.create_publisher(String, "/training/error_logs", 10)

        self.location_sub = self.create_subscription(
            Float32MultiArray, "/carla/vehicle/location", self.location_callback, 10
        )

        self.waypoint_check_timer = self.create_timer(1.0, self.check_waypoint_needs)

        self.summary_writer = None

        seed = 0  # Fixed seed for reproducibility
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # torch.autograd.set_detect_anomaly(True) # slows things down, so only enable it for debugging.

        self.get_logger().info(f"Model version: {VERSION}")
        self.get_logger().info(f"Checkpoint directory: {PPO_CHECKPOINT_DIR}")

        self.state = None
        self.action = [0.0, 0.0]
        self.reward = 0.0
        self.done = False

        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None

        self.t2 = None
        self.t1 = None

        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.entropy = 0.0

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
        self.heading_buffer = []  # List of headings for each waypoint
        self.track_length = 0.0  # Total track length in meters
        self.closest_waypoint_idx = 0  # Index of closest waypoint
        self.start_point = None
        self.prev_progress_distance = 0.0
        self.lap_completed = False
        self.lap_count = 0
        self.vehicle_location = None
        self.vehicle_rotation = None
        self.lap_start_time = None
        self.lap_end_time = None
        self.lap_time = None
        self.stagnation_counter = 0
        self.needs_track_waypoints = False
        self.ready_to_collect = False
        self.lateral_deviation = 0.0
        self.heading_deviation = 0.0
        self.vehicle_heading = 0.0
        self._prev_throttle = 0.0
        self._prev_steer = 0.0

        self.last_processed_frame_id = -1
        self.max_buffer_size = 100  # Prevent memory issues
        self.state_buffer = []

        self.episode_start_time = datetime.now()
        self.current_step_in_episode = 0
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.last_entropy = None

        self.prev_loc = None
        self.prev_t = None
        self.current_speed = 0.0

        self.state_subscription = self.create_subscription(
            String, "/simulation/state", self.handle_system_state, 10
        )

        self.waypoint_client = self.create_client(
            GetTrackWaypoints, "get_track_waypoints"
        )

        self.episode_complete_pub = self.create_publisher(
            String, "/episode_complete", 10
        )

        # Check if the service is available
        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for track waypoints service...")
        self.needs_progress_reset = False

        self.ppo_agent = ppo_agent.PPOAgent(
            summary_writer=self.summary_writer, logger=self.get_logger()
        )

        if MODEL_LOAD:
            if CHECKPOINT_FILE and LOAD_STATE_DICT_FROM_RUN:
                raise ValueError(
                    "MODEL_LOAD is True but both CHECKPOINT_FILE and LOAD_STATE_DICT_FROM_RUN are set. Please choose one. \nTo continue a run, set CHECKPOINT_FILE to the wanted run directory and set LOAD_STATE_DICT_FROM_RUN to None. \nTo load state dict from another run, set CHECKPOINT_FILE to None, set LOAD_STATE_DICT_FROM_RUN to the wanted run directory, and don't froget to set the wanted version in VERSION.\n See more info in the README."
                )
            if CHECKPOINT_FILE:
                # Resume entire run from same version
                self.run_dir = CHECKPOINT_FILE
                if os.path.exists(self.run_dir):
                    self.get_logger().info(
                        f"📂 Resuming full run from file: {self.run_dir}"
                    )
                    self.load_training_state(self.run_dir)
                    self.log_dir = os.path.join(self.run_dir, "logs")
                    self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")
                    self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)
                    self.ppo_agent.summary_writer = self.summary_writer
                else:
                    raise FileNotFoundError(
                        f"❌ Checkpoint file not found: {self.run_dir}"
                    )

            elif LOAD_STATE_DICT_FROM_RUN:
                # Load weights from another run into a new one
                self.create_new_run_dir(load_from_run=LOAD_STATE_DICT_FROM_RUN)
                state_dict_dir_path = os.path.join(
                    LOAD_STATE_DICT_FROM_RUN, "state_dict"
                )  # add /state_dict to the path
                self.ppo_agent.load_model_and_optimizers(state_dict_dir_path)
                self.get_logger().info(
                    f"🆕 Started new run ({VERSION}), loaded weights from: {LOAD_STATE_DICT_FROM_RUN}"
                )

            else:
                raise ValueError(
                    "MODEL_LOAD is True but neither CHECKPOINT_FILE nor LOAD_STATE_DICT_FROM_RUN was set."
                )

        else:
            self.create_new_run_dir()
            self.get_logger().info(f"🆕 Started new training run in: {self.run_dir}")

        self.log_hyperparameters()

        while not self.waypoint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for track waypoints service...")

        self.request_track_waypoints()

        self.get_logger().info(
            "PPOModelNode initialized, Subscribed to data topic and PPO model loaded."
        )

    ##################################################################################################
    #                                       ACTION PUBLISHING
    ##################################################################################################

    def publish_action(self, action=None):
        if self.current_sim_state in ["RESPAWNING", "MAP_SWAPPING"]:
            self.get_logger().debug("Skipping action publishing during transition")
            return

        if action is None:
            action = self.action

        action_list = action.tolist()

        if len(action_list) == 2:
            steer, throttle = action_list
            brake = 0.0
        elif len(action_list) == 3:
            steer, throttle, brake = action_list
        else:
            self.get_logger().error(f"Invalid action format: {action_list}")
            return

        # unpack
        steer, throttle = float(steer), float(throttle)

        # Idrees' mapping and clamps
        steer = max(min(steer, 1.0), -1.0)
        throttle = (throttle + 1.0) / 2.0
        throttle = max(min(throttle, 0.25), 0.0)
        steer = self._prev_steer*0.9 + steer*0.1
        throttle = self._prev_throttle*0.9 + throttle*0.1
        self._prev_steer = steer
        self._prev_throttle = throttle

        action_msg = Float32MultiArray()
        action_msg.data = [steer, throttle, brake]

        # self.get_logger().info(f"Publishing | Steer: {discrete_steer} | Throttle: {discrete_throttle}")

        self.action_publisher.publish(action_msg)

    ##################################################################################################
    #                                       REWARD FUNCTION
    ##################################################################################################

    def calculate_reward(self):
        if self.collision:
            self.reward = -40.0
            self.done = True
            return

        progress_delta = self.track_progress - self.prev_progress_distance
        progress_reward = 100.0 * max(0.0, progress_delta)

        time_penalty = -0.01

        dev_pen = -1.0 * min(self.lateral_deviation ** 2, 4.0)
        ang_pen = -0.5 * min(abs(self.heading_deviation), np.pi/4)

        self.reward = float(progress_reward + time_penalty + dev_pen + ang_pen)

    ##################################################################################################
    #                                       STORE TRANSITION
    ##################################################################################################


    def store_transition(self, state=None, action=None, log_prob=None, reward=None, done=None):
        state = self.state if state is None else state
        if self.current_sim_state in ["RESPAWNING", "MAP_SWAPPING"] or state is None:
            self.get_logger().debug("Skipping transition storage during respawn or with None state")
            return

        action = self.action if action is None else action
        log_prob = self.log_prob if log_prob is None else log_prob
        reward = self.reward if reward is None else reward
        done = self.done if done is None else done

        assert action is not None, "Trying to store transition with None action!"

        # Normalize log_prob input type; PPOAgent.store_transition handles either
        if isinstance(log_prob, torch.Tensor) and log_prob.numel() == 1:
            lp = log_prob  # keep as tensor
        else:
            lp = float(log_prob)

        self.ppo_agent.store_transition(state, action, lp, reward, done)


    ##################################################################################################
    #                                           RESET RUN
    ##################################################################################################

    def reset_run(self):
        """Reset the state, action, reward, done, current_ep_reward, and current_step_in_episode variables after an episode ends."""
        self.state = None
        self.action = None
        self.reward = 0.0
        self.done = False
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.current_ep_reward = 0.0
        self.current_step_in_episode = 0.0
        self.stagnation_counter = 0
        self.lateral_deviation = 0.0
        self.heading_deviation = 0.0
        self.vehicle_heading = 0.0
        self.heading_buffer = []
        self.collision = False
        self.lap_completed = False
        self.track_progress = 0.0
        self.prev_progress_distance = 0.0
        self._prev_throttle = 0.0
        self._prev_steer = 0.0
        self.vehicle_location = None
        self.vehicle_rotation = None
        self.lap_start_time = None
        self.lap_end_time = None
        self.lap_time = None
        self.needs_track_waypoints = False
        self.current_speed = 0.0
        self.last_processed_frame_id = -1
        self.max_buffer_size = 100
        self.state_buffer.clear()
        
        
        self.termination_reason = "unknown"
        self.get_logger().info(f"Episode {self.episode_counter} finished. Resetting.")

    ##################################################################################################
    #                                           THREADING
    ##################################################################################################

    def _process_background_tasks(self):
        """Worker thread that processes background tasks from the queue"""
        self.get_logger().info("Background task processor started")
        while True:
            try:
                task, args, kwargs = self.task_queue.get()
                if not callable(task):
                    self.get_logger().error(f"Task is not callable: {task}")
                    continue
                # Execute the task
                start_time = time.time()
                task(*args, **kwargs)
                duration = time.time() - start_time
                if duration > 0.1:
                    self.get_logger().debug(
                        f"Background task {task.__name__} took {duration:.4f}s"
                    )

                self.task_queue.task_done()
            except Exception as e:
                self.get_logger().error(f"Error in background task: {e}")

    def log_system_metrics(self):
        """Queue system metrics logging to run in background"""
        self._queue_background_task(self._log_system_metrics_impl)

    def log_every_learn_step_metrics(self):
        """Queue step metrics logging to run in background"""
        self._queue_background_task(self._log_every_learn_step_metrics_impl)

    def _queue_background_task(self, task, *args, **kwargs):
        """Queue a task to run in the background thread"""
        if task is None or not callable(task):
            self.get_logger().error(f"Invalid task added to queue: {task}")
            return
        self.get_logger().debug(f"Task added to queue: {task.__name__}")
        self.task_queue.put((task, args, kwargs))
        
    ##################################################################################################
    #                                       TRAINING
    ##################################################################################################

    def training(self, msg):
        if not hasattr(self, "ready_to_collect") or not self.ready_to_collect:
            return

        if self.timestep_counter < self.total_timesteps:
            self.state = np.array(msg.data[1:1+PPO_INPUT_DIM], dtype=np.float32)

            self.calculate_reward()  # compute reward for previous transition
            reward_to_store = self.reward
            done_to_store = self.done
            if (
                self.prev_state is not None
                and self.prev_action is not None
                and self.prev_log_prob is not None
            ):
                self.store_transition(
                    state=self.prev_state,
                    action=self.prev_action,
                    log_prob=self.prev_log_prob,
                    reward=reward_to_store,
                    done=done_to_store,
                )

            self.t1 = datetime.now()

            if self.current_step_in_episode < self.episode_length:
                self.current_step_in_episode += 1
                # self.get_action(self.state)
                self.action, self.log_prob = self.ppo_agent.select_action(self.state)

                self.publish_action()

                self.prev_state = self.state
                self.prev_action = self.action
                self.prev_log_prob = self.log_prob

                # Mark episode as done if episode_length reached
                if self.current_step_in_episode >= self.episode_length:
                    self.done = True
                    self.termination_reason = "episode_length"
                    completion_msg = String()
                    completion_msg.data = "episode_length"
                    self.episode_complete_pub.publish(completion_msg)
                    self.get_logger().info("Published episode completion notification")

                self.timestep_counter += 1
                self.current_ep_reward += self.reward

                if self.timestep_counter % LEARN_EVERY_N_STEPS == 0:
                    try:
                        self.get_logger().info("\nLearning step started\n")
                        # stop vehicle before learning
                        stop_action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        self.publish_action(stop_action)
                        learn_start_time = time.time()
                        self.actor_loss, self.critic_loss, self.entropy = (
                            self.ppo_agent.learn()
                        )
                        learn_duration = time.time() - learn_start_time
                        self.get_logger().info(
                            "Learn duration: {:.4f}s".format(learn_duration)
                        )
                        self.log_every_learn_step_metrics()
                        self.log_system_metrics()
                    except RuntimeError as e:
                        self.get_logger().error(f"CUDA Error during learning: {e}")
                if self.timestep_counter % SAVE_EVERY_N_TIMESTEPS == 0:
                    self.save_training_state(self.run_dir)
                if self.done:
                    self.t2 = datetime.now()
                    self.get_logger().info(f"Episode duration: {self.t2 - self.t1}")
                    self.save_training_state(self.run_dir)
                    self.log_episode_metrics()
                    self.log_system_metrics()
                    self.get_logger().info(f"Checkpoint saved at {self.run_dir}")
                    self.reset_run()

                    self.summary_writer.flush()  # Added v3.1.1, check that it doesn't break anything
                    # torch.cuda.empty_cache()

                    self.episode_counter += 1
                    return 1
            self.get_logger().info(
                f"Episode: {self.episode_counter}, Timestep: {self.current_step_in_episode}, Reward: {self.current_ep_reward}"
            )

        else:
            # End of training
            self.log_episode_metrics()
            self.save_training_state(self.run_dir)
            self.reset_run()
            self.episode_counter += 1

    ##################################################################################################
    #                                       TESTING
    ##################################################################################################

    def testing(self, msg):
        if not hasattr(self, "ready_to_collect") or not self.ready_to_collect:
            return
        
        try:
            # Extract frame_id from the first element of the message
            if len(msg.data) < 2:  # Should have at least frame_id + some state data
                self.get_logger().warn("Received message with insufficient data")
                return
                
            current_frame_id = int(msg.data[0])
            state_data = np.array(msg.data[1:], dtype=np.float32)  # Skip frame_id
            
            # Only process if this is a new frame
            if current_frame_id != self.last_processed_frame_id:
                # Add to buffer as tuple (frame_id, state_data)
                self.state_buffer.append((current_frame_id, state_data))
                self.last_processed_frame_id = current_frame_id
                
                # Limit buffer size to prevent memory issues
                if len(self.state_buffer) > self.max_buffer_size:
                    self.state_buffer.pop(0)  # Remove oldest
                    
                self.get_logger().info(f"Added new state with frame_id {current_frame_id} to buffer")
            else:
                self.get_logger().info(f"Ignoring duplicate frame_id {current_frame_id}")
                return
            
            # Process states from buffer
            if self.state_buffer and self.current_step_in_episode < self.episode_length:
                # Pop the oldest state from buffer
                frame_id, state = self.state_buffer.pop(0)
                self.state = state
                
                self.current_step_in_episode += 1
                
                # Get action for this unique state
                self.action, self.log_prob = self.ppo_agent.select_action(self.state)
                self.publish_action()
                
                # Handle episode completion
                if self.current_step_in_episode >= self.episode_length:
                    self.done = True
                    self.termination_reason = "episode_length"
                
                self.timestep_counter += 1
                self.current_ep_reward += self.reward
                
                if self.done:
                    self.reset_run()
                    # torch.cuda.empty_cache()
                    self.episode_counter += 1
                    
                self.get_logger().debug(f"Processed state from frame_id {frame_id}")
                return 1
                
        except Exception as e:
            self.get_logger().error(f"Error in testing method: {e}")
            return 0

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
        try:
            action_var = self.ppo_agent.policy.cov_var.detach().cpu().tolist()

            meta = {
                "episode_counter": self.episode_counter,
                "timestep_counter": self.timestep_counter,
                "action_var": action_var,
                "current_ep_reward": self.current_ep_reward,
                "current_step_in_episode": self.current_step_in_episode,
                "learn_step_counter": self.ppo_agent.learn_step_counter,
                "entropy_coef": self.ppo_agent.entropy_coef,
            }

            meta_path = os.path.join(state_dict_dir, "meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            self.get_logger().info(f"Metadata saved: {meta}")
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
                self.ppo_agent.learn_step_counter = meta.get("learn_step_counter", 0)
                self.ppo_agent.entropy_coef = meta.get("entropy_coef", ENTROPY_COEF)

                action_var_list = meta.get("action_var")
                if action_var_list is not None:
                    var_tensor = torch.tensor(action_var_list, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        for ac in (self.ppo_agent.policy, self.ppo_agent.old_policy):
                            ac.cov_var.copy_(var_tensor)
                            ac.cov_mat.copy_(torch.diag(ac.cov_var).unsqueeze(0))

                self.get_logger().info(f"Metadata loaded: {meta}")
            else:
                self.get_logger().warn(f"No meta.json found in {state_dict_dir}. No metadata loaded.")
        except Exception as e:
            self.get_logger().error(f"❌ Metadata not loaded: {e}")


    def log_hyperparameters(self):
        hparams = {
            "episode_length": EPISODE_LENGTH,
            "learn_every_N_steps": LEARN_EVERY_N_STEPS,
            "learn_epochs": NUM_EPOCHS,
            "learning_rate": PPO_LEARNING_RATE,
            "gamma": GAMMA,
            "entropy_coef": ENTROPY_COEF,
            "policy_clip": POLICY_CLIP,
            "input_dim": PPO_INPUT_DIM,
        }

        markdown_table = "|param|value|\n|-|-|\n" + "\n".join(
            [f"|{key}|{value}|" for key, value in hparams.items()]
        )

        self.summary_writer.add_text(
            "hyperparameters/text_summary", markdown_table, global_step=0
        )

    def _log_every_learn_step_metrics_impl(self):
        log_file = os.path.join(self.log_dir, "training_every_learn_step_log.csv")
        row = {
            "episode": self.episode_counter,
            "step": self.ppo_agent.learn_step_counter,
            "timestep": self.timestep_counter,
            "actor_loss": self.actor_loss,
            "critic_loss": self.critic_loss,
            "entropy": self.entropy,
            "learn_step_reward": self.reward,
            "cumulative_reward": self.current_ep_reward,
        }
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.summary_writer.add_scalar(
            "Loss/actor", self.actor_loss, self.timestep_counter
        )
        self.summary_writer.add_scalar(
            "Loss/critic", self.critic_loss, self.timestep_counter
        )
        self.summary_writer.add_scalar("Entropy", self.entropy, self.timestep_counter)

        try:
            # Create metrics message
            metrics = {
                "episode": self.episode_counter,
                "step": self.timestep_counter,
                "timestep": self.timestep_counter * 64,  # Assuming batch size is 64
                "actor_loss": (
                    float(self.actor_loss) if self.actor_loss is not None else None
                ),
                "critic_loss": (
                    float(self.critic_loss) if self.critic_loss is not None else None
                ),
                "entropy": float(self.entropy) if self.entropy is not None else None,
                "step_reward": float(
                    self.current_ep_reward / max(1, self.current_step_in_episode)
                ),
                "cumulative_reward": float(self.current_ep_reward),
            }

        except Exception as e:
            self.get_logger().error(f"Failed to log step metrics: {e}")
            # self.log_error("log_every_learn_step_metrics", str(e))

    # def save_to_mongodb(self, schema):
    #     if self.db is not None:
    #         collection = self.db["episodes"]
    #         collection.insert_one(schema)
    #         self.get_logger().info(f"✅ Step metrics saved to MongoDB: {schema}")
    #     else:
    #         self.get_logger().error("❌ MongoDB connection is not initialized.")

    # def log_error(self, component, message):
    #     """Log error to MongoDB through DB service"""
    #     try:
    #         error_data = {"component": f"ppo_node.{component}", "message": message}

    #         msg = String()
    #         msg.data = json.dumps(error_data)
    #         self.error_logs_pub.publish(msg)
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to publish error log: {e}")

    def log_episode_metrics(self):
        log_file = os.path.join(
            self.log_dir,
            "training_episode_log.csv" if TRAIN else "testing_episode_log.csv",
        )
        row = {
            "episode": self.episode_counter,
            "timestep": self.timestep_counter,
            "episode_reward": self.current_ep_reward,
            "episode_length": self.current_step_in_episode,
            "termination_reason": self.termination_reason,
        }

        # self.save_to_mongodb(schema=row)

        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.summary_writer.add_scalar(
            f"Episode/Episode Duration (s)",
            (self.t2 - self.t1).total_seconds(),
            self.episode_counter,
        )
        self.summary_writer.add_scalar(
            f"Episode/Episode Length",
            self.current_step_in_episode,
            self.episode_counter,
        )
        self.summary_writer.add_scalar(
            f"Rewards/Episode Reward", self.current_ep_reward, self.episode_counter
        )

    def log_and_publish_episode_metrics(self):
        metrics = {
            "episode": self.episode_counter,
            "episode_reward": self.current_ep_reward,
            "actor_loss": float(self.last_actor_loss) if self.last_actor_loss else None,
            "critic_loss": (
                float(self.last_critic_loss) if self.last_critic_loss else None
            ),
            "entropy": float(self.last_entropy) if self.last_entropy else None,
            "duration": (self.t2 - self.t1).total_seconds(),
            "episode_length": self.current_step_in_episode,
            "train": TRAIN,
        }
        msg = String()
        msg.data = json.dumps(metrics)
        self.episode_metrics_pub.publish(msg)

    def log_and_publish_performance_metrics(self):
        performance = {
            "episode_id": f"ep_{self.episode_counter}",
            "lap_times": [],  # Add actual lap times if available
            "track_progress": self.track_progress,
            "collisions": 1 if self.collision else 0,
            "lap_progress": self.track_progress,
        }

        msg = String()
        msg.data = json.dumps(performance)
        self.performance_metrics_pub.publish(msg)

    def _log_system_metrics_impl(self):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        process = psutil.Process()
        process_mem_mb = process.memory_info().rss / 1024**2  # in MB

        self.summary_writer.add_scalar(
            "System/CPU_Usage (%)", cpu_percent, self.timestep_counter
        )
        self.summary_writer.add_scalar(
            "System/RAM_Usage (%)", ram_percent, self.timestep_counter
        )
        self.summary_writer.add_scalar(
            "System/Process_RAM (MB)", process_mem_mb, self.timestep_counter
        )
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2
            self.summary_writer.add_scalar(
                "System/GPU_Memory_Allocated (MB)",
                gpu_mem_allocated,
                self.timestep_counter,
            )
            self.summary_writer.add_scalar(
                "System/GPU_Memory_Reserved (MB)",
                gpu_mem_reserved,
                self.timestep_counter,
            )

    ##################################################################################################
    #                                           UTILITIES
    ##################################################################################################

    def handle_system_state(self, msg):
        """Handle state messages from the simulation coordinator"""
        state_msg = msg.data

        # Parse state
        if ":" in state_msg:
            state_name, details = state_msg.split(":", 1)
        else:
            state_name = state_msg
            details = ""

        self.current_sim_state = state_name
        # Reset model state on simulation reset events
        if state_name == "RUNNING" and "vehicle_" in details and "ready" in details:
            try:
                # Extract vehicle_id from "vehicle_{id}_ready"
                vehicle_id_str = details.split("vehicle_")[1].split("_ready")[0]
                self.vehicle_id = int(vehicle_id_str)

                # Reset tracking variables for new episode
                self.collision = False
                self.ready_to_collect = True
                self.track_progress = 0.0
                self.heading_deviation = 0.0
                self.lateral_deviation = 0.0
                self.vehicle_heading = 0.0
                self.heading_buffer = []  # Reset heading buffer
                self.prev_progress_distance = 0.0
                self.closest_waypoint_idx = 0
                self.start_point = None
                self.lap_completed = False
                self.lap_count = 0

                # Reset location tracking
                self.vehicle_location = None
                self.needs_progress_reset = True

                self.get_logger().info(
                    "PPO ready for new episode with vehicle ID: " + vehicle_id_str
                )
            except Exception as e:
                self.get_logger().error(f"Error handling vehicle ready state: {e}")

        # Handle state transitions
        elif state_name == "RESPAWNING":
            self.ready_to_collect = False
            self.get_logger().info(f"Pausing PPO during respawn: {details}")
            if "collision" in details.lower():
                # Only set collision flags if it's actually a collision
                self.collision = True
                self.done = True
                self.termination_reason = "collision"
                if (
                    self.prev_state is not None
                    and self.prev_action is not None
                    and self.prev_log_prob is not None
                ):
                    self.calculate_reward()
                    self.store_transition(
                        state=self.prev_state,
                        action=self.prev_action,
                        log_prob=self.prev_log_prob,
                        reward=self.reward,
                        done=True,
                    )
            elif "episode_complete" in details.lower():
                # Episode completed normally - don't apply collision penalty
                self.collision = False
                self.done = True
                self.get_logger().info(
                    "RESPAWNING due to episode completion: Paused data collection"
                )
            else:
                self.get_logger().info(
                    f"RESPAWNING for other reason: {details}: Paused data collection"
                )

        elif state_name == "MAP_SWAPPING":
            self.ready_to_collect = False
            self.collision = True
            self.done = True
            self.termination_reason = "collision"
            if (
                self.prev_state is not None
                and self.prev_action is not None
                and self.prev_log_prob is not None
            ):
                self.calculate_reward()
                self.store_transition(
                    state=self.prev_state,
                    action=self.prev_action,
                    log_prob=self.prev_log_prob,
                    reward=self.reward,
                    done=True,
                )
            self.get_logger().info(f"Pausing PPO during map swap: {details}")
            # Request fresh waypoints when the map changes
            self.needs_track_waypoints = True
            # Reset tracking variables for new map
            self.track_waypoints = []
            self.track_length = 0.0
            self.vehicle_location = None  # Reset location too
            self.get_logger().info(
                "Reset track data, will request new waypoints after map swap"
            )

    def check_waypoint_needs(self):
        """Check and process waypoint requests when needed"""
        if hasattr(self, "needs_track_waypoints") and self.needs_track_waypoints:
            self.get_logger().info(
                "Detected need for fresh track waypoints, requesting..."
            )
            self.needs_track_waypoints = False  # Reset flag
            self.request_track_waypoints()

    def request_track_waypoints(self):
        """Request track waypoints from the map loader service"""
        if (
            not hasattr(self, "waypoint_client")
            or not self.waypoint_client.service_is_ready()
        ):
            self.get_logger().info("Waypoint service not ready, will retry later")
            if hasattr(self, "retry_timer") and self.retry_timer:
                # Cancel any existing timer
                self.retry_timer.cancel()
            self.retry_timer = self.create_timer(1.0, self.retry_request_waypoints)
            return

        self.get_logger().info("Requesting track waypoints from map loader")
        request = GetTrackWaypoints.Request()
        future = self.waypoint_client.call_async(request)
        future.add_done_callback(self.handle_track_waypoints)

    def retry_request_waypoints(self):
        """Retry waypoint requests"""
        self.retry_timer.cancel()
        self.request_track_waypoints()

    def handle_track_waypoints(self, future):
        """Handle waypoints response"""
        try:
            response = future.result()

            if len(response.waypoints_x) == 0:
                self.retry_timer = self.create_timer(2.0, self.retry_request_waypoints)
                return

            # Process waypoints
            self.track_waypoints = []
            for i in range(len(response.waypoints_x)):
                self.track_waypoints.append(
                    (response.waypoints_x[i], response.waypoints_y[i])
                )

            self.track_length = response.track_length
            self.waypoint_count = len(self.track_waypoints)

            # Reset progress trackers for the new track
            self.reset_progress_tracking()
        except Exception as e:
            self.retry_timer = self.create_timer(2.0, self.retry_request_waypoints)

    def reset_progress_tracking(self):
        """Reset all progress-related variables"""
        if len(self.track_waypoints) == 0:
            self.get_logger().warn(
                "Can't reset progress tracking - no waypoints available"
            )
            return

        self.prev_waypoint_idx = 0
        self.current_waypoint_idx = 0
        self.closest_waypoint_idx = 0
        self.progress_buffer = []
        self.progress_timestamps = []
        self.prev_progress_value = 0.0
        self.total_progress = 0.0
        self.lap_progress = 0.0
        self.needs_progress_reset = False

        self.get_logger().info("Track progress variables fully reset for new track")

    def location_callback(self, msg):
        """Process vehicle location updates and calculate track progress"""
        if not self.ready_to_collect or self.current_sim_state in [
            "RESPAWNING",
            "MAP_SWAPPING",
        ]:
            return

        if len(self.track_waypoints) == 0:
            self.get_logger().warn(
                "Track waypoints not available yet. Cannot calculate progress."
            )
            return

        # Store previous location for direct movement calculation
        previous_location = self.vehicle_location

        # Extract vehicle location
        self.vehicle_location = (msg.data[0], msg.data[1], msg.data[2])
        self.vehicle_rotation = msg.data[3]


        # Experimetal

        now = time.time()
        if previous_location is not None and self.prev_t is not None:
            dx = self.vehicle_location[0] - previous_location[0]
            dy = self.vehicle_location[1] - previous_location[1]
            dt = max(1e-4, now - self.prev_t)
            # speed in m/s
            self.current_speed = math.hypot(dx, dy) / dt
        # update for next step
        self.prev_loc = self.vehicle_location
        self.prev_t = now

        # First call - just store the location
        if previous_location is None:
            self.get_logger().info(
                f"Initial vehicle position: ({self.vehicle_location[0]:.2f}, {self.vehicle_location[1]:.2f})"
            )
            self.update_track_progress()
            return

        # Calculate direct movement vector for debugging
        movement_x = self.vehicle_location[0] - previous_location[0]
        movement_y = self.vehicle_location[1] - previous_location[1]
        distance_moved = math.sqrt(movement_x**2 + movement_y**2)

        # Calculate heading from movement
        if distance_moved > 0.01:  # Only update if significant movement
            # Initialize heading buffer if not exists
            if not hasattr(self, "heading_buffer"):
                self.heading_buffer = []

            # Calculate heading from movement direction
            movement_heading = math.atan2(movement_y, movement_x)

            # Add to heading buffer for smoothing
            self.heading_buffer.append(movement_heading)
            if len(self.heading_buffer) > 5:  # Keep last 5 headings for smoothing
                self.heading_buffer.pop(0)

            # Calculate smoothed heading using circular mean
            sin_sum = sum(math.sin(h) for h in self.heading_buffer)
            cos_sum = sum(math.cos(h) for h in self.heading_buffer)
            self.vehicle_heading = math.atan2(sin_sum, cos_sum)

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
        are_adjacent = (abs(closest_idx - second_idx) == 1) or (
            abs(closest_idx - second_idx) == track_len - 1
        )

        if not are_adjacent:
            # The closest points aren't adjacent - find best segment by checking all point pairs
            min_segment_dist = float("inf")
            segment_start_idx = 0

            for i in range(track_len):
                next_i = (i + 1) % track_len
                p1 = self.track_waypoints[i]
                p2 = self.track_waypoints[next_i]

                # Calculate distance from point to line segment
                segment_dist = self.point_to_segment_distance(
                    self.vehicle_location, p1, p2
                )

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
            self.get_logger().info(
                f"Progress tracking initialized at waypoint {closest_idx}/{track_len}"
            )
            return

        # We need to know direction of track
        wp_current = self.track_waypoints[closest_idx]
        wp_next = self.track_waypoints[(closest_idx + 1) % track_len]

        # Project vehicle position onto the track segment to get precise location
        projection = self.project_point_to_segment(
            self.vehicle_location, wp_current, wp_next
        )

        # Calculate lateral deviation from centerline
        self.lateral_deviation = self.calculate_distance(
            self.vehicle_location, projection
        )
        

        track_vector = (wp_next[0] - wp_current[0], wp_next[1] - wp_current[1])
        track_angle = math.atan2(track_vector[1], track_vector[0])

        

        if hasattr(self, "vehicle_heading"):
            # Calculate angle difference (normalized to [-π, π])
            angle_diff = (
                (self.vehicle_heading - track_angle + math.pi) % (2 * math.pi)
            ) - math.pi
            self.heading_deviation = abs(angle_diff)
        else:
            self.get_logger().warn(
                "Vehicle heading not available for deviation calculation"
            )

        segment_progress = self.calculate_distance(
            wp_current, projection
        ) / self.calculate_distance(wp_current, wp_next)

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

        # Update progress
        self.track_progress = raw_progress

        # Log progress occasionally
        if (
            int(self.track_progress * 100) % 10 == 0
            and int(self.prev_progress_distance * 100) % 10 != 0
        ):
            self.get_logger().info(f"Track progress: {self.track_progress:.4f}")

    def point_to_segment_distance(self, p, v, w):
        """Calculate the distance from point p to line segment vw"""
        # Length squared of segment
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0:  # v == w case
            return self.calculate_distance(p, v)

        # Consider the line extending the segment, parameterized as v + t (w - v)
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment
        t = max(
            0,
            min(
                1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
            ),
        )

        # Projection falls on the segment
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))

        return self.calculate_distance(p, projection)

    def project_point_to_segment(self, p, v, w):
        """Project point p onto line segment vw and return the projection point"""
        # Length squared of segment
        l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
        if l2 == 0:  # v == w case
            return v

        # Consider the line extending the segment, parameterized as v + t (w - v)
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment
        t = max(
            0,
            min(
                1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
            ),
        )

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
                self.lap_time = (
                    self.lap_end_time - self.lap_start_time
                ).total_seconds()
                self.get_logger().info(
                    f"🏁 Lap {self.lap_count} completed in {self.lap_time:.2f} seconds!"
                )

                # Add lap time to tensorboard
                if self.summary_writer:
                    self.summary_writer.add_scalar(
                        "Laps/time", self.lap_time, self.lap_count
                    )

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
            self.get_logger().error(f"Service call failed: {e}")

    def calculate_distance(self, point1, point2):
        if point1 is None or point2 is None:
            return float("inf")
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

    def set_global_seed(self, seed=42):
        """
        Sets global seeds for reproducibility.
        """
        self.get_logger().info(f"🔒 [SEEDING] Setting global seed to {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.get_logger().info("✅ CUDA is available.")
        else:
            self.get_logger().info("⚠️ CUDA is not available. Running on CPU.")

        self.get_logger().info("Global seed setup complete.")

    def set_deterministic_cudnn(self, deterministic_cudnn=False):
        """
        Configures CuDNN for deterministic or non-deterministic behavior.
        """
        if (
            torch.cuda.is_available()
            and hasattr(torch.backends, "cudnn")
            and torch.backends.cudnn.enabled
        ):
            if deterministic_cudnn:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.get_logger().info("CuDNN deterministic mode enabled.")
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                self.get_logger().info(
                    "⚠️ CuDNN benchmarking mode enabled (faster but nondeterministic)."
                )
        else:
            self.get_logger().info("⚠️ CuDNN is not enabled or available.")

    def create_new_run_dir(self, load_from_run=None):

        version_dir = os.path.join(PPO_CHECKPOINT_DIR, VERSION)
        os.makedirs(version_dir, exist_ok=True)

        existing = [d for d in os.listdir(version_dir) if d.startswith("run_")]
        serial = len(existing) + 1

        timestamp = datetime.now().strftime("%Y%m%d")
        self.run_name = f"run_{timestamp}_{serial:04d}"
        self.run_dir = os.path.join(version_dir, self.run_name)

        # Create subfolders
        for sub in ["state_dict", "logs", "tensorboard"]:
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)

        self.state_dict_dir = os.path.join(self.run_dir, "state_dict")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tensorboard_dir = os.path.join(self.run_dir, "tensorboard")

        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.ppo_agent.summary_writer = self.summary_writer

        if load_from_run:
            self.get_logger().info(f"Loading weights from: {load_from_run}")
            self.ppo_agent.load_model_and_optimizers(
                os.path.join(load_from_run, "state_dict")
            )

    def shutdown_writer(self):
        self.summary_writer.close()
        self.get_logger().info("SummaryWriter closed.")

    def destroy_node(self):
        self.get_logger().info("Waiting for background tasks to complete...")
        try:
            self.task_queue.join()
            self.get_logger().info("All background tasks completed")
        except Exception as e:
            self.get_logger().error(f"Error waiting for background tasks: {e}")

        # if hasattr(self, "db"):
        #     mongo_connection.close_db()


def main(args=None):
    start_time = datetime.now()
    rclpy.init(args=args)
    node = PPOModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "Keyboard interrupt received, shutting down... (ppo node)"
        )
    finally:
        time.sleep(1)
        print("Saving training state in finally block")
        node.save_training_state(node.run_dir)
        node.destroy_node()
        node.shutdown_writer()
        end_time = datetime.now()
        node.get_logger().info(f"Total running time: {end_time - start_time}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
