"""
All the hyper-parameters needed for the PPO algorithm implementation.
"""

import numpy as np

MODEL_LOAD = True
# Directory to save model checkpoints - always stays the same
PPO_CHECKPOINT_DIR = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models"

# Set to None unless you're continuing an exact run (same version/run folder)
CHECKPOINT_FILE = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v4.0.2/run_20250827_0008"

VERSION = "v4.0.2"

# Point this to a full run directory from any version (must contain actor.pth etc.)
# Example: "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v2.1.3/run_20250325_0001"
LOAD_STATE_DICT_FROM_RUN = None

DETERMINISTIC_CUDNN = True

TRAIN = True
EPISODE_LENGTH = 7000  # Maximum timesteps per episode
LEARN_EVERY_N_STEPS = 2048  # Number of timesteps collected before a policy update
NUM_EPOCHS = 7  # PPO best practice is 3-10
SAVE_EVERY_N_TIMESTEPS = LEARN_EVERY_N_STEPS * 2  # Save model every 2 policy updates
ACTION_STD_INIT = 0.05
PPO_INPUT_DIM = 100
TOTAL_TIMESTEPS = 2e8  # Total number of timesteps for training
EPISODES = 1e8  # Not in use

# PPO optimization parameters
PPO_LEARNING_RATE = 1e-4
POLICY_CLIP = 0.2
ENTROPY_COEF = 0.01
GAMMA = 0.99  # Discount Factor for future rewards
VF_COEF = 0.5  # Value function coefficient for the loss calculation

# Evaluation settings
TEST_TIMESTEPS = 5e4
MINIBATCH_SIZE = 256
# Experimental
MIN_SPEED = 1.0  # m/s
TARGET_SPEED = 10.0  # m/s (choose your “cruise” speed)
MAX_SPEED = 30.0  # m/s (or whatever Carla max is)
MAX_DISTANCE_FROM_CENTER = 2.5  # meters
ANGLE_THRESH_RAD = np.deg2rad(45)  # 45 degrees
