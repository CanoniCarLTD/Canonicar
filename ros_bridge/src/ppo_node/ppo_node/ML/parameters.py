"""
All the hyper-parameters needed for the PPO algorithm implementation.
"""

import numpy as np

MODEL_LOAD = True
# Directory to save model checkpoints - always stays the same
PPO_CHECKPOINT_DIR = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models"

# Set to None unless you're continuing an exact run (same version/run folder)
CHECKPOINT_FILE = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v4.0.2/run_20250825_0002"

VERSION = "v4.0.2"

# Point this to a full run directory from any version (must contain actor.pth etc.)
# Example: "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v2.1.3/run_20250325_0001"
LOAD_STATE_DICT_FROM_RUN = None


DETERMINISTIC_CUDNN = False

# Training configuration
TRAIN = True
EPISODE_LENGTH = 7000  # Maximum timesteps per episode
LEARN_EVERY_N_STEPS = 2048  # Number of timesteps collected before a policy update
MINIBATCH_SIZE = 256  # Each PPO update uses mini-batches of MINIBATCH_SIZE
NUM_EPOCHS = 5  # Each mini-batch is seen 5 times (full data 5×) in PPO update (Best practice: 3-10)
SAVE_EVERY_N_TIMESTEPS = LEARN_EVERY_N_STEPS * 2  # Save model every 2 policy updates

EPISODES = 1e8  # Not in use

# PPO-specific hyperparameters
PPO_INPUT_DIM = 100
TOTAL_TIMESTEPS = 2e8  # Total number of timesteps for training
USE_MONTE_CARLO = True  # try MC; set False to go back to GAE

# Exploration settings (action noise)
ACTION_STD_INIT = 0.2
ACTION_STD_DECAY_RATE = 0.0  # Not used as we are currently using learnable action std
MIN_ACTION_STD = 0.1

# PPO optimization parameters
ACTOR_LEARNING_RATE = 7.5e-5  # 0.000075
CRITIC_LEARNING_RATE = 7.5e-5  # 0.000075
POLICY_CLIP = 0.1  # CHANGED from 0.2 to 0.1
ENTROPY_COEF = 0.01  # might wanna do 0.005 later
LAMBDA_GAE = 0.95
VF_COEF = 0.5  # Giving half the weight to critic loss relative to the summed losses
GAMMA = 0.99  # Discount Factor for future rewards


# Evaluation settings
TEST_TIMESTEPS = 5e4


# Experimental
MIN_SPEED = 1.0  # m/s
TARGET_SPEED = 10.0  # m/s (choose your “cruise” speed)
MAX_SPEED = 30.0  # m/s (or whatever Carla max is)
MAX_DISTANCE_FROM_CENTER = 2.5  # meters
ANGLE_THRESH_RAD = np.deg2rad(45)  # 45 degrees
