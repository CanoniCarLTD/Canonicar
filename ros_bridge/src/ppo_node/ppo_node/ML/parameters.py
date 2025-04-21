"""
All the hyper-parameters needed for the PPO algorithm implementation.
"""

MODEL_LOAD = False

# Directory to save model checkpoints - always stays the same
PPO_CHECKPOINT_DIR = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models"

# Set to None unless you're continuing an exact run (same version/run folder)
CHECKPOINT_FILE = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v3.4.1/run_20250408_0002"

VERSION = "v3.4.2"

# Point this to a full run directory from any version (must contain actor.pth etc.)
# Example: "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v2.1.3/run_20250325_0001"
LOAD_STATE_DICT_FROM_RUN = None #"/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v3.2.0/run_20250330_0001"

DETERMINISTIC_CUDNN = False

# Training configuration
TRAIN = True
EPISODE_LENGTH = 3000  # Maximum timesteps per episode
LEARN_EVERY_N_STEPS = 2048 # Number of timesteps collected before a policy update
MINIBATCH_SIZE = 128  # Each PPO update uses mini-batches of MINIBATCH_SIZE
NUM_EPOCHS = 3  # Each mini-batch is seen 3 times (full data 3×) in PPO update (Best practice: 3-10)
SAVE_EVERY_N_TIMESTEPS = LEARN_EVERY_N_STEPS * 2 # Save model every 2 policy updates

EPISODES = 1e8 # Not in use

# PPO-specific hyperparameters
PPO_INPUT_DIM = 197
TOTAL_TIMESTEPS = 2e8  # Total number of timesteps for training

# Exploration settings (action noise)
ACTION_STD_INIT = 0.15
ACTION_STD_DECAY_RATE = 0.05 # Not used as we are currently using learnable action std
MIN_ACTION_STD = 0.05

# PPO optimization parameters
ACTOR_LEARNING_RATE = 2e-4
CRITIC_LEARNING_RATE = 3e-4
POLICY_CLIP = 0.2
ENTROPY_COEF = 0.02 # might wanna do 0.005 later
LAMBDA_GAE = 0.95
VF_COEF = 0.5  # Giving half the weight to critic loss relative to the summed losses
GAMMA = 0.99 # Discount Factor for future rewards

# Evaluation settings
TEST_TIMESTEPS = 5e4
