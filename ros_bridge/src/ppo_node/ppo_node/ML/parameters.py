"""
All the hyper-parameters needed for the PPO algorithm implementation.
"""

# Set to True to load a previous run
MODEL_LOAD = False 

# Directory to save model checkpoints - always stays the same
PPO_CHECKPOINT_DIR = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models"

# Set to None unless you're continuing an exact run (same version/run folder)
CHECKPOINT_FILE = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v3.1.2/run_20250328_0003"

VERSION = "v3.1.2"

# Point this to a full run directory from any version (must contain actor.pth etc.)
# Example: "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v2.1.3/run_20250325_0001"
LOAD_STATE_DICT_FROM_RUN = None #"/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v3.1.1/run_20250327_0001"

TRAIN = True  # Set to False to disable training and only run inference
DETERMINISTIC_CUDNN = True

# Training configuration
SEED = 42
EPISODE_LENGTH = 512  # Maximum timesteps per episode
LEARN_EVERY_N_STEPS = 1024 # Number of timesteps collected before a policy update
MINIBATCH_SIZE = 64  # Each PPO update uses mini-batches of MINIBATCH_SIZE
NUM_EPOCHS = 4  # Each mini-batch is seen 4 times (full data 4×) in PPO update (Best practice: 3-10)
EPISODES = 1e8
SAVE_EVERY_N_TIMESTEPS = LEARN_EVERY_N_STEPS * 2 # Save model every 2 policy updates

# PPO-specific hyperparameters
PPO_INPUT_DIM = 198 
TOTAL_TIMESTEPS = 2e8  # Total number of timesteps for training

# Discount Factor
GAMMA = 0.99  # Discount factor for future rewards

# Exploration settings (action noise)
ACTION_STD_INIT = 0.2
ACTION_STD_DECAY_RATE = 0.05 # Not used as we are currently use learnable action std
MIN_ACTION_STD = 0.05
MAX_KL = 0.2 # Maximum KL divergence between new and old policy

# PPO optimization parameters
ACTOR_LEARNING_RATE = 3e-4 # PPO is stable enough in lr of 2.5e-4 - 3e-4
CRITIC_LEARNING_RATE = 3e-4 # PPO is stable enough in lr of 2.5e-4 - 3e-4
POLICY_CLIP = 0.2
ENTROPY_COEF = 0.01 # might wanna do 0.005 later
LAMBDA_GAE = 0.95  # Generalized Advantage Estimation lambda
VF_COEF = 0.5  # Giving half the weight to critic loss relative to the summed losses

# Evaluation settings
TEST_TIMESTEPS = 5e4
