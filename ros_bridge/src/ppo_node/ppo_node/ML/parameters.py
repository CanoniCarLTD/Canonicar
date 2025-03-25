"""
All the hyper-parameters needed for the PPO algorithm implementation.
"""

# Directory and version control for checkpoint saving and loading
PPO_CHECKPOINT_DIR = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models"
VERSION = "v2.1.2"

MODEL_LOAD = True  # Set to True to resume training from checkpoint
CHECKPOINT_FILE = "/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v2.1.2/run_20250325_0001" # Specify file when MODEL_LOAD=True, else set to NONE

TRAIN = True  # Set to False to disable training and only run inference
DETERMINISTIC_CUDNN = True

# Training configuration
SEED = 42
LEARN_EVERY_N_STEPS = 128 # Number of timesteps collected before a policy update
MINIBATCH_SIZE = 32  # Number of samples in each minibatch
NUM_EPOCHS = 6  # PPO update epochs per batch (Best practice: 3-10)
EPISODES = 100000
SAVE_EVERY_N_TIMESTEPS = LEARN_EVERY_N_STEPS * 10

# PPO-specific hyperparameters
PPO_INPUT_DIM = 203
EPISODE_LENGTH = 640  # Maximum timesteps per episode
TOTAL_TIMESTEPS = 2e8  # Total number of timesteps for training

# Discount Factor
GAMMA = 0.99  # Discount factor for future rewards

# Exploration settings (action noise)
ACTION_STD_INIT = 0.2
ACTION_STD_DECAY_RATE = 0.05
MIN_ACTION_STD = 0.05
MAX_KL = 0.2 # Maximum KL divergence between new and old policy

# PPO optimization parameters
ACTOR_LEARNING_RATE = 1e-4 # We might want to change to 2.5e-4 as PPO is stable enough in lr of 2.5e-4 - 3e-4
CRITIC_LEARNING_RATE = 2.5e-4 # We might want to change to 2.5e-4 as PPO is stable enough in lr of 2.5e-4 - 3e-4
POLICY_CLIP = 0.2
ENTROPY_COEF = 0.01
LAMBDA_GAE = 0.95  # Generalized Advantage Estimation lambda
VF_COEF = 0.5  # Weight for value function loss in PPO

# Evaluation settings
TEST_TIMESTEPS = 5e4
