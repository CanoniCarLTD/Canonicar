"""

    All the hyper-parameters needed for the algorithm implementation. 

"""

# Directory and version control for checkpoint saving and loading
PPO_CHECKPOINT_DIR = "preTrained_PPO_models/"
VERSION = "1.0.0"

MODEL_LOAD = False  # Set to True to resume training from checkpoint
CHECKPOINT_FILE = None  # Specify file when MODEL_LOAD=True

SEED = 0
BATCH_SIZE = 64 # The number of timesteps collected before the model performs a single policy update.
GAMMA = 0.99
EPISODES = 1000

PPO_INPUT_DIM = None # Can be dynamic using data colletor node

# PPO hyper parameters

EPISODE_LENGTH = 7500 # Maximum number of timesteps the agent can take within a single episode.
TOTAL_TIMESTEPS = 2e6 # Total number of timesteps allocated for training across ALL episodes

ACTION_STD_INIT = 0.2
ACTION_STD_DECAY_RATE = 0.05
MIN_ACTION_STD = 0.05

TEST_TIMESTEPS = 5e4
LEARNING_RATE = 1e-4
POLICY_CLIP = 0.2
ENTROPY_COEF = 0.01
LAMBDA_GAE = 0.95
