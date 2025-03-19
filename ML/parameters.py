"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""
# Checkpoint saving directory settings
PPO_CHECKPOINT_DIR = "preTrained_PPO_models/"
VERSION = "1.0.0"

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
# IM_WIDTH = 160
# IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 5000
EPISODES = 1000

TO_PPO_DIM = 95  # CHANGE WITH ACCORDANCE TO OUR MODEL (ETAI)

# Proximal Policy Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4
POLICY_CLIP = 0.2
OBS_DIM = 100
