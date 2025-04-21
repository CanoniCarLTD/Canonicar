## Watching live training metrics progress

```bash
tensorboard --logdir=/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models --bind_all
```

If you want to monitor a specific version runs, simply add /vX.Y.Z - e.g:

```bash
tensorboard --logdir=/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models/v3.0.0 --bind_all
```

# Versions

## v1.0.0

### v1.1.0

### v1.1.1

## v2.0.0

- Fixing bugs, returned brake as an action we send, full review of PPO math implementation, removed logs to terminal.

### v2.1.0

- Switched from manually decaying action_std to a learnable log_std, fixed loading metadata.

### v2.1.1

- Added logging of Learned_Action_Std and mean_log_std, commented out KL early stop.
- Also changed save and load metadata to handle log_std.

### v2.1.2

- Modified deterministic settings to include all algorithms and random callings, Modified logging setting.
- An inhanced and imporved reward function is introduced.

### v2.1.3

v2.1.3 is a milestone - From here, We are hoping to do a long training run in order to see an improvement and some good metrics.

- Added a logic for loading a model state dict from a different version/file.

- Checked for correct loading:

```bash
root@----:/ros_bridge# sha256sum "path.../preTrained_PPO_models/v2.1.3/run_20250325_0001/state_dict/actor.pth"
a706b6a722cc69fab66c03bd77d9ed59128959332f5fb6477aa8c8751d5f2a4b  path.../preTrained_PPO_models/v2.1.3/run_20250325_0001/state_dict/actor.pth
root@----:/ros_bridge# sha256sum "path.../preTrained_PPO_models/v2.1.4/run_20250325_0001/state_dict/actor.pth"
a706b6a722cc69fab66c03bd77d9ed59128959332f5fb6477aa8c8751d5f2a4b  path.../preTrained_PPO_models/v2.1.4/run_20250325_0001/state_dict/actor.pth
```

## v3.0.0

- Removed GNSS sensor - Begin new training from scratch

- PPO_INPUT_DIM = 203 -> **198** 


### v3.1.0

- Modified map swapping, Fixed errors regarding to vehicle respawn and destroy sensors function.

### v3.1.1

Changed Hyperparameters:
- LEARN_EVERY_N_STEPS = 128 -> **512**
- EPISODE_LENGTH = 640 -> **1024**
- MINIBATCH_SIZE = 32 -> **64**
- EPISODES = 1e5 -> **1e8** (Negligible at the moment)

### v3.1.2

Changed Hyperparameters:
- LEARN_EVERY_N_STEPS = 512 -> **1024**
- EPISODE_LENGTH = 1024 -> **512**
- NUM_EPOCHS = 6 -> **4**
- ACTOR_LEARNING_RATE = 1e-4 -> **3e-4**
- CRITIC_LEARNING_RATE = 2.5e-4 -> **3e-4**

### v3.2.0

- Changed activation function to tanh (was relu)
- Changed weights initialization to xavier uniform (was kaiming normal)

### v3.2.1

- Removed unnecessary prints

### v3.3.0

- Implemented ground filtering
- Improved Vision Model components

### v3.3.1

- Fixed rewards normalization bug
- Integration of get_logger into ppo_agent
- Fixed store transition order to align all vars storage
- Commented cuda bebug flag (slows down the process when turned on)

Changed Hyperparameters:
- EPISODE_LENGTH = 512 -> **3000**

### v3.3.2

- Lowered upper bound for log_std: log(1.5) -> **log(0.4)**

Changed Hyperparameters:
- ACTOR_LEARNING_RATE = 3e-4 -> **2.5e-4**

### v3.3.3

- Fixed and improved pipeline's workflow.
- Added potential fix for Carla's crash.
- Canceled determinism
- Added gas/brake ratio penalty
- Gas/brake penalty bug fix
- Changed respawn approach into relocating

Changed Hyperparameters:
- DETERMINISTIC_CUDNN = True -> **False**

### v3.3.4

- Changed to raw values sample with reparameterized tanh-squashed actions

### v3.4.0

- Changed activations from tanh to leaky_relu
- Changed weights initialization to kaiming_uniform
- Removed IMU **FOR NOW** until sensor delay fix
- Therefore - **ACTION_DIM = 192**

Changed Hyperparameters:
- ACTOR_LEARNING_RATE = 2.5e-4 -> **2e-4**

### v3.4.1

Changed Hyperparameters:
- LEARN_EVERY_N_STEPS = 1024 -> **2048**
- MINIBATCH_SIZE = 64 -> **128**

### v3.4.2

- Brought back IMU
- Removed brake
- Fixed collision spawn bug
- Added vehicle deviation penalty
