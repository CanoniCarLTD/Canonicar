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

Fixing bugs, returned brake as an action we send, full review of PPO math implementation, removed logs to terminal.

### v2.1.0

Switched from manually decaying action_std to a learnable log_std, fixed loading metadata.

### v2.1.1

Added logging of Learned_Action_Std and mean_log_std, commented out KL early stop.
Also changed save and load metadata to handle log_std.

### v2.1.2

Modified deterministic settings to include all algorithms and random callings, Modified logging setting.
An inhanced and imporved reward function is introduced.

### v2.1.3

v2.1.3 is a milestone - From here, We are hoping to do a long training run in order to see an improvement and some good metrics.

Added a logic for loading a model state dict from a different version/file.

-Checked for correct loading:

```bash
root@----:/ros_bridge# sha256sum "path.../preTrained_PPO_models/v2.1.3/run_20250325_0001/state_dict/actor.pth"
a706b6a722cc69fab66c03bd77d9ed59128959332f5fb6477aa8c8751d5f2a4b  path.../preTrained_PPO_models/v2.1.3/run_20250325_0001/state_dict/actor.pth
root@----:/ros_bridge# sha256sum "path.../preTrained_PPO_models/v2.1.4/run_20250325_0001/state_dict/actor.pth"
a706b6a722cc69fab66c03bd77d9ed59128959332f5fb6477aa8c8751d5f2a4b  path.../preTrained_PPO_models/v2.1.4/run_20250325_0001/state_dict/actor.pth
```

## v3.0.0

Removed GNSS sensor - Begin new training from scratch