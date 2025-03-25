## **Understanding Episodes and Timesteps in Reinforcement Learning**

### **Timestep**
- **Definition:** A single interaction where the agent **observes** the environment, **takes an action**, and **receives feedback** (reward and new state).  
- **Analogy:** Think of a timestep as **one move in a chess game** or **one frame in a video**.  
- **Example:** In a **self-driving car simulation**, a timestep represents the car **reading sensor data, deciding to accelerate or brake, and observing the outcome**.  

### **Episode**
- **Definition:** A sequence of **timesteps** that starts from an **initial state** and continues until a **termination condition** is met (e.g., reaching a goal or running out of time).  
- **Analogy:** Consider an episode as **a full lap in a race** or **a complete attempt at a video game level**.  
- **Example:** In a **self-driving car environment**, an episode might begin with the car at the **starting line** and end when it **completes a lap or crashes**.  

### **Relationship Between Episodes and Timesteps**
- **Composition:** Each **episode** consists of **multiple timesteps**.  
- **Structure:** The agent **makes decisions** at every timestep, and the cumulative experience **forms an episode**, which helps the agent learn from long-term consequences.  

---

## **Clarifying `EPISODE_LENGTH` and `TOTAL_TIMESTEPS`**
In our `parameters.py`, we define two key values:

### **`EPISODE_LENGTH = 7500`**
- **Definition:** The **maximum number of timesteps allowed per episode**.  
- **Analogy:** If an episode is **a single lap in a race**, `EPISODE_LENGTH` is the **maximum number of time intervals** before the lap is forced to end.  
- **Example:** If the car reaches 7,500 timesteps but **hasn’t finished the lap**, the episode **ends automatically**.

### **`TOTAL_TIMESTEPS = 2e6`**
- **Definition:** The **total number of timesteps used for training**, across all episodes.  
- **Analogy:** If `EPISODE_LENGTH` is the number of **steps per lap**, `TOTAL_TIMESTEPS` is the **total number of steps across all training laps**.  
- **Example:** If each episode is **7500 timesteps**, then `TOTAL_TIMESTEPS = 2,000,000` means the agent will experience **~267 full episodes** during training (`2,000,000 ÷ 7,500 ≈ 267`).  

---

## Watching live training metrics progress (STILL NEEDS TESTING)

```bash
tensorboard --logdir=/ros_bridge/src/ppo_node/ppo_node/ML/preTrained_PPO_models --bind_all
```

# Versions

## v1.0.0

### v1.1.0

### v1.1.1

## v2.0.0

Fixing bugs, returned brake as an action we send, full review of PPO math implementation, removed logs to terminal.

### v2.1.0

Switched from manually decaying action_std to a learnable log_std, fixed loading metadata

### v2.1.1

Added logging of Learned_Action_Std and mean_log_std, commented out KL early stop.
Also changed save and load metadata to handle log_std