import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Title
st.title("Dashboard")

st.set_page_config(page_title="Canonicar Dashboard", page_icon="🚗", layout="wide")

# Example 1: Reward Trend Over Episodes
episodes = np.arange(1, 101)
rewards = np.sin(episodes / 10) * 10 + episodes  # Simulated reward data
reward_fig = go.Figure()
reward_fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='lines', name='Reward'))
reward_fig.update_layout(title="Reward Trend Over Episodes", xaxis_title="Episodes", yaxis_title="Reward", height=300)

# Example 2: Action Distribution
actions = ["Accelerate", "Brake", "Steer Left", "Steer Right"]
action_counts = [50, 30, 70, 40]  # Simulated action counts
action_fig = go.Figure()
action_fig.add_trace(go.Bar(x=actions, y=action_counts, name='Actions', marker_color='skyblue'))
action_fig.update_layout(title="Action Distribution", xaxis_title="Actions", yaxis_title="Count", height=300)

# Example 3: PPO Loss Curve
steps = np.arange(1, 201)
loss = np.exp(-steps / 50) + np.random.normal(0, 0.02, size=len(steps))  # Simulated loss data
loss_fig = go.Figure()
loss_fig.add_trace(go.Scatter(x=steps, y=loss, mode='lines', name='Loss', line=dict(color='red')))
loss_fig.update_layout(title="PPO Loss Curve", xaxis_title="Training Steps", yaxis_title="Loss", height=300)

# Example 4: Cumulative Reward
cumulative_rewards = np.cumsum(rewards)
cumulative_reward_fig = go.Figure()
cumulative_reward_fig.add_trace(go.Scatter(x=episodes, y=cumulative_rewards, mode='lines', name='Cumulative Reward'))
cumulative_reward_fig.update_layout(title="Cumulative Reward Over Episodes", xaxis_title="Episodes", yaxis_title="Cumulative Reward", height=300)

# Example 5: Action Percentage
action_percentage = [count / sum(action_counts) * 100 for count in action_counts]
action_percentage_fig = go.Figure()
action_percentage_fig.add_trace(go.Pie(labels=actions, values=action_percentage, name='Action Percentage'))
action_percentage_fig.update_layout(title="Action Percentage Distribution", height=300)

# Example 6: Random Noise Data
random_data = np.random.normal(0, 1, 100)
random_data_fig = go.Figure()
random_data_fig.add_trace(go.Histogram(x=random_data, name='Random Data', marker_color='purple'))
random_data_fig.update_layout(title="Random Noise Data Distribution", xaxis_title="Value", yaxis_title="Frequency", height=300)

# Display the plots
# Create a 3-column layout for the charts
col1, col2, col3 = st.columns(3)

# First row
with col1:
    st.plotly_chart(reward_fig, use_container_width=True)
with col2:
    st.plotly_chart(action_fig, use_container_width=True)
with col3:
    st.plotly_chart(loss_fig, use_container_width=True)

# Second row
with col1:
    st.plotly_chart(cumulative_reward_fig, use_container_width=True)
with col2:
    st.plotly_chart(action_percentage_fig, use_container_width=True)
with col3:
    st.plotly_chart(random_data_fig, use_container_width=True)
