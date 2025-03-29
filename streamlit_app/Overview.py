import streamlit as st
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set the title and icon for the app
st.set_page_config(page_title="Canonicar", page_icon="🚗", layout="wide")

# Create a header with the project name
st.markdown("""
    <h1 style='text-align: center;'>🚗 Canonicar</h1>
    <h3 style='text-align: center;'>Data-Driven Autonomous Racing Simulation</h3>
""", unsafe_allow_html=True)

# Add a divider
st.markdown("<hr>", unsafe_allow_html=True)

# Main Content Section
col1, main_col, col2 = st.columns([1, 2, 1])

with main_col:
    st.markdown("## Welcome to Canonicar")

    # Project Summary
    with st.container():
        st.markdown("""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;color:#333333;">
            <h4 style="color:#000000;">Project Overview</h4>
            <p>
                Canonicar is an AI-driven autonomous racing simulation built on the CARLA simulator. 
                Using reinforcement learning (PPO), our model learns to optimize racing lines for efficient lap times. 
                This dashboard provides real-time insights, performance analytics, and interactive simulation controls.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Navigation
    st.markdown("### Quick Navigation")
    col_dash, col_sim, col_tensor = st.columns(3)

    with col_dash:
        st.markdown("""
        <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;text-align:center;color:#333333;">
            <h4 style="color:#000000;">Dashboard</h4>
            <p>View real-time simulation metrics and model performance</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dashboard", key="dashboard_button", use_container_width=True, type="primary"):
            st.switch_page("pages/Dashboard.py")

    with col_sim:
        st.markdown("""
        <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;text-align:center;color:#333333;">
            <h4 style="color:#000000;">Simulation</h4>
            <p>Run and control the CARLA racing simulation</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Simulation", key="simulation_button", use_container_width=True, type="primary"):
            st.switch_page("pages/Simulation.py")

    with col_tensor:
        st.markdown("""
        <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;text-align:center;color:#333333;">
            <h4 style="color:#000000;">TensorBoard</h4>
            <p>Analyze training performance and learning curves</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open TensorBoard", key="tensorboard_button", use_container_width=True, type="primary"):
            st.switch_page("pages/TensorBoard.py")

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Project Details Section
    with st.expander("Project Details"):
        st.write("""
        Canonicar is a research-driven project that aims to push the boundaries of autonomous racing. 
        Our approach involves:
        
        ### Technologies Used:
        - **CARLA Simulator**: High-fidelity urban driving and racing environment
        - **Reinforcement Learning (PPO)**: Policy optimization for strategic race navigation
        - **Sensor Fusion**: Utilizing camera, LiDAR, and IMU for real-time decision-making
        - **Performance Metrics**: Evaluating model efficiency through lap time analysis
        
        ### Key Features:
        - Real-time simulation and control
        - Adaptive reinforcement learning model
        - Interactive data visualization
        - Customizable race track configurations
        
        ### Research Goals:
        - Enhance generalization across different tracks
        - Optimize decision-making using multi-sensor data
        
        We continuously refine our model, incorporating real-world racing strategies into AI-driven decision-making.
        """)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;color:#333333;'>Canonicar v1.0.0 | © 2025</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;color:#333333;'>Developed by Canonicar Team</p>", unsafe_allow_html=True)
