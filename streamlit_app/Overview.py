import streamlit as st
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set the title and icon for the app
st.set_page_config(page_title="Canonicar", page_icon="🚗", layout="wide")

# Custom CSS for modern styling with updated colors and fonts
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Raleway:wght@400;500;600;700&display=swap');
        
        body {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Raleway', sans-serif;
        }
        .main-header {
            background: linear-gradient(135deg, #e63946, #ff8c94);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        .content-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            font-family: 'Raleway', sans-serif;
            color: #212529 !important;
        }
        /* Dark mode overrides to ensure text visibility */
        [data-theme="dark"] .content-section,
        [data-theme="dark"] .project-card,
        [data-theme="dark"] .key-feature {
            color: #212529 !important;
        }
        [data-theme="dark"] .content-section h3,
        [data-theme="dark"] .content-section p,
        [data-theme="dark"] .content-section ul,
        [data-theme="dark"] .content-section li,
        [data-theme="dark"] .project-card p,
        [data-theme="dark"] .project-card ul,
        [data-theme="dark"] .project-card li,
        [data-theme="dark"] .key-feature p,
        [data-theme="dark"] h4 {
            color: #212529 !important;
        }
        [data-theme="dark"] strong {
            color: #111111 !important;
        }
        .video-container {
            position: relative;
            padding-bottom: 56.25%;
            height: 0;
            overflow: hidden;
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin: 2rem 0;
        }
        .video-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }
        .footer {
            text-align: center;
            padding: 1.5rem;
            background: #1a1a1a;
            color: white;
            border-radius: 10px;
            margin-top: 2rem;
            font-family: 'Montserrat', sans-serif;
        }
        .project-card {
            background: #f5f5f5;
            border-left: 4px solid #d62828;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-radius: 5px;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
        }
        h3 {
            color: #212529;
            margin-bottom: 1.2rem;
        }
        h4 {
            color: #d62828;
            margin-bottom: 1rem;
        }
        p, ul, li {
            font-family: 'Raleway', sans-serif;
            line-height: 1.6;
        }
        .key-feature {
            background-color: #f8f8f8;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #d62828;
        }
        .stButton button {
            background-color: #d62828;
            color: white;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #b51d1d;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Create a modern header with the project name
st.markdown("""
    <div class="main-header">
        <h1 style='text-align: center;'>🚗 Canonicar</h1>
        <h3 style='text-align: center;'>Data-Driven Autonomous Racing Simulation</h3>
    </div>
""", unsafe_allow_html=True)

# Main Content Section
col1, main_col, col2 = st.columns([1, 3, 1])

with main_col:
    # Project Summary
    with st.container():
        st.markdown("""
        <div class="content-section">
            <h3>Project Overview</h3>
            <p>
                Canonicar is an AI-driven autonomous racing simulation built on the CARLA simulator. 
                Using reinforcement learning (PPO), our model learns to optimize racing lines for efficient lap times. 
                This platform provides real-time insights, performance analytics, and interactive simulation controls.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # YouTube Video
    st.markdown("""
        <div class="content-section">
            <h3>Canonicar in Action</h3>
            <div class="video-container">
                <iframe src="https://www.youtube.com/embed/Dq6RqXCa_io" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # TensorBoard Navigation
    st.markdown("""
        <div class="content-section">
            <h3>Data Analysis</h3>
            <p>Analyze our model's performance metrics and learning progress through the interactive TensorBoard dashboard.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Open TensorBoard", key="tensorboard_button", use_container_width=True, type="primary"):
        st.switch_page("pages/Tensor_board.py")

    # Project Details Section - Now directly displayed (not in dropdown)
    st.markdown("""
        <div class="content-section">
            <h3>Project Details</h3>
            <div class="project-card">
                <h4>Technologies Used</h4>
                <ul>
                    <li><strong>CARLA Simulator</strong>: High-fidelity urban driving and racing environment</li>
                    <li><strong>Reinforcement Learning (PPO)</strong>: Policy optimization for strategic race navigation</li>
                    <li><strong>Sensor Fusion</strong>: Utilizing camera, LiDAR, and IMU for real-time decision-making</li>
                    <li><strong>Performance Metrics</strong>: Evaluating model efficiency through lap time analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Key Features - Using more explicit dark mode compatible styling
    st.markdown("""
        <style>
            /* Additional dark mode overrides for key features and research goals */
            .dark-text {
                color: #111111 !important;
            }
            .dark-text p, .dark-text li, .dark-text ul, .dark-text strong {
                color: #111111 !important;
            }
            .dark-text h4 {
                color: #d62828 !important;
            }
            .key-feature-dark {
                background-color: #f8f8f8 !important;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border-left: 4px solid #d62828;
            }
            .project-card-dark {
                background: #f5f5f5 !important;
                border-left: 4px solid #d62828;
                padding: 1.5rem;
                margin-bottom: 1rem;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""<h4 class="dark-text" style="font-family: Montserrat, sans-serif; font-weight: 600;">Key Features</h4>""", unsafe_allow_html=True)

    features = [
        ("Real-time simulation and control", "🏎️"),
        ("Adaptive reinforcement learning model", "🧠"),
        ("Interactive data visualization", "📊"),
        ("Customizable race track configurations", "🛣️")
    ]

    for feature, emoji in features:
        st.markdown(f"""
        <div class="key-feature-dark">
            <p class="dark-text" style="margin-bottom: 0;">{emoji} <strong>{feature}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="project-card-dark">
            <h4 class="dark-text">Research Goals</h4>
            <ul class="dark-text">
                <li>Enhance generalization across different tracks</li>
                <li>Optimize decision-making using multi-sensor data</li>
                <li>Incorporate real-world racing strategies into AI-driven decision-making</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Canonicar v1.0.0 | © 2025</p>
        <p>Developed by Canonicar Team</p>
    </div>
""", unsafe_allow_html=True)
