import streamlit as st
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Get the ETAY_IP from environment variables
ETAY_IP = os.getenv("ETAY_IP", "localhost")

# Set the title and icon for the app
st.set_page_config(page_title="Canonicar", page_icon="🚗", layout="wide")
# Set page to wide mode to use full width

# Create a header with the project name
st.markdown("<h1 style='text-align: center;'>🚗 Canonicar</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Autonomous Driving Simulation Project</h3>", unsafe_allow_html=True)

# Add a nice divider
st.markdown("<hr>", unsafe_allow_html=True)

# Create a centered container for the main content
col1, main_col, col2 = st.columns([1, 2, 1])

with main_col:
    st.markdown("## Welcome to Canonicar")

    # Project description container with a light background and dark text
    with st.container():
        st.markdown("""
        <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;color:#333333;">
            <h4 style="color:#000000;">About the Project</h4>
            <p>Canonicar is an autonomous driving simulation project built on the CARLA simulator. 
            This dashboard provides real-time insights and controls for the simulation environment.</p>
            
        </div>
        """, unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Quick navigation links section
    st.markdown("### Quick Navigation")

    col_dash, col_fox = st.columns(2)

    with col_dash:
        st.markdown(
            """
            <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;text-align:center;color:#333333;">
                <h4 style="color:#000000;">Dashboard</h4>
                <p>View real-time simulation metrics and controls</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Use Streamlit's native button with navigation callback
        if st.button("Open Dashboard", key="dashboard_button",
                     use_container_width=True,
                     type="primary"):
            st.switch_page("pages/Dashboard.py")

    with col_fox:
        st.markdown(
            """
            <div style="background-color:#e1e9f7;padding:20px;border-radius:10px;text-align:center;color:#333333;">
                <h4 style="color:#000000;">Foxglove Visualization</h4>
                <p>View detailed 3D visualization of your simulation</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Use Streamlit's native button for external link
        if st.button("Open Foxglove", key="foxglove_button",
                     use_container_width=True,
                     type="primary"):
            foxglove_url = f'https://app.foxglove.dev/canonicar/view?ds=foxglove-websocket&ds.url=ws://{ETAY_IP}:8765&layoutId=lay_0dXyFqtjCo11IIVq'
            st.markdown(f'<a href="{foxglove_url}" target="_blank">Click here if the page did not open automatically</a>', unsafe_allow_html=True)
            # Use webbrowser to open URL
            import webbrowser
            webbrowser.open_new_tab(foxglove_url)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Project details expandable section
    with st.expander("Project Details"):
        st.write("""
        Add your detailed project description here. You can include:
        - Technical architecture
        - Features and capabilities
        - Development roadmap
        - Team information
        - References and resources
        """)

# Add footer with version info
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;color:#333333;'>Canonicar v1.0.0 | © 2025</p>", unsafe_allow_html=True)
