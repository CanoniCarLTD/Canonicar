import streamlit as st
import websocket
import json
import numpy as np
import plotly.graph_objects as go
from collections import deque
import cv2

class CarlaViewer:
    def __init__(self):
        self.imu_history = {
            'angular_velocity': deque(maxlen=100),
            'linear_acceleration': deque(maxlen=100)
        }
        self.gnss_history = deque(maxlen=100)
        self.latest_frame = None
        self.latest_lidar = None
        
    def connect_websocket(self):
        self.ws = websocket.WebSocketApp(
            "ws://localhost:8766",
            on_message=self.on_message,
            on_error=lambda ws, err: st.error(f"WebSocket error: {err}"),
            on_close=lambda ws: st.warning("Connection closed")
        )
        return self.ws.run_forever

    def on_message(self, ws, message):
        data = json.loads(message)
        if data['type'] == 'image':
            # Convert hex string back to image
            img_array = np.frombuffer(bytes.fromhex(data['data']), dtype=np.uint8)
            self.latest_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif data['type'] == 'imu':
            self.imu_history['angular_velocity'].append(data['data']['angular_velocity'])
            self.imu_history['linear_acceleration'].append(data['data']['linear_acceleration'])

# Main Streamlit App
st.set_page_config(page_title="Live View", page_icon="📹", layout="wide")
st.title("🚗 Canonicar Live View")

viewer = CarlaViewer()

# Start WebSocket connection in a thread
if 'ws_thread' not in st.session_state:
    from threading import Thread
    st.session_state.ws_thread = Thread(target=viewer.connect_websocket, daemon=True)
    st.session_state.ws_thread.start()

# Create layout
col1, col2 = st.columns([2, 1])

with col1:
    # Camera view
    st.subheader("📸 Camera Feed")
    camera_placeholder = st.empty()
    
    # LiDAR view
    st.subheader("🔍 LiDAR Point Cloud")
    lidar_placeholder = st.empty()

with col2:
    # IMU data
    st.subheader("📊 IMU Data")
    imu_placeholder = st.empty()
    
    # GNSS data
    st.subheader("🌍 GNSS Path")
    gnss_placeholder = st.empty()

# Update loop
while True:
    node = st.session_state.node
    
    # Update camera feed
    if node.latest_frame is not None:
        camera_placeholder.image(
            node.latest_frame, 
            channels="BGR",
            use_column_width=True
        )
    
    # Update LiDAR visualization
    if node.latest_lidar is not None and len(node.latest_lidar) > 0:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=node.latest_lidar[:, 0],
                y=node.latest_lidar[:, 1],
                z=node.latest_lidar[:, 2],
                mode='markers',
                marker=dict(size=1)
            )
        ])
        fig.update_layout(height=400)
        lidar_placeholder.plotly_chart(fig, use_container_width=True)
    
    # Update IMU plots
    if node.latest_imu is not None:
        imu_fig = go.Figure()
        
        for i, data_type in enumerate(['angular_velocity', 'linear_acceleration']):
            data = np.array(list(node.imu_history[data_type]))
            for j, axis in enumerate(['x', 'y', 'z']):
                imu_fig.add_trace(go.Scatter(
                    y=data[:, j],
                    name=f"{data_type}_{axis}"
                ))
        
        imu_fig.update_layout(height=300)
        imu_placeholder.plotly_chart(imu_fig, use_container_width=True)
    
    # Update GNSS plot
    if len(node.gnss_history) > 0:
        gnss_data = np.array(list(node.gnss_history))
        gnss_fig = go.Figure()
        gnss_fig.add_trace(go.Scatter(
            x=gnss_data[:, 1],
            y=gnss_data[:, 0],
            mode='lines+markers',
            name='Vehicle Path'
        ))
        gnss_fig.update_layout(height=300)
        gnss_placeholder.plotly_chart(gnss_fig, use_container_width=True)
    
    st.experimental_rerun()