import streamlit as st
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
from threading import Thread
import struct
import plotly.graph_objects as go
from collections import deque

class CarlaViewerNode(Node):
    def __init__(self):
        super().__init__('carla_viewer_node')
        
        # Initialize data buffers
        self.latest_frame = None
        self.latest_imu = None
        self.latest_gnss = None
        self.latest_lidar = None
        
        # Keep history for plots
        self.imu_history = {
            'angular_velocity': deque(maxlen=100),
            'linear_acceleration': deque(maxlen=100)
        }
        self.gnss_history = deque(maxlen=100)
        
        # Create subscribers
        self.create_subscription(Image, '/carla/rgb_front/image_raw', self.image_callback, 10)
        self.create_subscription(Imu, '/carla/imu/imu', self.imu_callback, 10)
        self.create_subscription(NavSatFix, '/carla/gnss/gnss', self.gnss_callback, 10)
        self.create_subscription(PointCloud2, '/carla/lidar/points', self.lidar_callback, 10)
        
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def imu_callback(self, msg):
        self.latest_imu = msg
        self.imu_history['angular_velocity'].append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        self.imu_history['linear_acceleration'].append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def gnss_callback(self, msg):
        self.latest_gnss = msg
        self.gnss_history.append([msg.latitude, msg.longitude])

    def lidar_callback(self, msg):
        points = []
        for point in struct.iter_unpack('ffff', msg.data):
            points.append([point[0], point[1], point[2]])
        self.latest_lidar = np.array(points)

def start_ros_node():
    if not rclpy.ok():
        rclpy.init()
    node = CarlaViewerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Main Streamlit App
st.set_page_config(page_title="Live View", page_icon="📹", layout="wide")

st.title("🚗 Canonicar Live View")
st.markdown("Real-time visualization of CARLA simulation data")

# Initialize ROS node if not already initialized
if 'ros_thread' not in st.session_state:
    st.session_state.ros_thread = Thread(target=start_ros_node, daemon=True)
    st.session_state.ros_thread.start()
    st.session_state.node = CarlaViewerNode()

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