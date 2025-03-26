import asyncio
import websockets
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2
import cv2
import numpy as np
from cv_bridge import CvBridge

class CarlaDataServer(Node):
    def __init__(self, websocket_port=8766):
        super().__init__('carla_data_server')
        self.port = websocket_port
        self.connected_clients = set()
        self.bridge = CvBridge()
        
        # Create ROS subscribers
        self.create_subscription(Image, '/carla/rgb_front/image_raw', self.image_callback, 10)
        self.create_subscription(Imu, '/carla/imu/imu', self.imu_callback, 10)
        self.create_subscription(NavSatFix, '/carla/gnss/gnss', self.gnss_callback, 10)
        self.create_subscription(PointCloud2, '/carla/lidar/points', self.lidar_callback, 10)

    async def start_server(self):
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            self.get_logger().info(f"WebSocket server running on port {self.port}")
            await asyncio.Future()  # run forever

    async def handle_client(self, websocket):
        self.connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)

    async def broadcast(self, data):
        if self.connected_clients:
            message = json.dumps(data)
            await asyncio.gather(*[client.send(message) for client in self.connected_clients])

    def image_callback(self, msg):
        if not self.connected_clients:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, jpeg = cv2.imencode('.jpg', cv_image)
        data = {
            'type': 'image',
            'data': jpeg.tobytes().hex()
        }
        asyncio.create_task(self.broadcast(data))

    def imu_callback(self, msg):
        if not self.connected_clients:
            return
        data = {
            'type': 'imu',
            'data': {
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ],
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ]
            }
        }
        asyncio.create_task(self.broadcast(data))