import asyncio
import websockets
import json

ROS2_WS_URL = "ws://localhost:9090"  # WebSocket connection


async def subscribe_to_ros_topic():
    async with websockets.connect(ROS2_WS_URL) as websocket:
        print("Connected to ROS2 WebSocket!")

        subscribe_msg = {
            "op": "subscribe",
            "topic": "/clicked_point",  # Replace with the topic you want
            "type": "geometry_msgs/msg/PointStamped"  # Update with the correct message type
        }
        await websocket.send(json.dumps(subscribe_msg))

        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received message:\n{json.dumps(data, indent=2)}")

asyncio.run(subscribe_to_ros_topic())
