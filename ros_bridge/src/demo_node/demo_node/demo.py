# filepath: /ros_bridge/src/demo_node/demo_node/demo_data.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import random

class DemoDataPublisher(Node):
    def __init__(self):
        super().__init__('demo_data_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'carla_demo_topic', 100)
        self.timer = self.create_timer(1.0, self.publish_data)

    def publish_data(self):
        msg = Float32MultiArray()
        # Simulate sensor data
        msg.data = [random.uniform(0, 100) for _ in range(100)]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = DemoDataPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()