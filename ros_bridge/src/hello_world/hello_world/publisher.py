import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloWorldPublisher(Node):
    def __init__(self):
        super().__init__('hello_world_publisher')
        self.publisher = self.create_publisher(String, 'hello_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_message)
        self.get_logger().info("HelloWorldPublisher is running...")

    def publish_message(self):
        msg = String()
        msg.data = "Hello, World!"
        self.publisher.publish(msg)
        self.get_logger().info(f"Published: '{msg.data}'")

def main(args=None):
    rclpy.init(args=args)
    node = HelloWorldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
