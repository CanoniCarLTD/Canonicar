import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloWorldSubscriber(Node):
    def __init__(self):
        super().__init__('hello_world_subscriber')
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10
        )
        self.get_logger().info("HelloWorldSubscriber is running...")

    def listener_callback(self, msg):
        self.get_logger().info(f"Received: '{msg.data}'")

def main(args=None):
    rclpy.init(args=args)
    node = HelloWorldSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
