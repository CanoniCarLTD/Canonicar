import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from pymongo import MongoClient
import json
from datetime import datetime
import os
from dotenv import load_dotenv

class DBService(Node):
    def __init__(self):
        super().__init__('db_service_node')
        
        # Load environment variables
        load_dotenv()
        
        # Connect to MongoDB using environment variables
        try:
            mongo_url = os.getenv("MONGO_CONNECTION_STRING")
            self.client = MongoClient(mongo_url)
            self.db = self.client.canonicar
            
            # Test connection
            self.client.admin.command('ping')
            self.get_logger().info('Connected to MongoDB Atlas')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to MongoDB: {e}')
            return

        # Create subscribers for PPO metrics
        self.create_subscription(
            String,
            '/training/episode_metrics',
            self.episode_callback,
            10
        )
        
        self.create_subscription(
            String,
            '/training/performance_metrics',
            self.performance_callback,
            10
        )
        
        self.create_subscription(
            String,
            '/training/error_logs',
            self.error_callback,
            10
        )

    def episode_callback(self, msg):
        try:
            data = json.loads(msg.data)
            episode_doc = {
                "episode_id": f"ep_{data['episode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now(),
                "reward": data['episode_reward'],
                "average_reward": data.get('average_reward'),
                "policy_loss": data.get('actor_loss'),
                "value_loss": data.get('critic_loss'),
                "entropy": data.get('entropy'),
                "duration": data.get('duration'),
                "num_steps": data['episode_length'],
                "stage": 'training' if data.get('train', True) else 'testing'
            }
            
            result = self.db.episodes.insert_one(episode_doc)
            self.get_logger().info(f'Saved episode metrics with ID: {result.inserted_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to save episode metrics: {e}')

    def performance_callback(self, msg):
        try:
            data = json.loads(msg.data)
            performance_doc = {
                "performance_id": f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now(),
                "race_id": data['episode_id'],
                "agent_type": 'ppo',
                "lap_times": data.get('lap_times', []),
                "total_time": data.get('total_time'),
                "path_deviation": data.get('track_progress', 0),
                "collisions": data.get('collisions', 0),
                "lap_progress": data.get('lap_progress', 0)
            }
            
            result = self.db.performance.insert_one(performance_doc)
            self.get_logger().info(f'Saved performance metrics with ID: {result.inserted_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to save performance metrics: {e}')

    def error_callback(self, msg):
        try:
            data = json.loads(msg.data)
            error_doc = {
                "error_id": f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now(),
                "component": data.get('component', 'unknown'),
                "message": data.get('message'),
                "resolution_status": False
            }
            
            result = self.db.errors.insert_one(error_doc)
            self.get_logger().info(f'Saved error log with ID: {result.inserted_id}')
        except Exception as e:
            self.get_logger().error(f'Failed to save error log: {e}')

    def destroy_node(self):
        if hasattr(self, 'client'):
            self.client.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DBService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()