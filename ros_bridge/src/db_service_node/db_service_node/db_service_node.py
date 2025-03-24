import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from mongoengine import connect
import json
from datetime import datetime
import os
from db.schemas.episode_schema import Episode
from db.schemas.performance_schema import Performance
from db.schemas.error_log_schema import ErrorLog

class DBServiceNode(Node):
    def __init__(self):
        super().__init__('db_service_node')
        
        # Connect to MongoDB using environment variables
        try:
            mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
            connect("canonicar", host=mongo_url)
            self.get_logger().info('Connected to MongoDB')
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
            episode = Episode(
                episode_id=f"ep_{data['episode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                reward=data['episode_reward'],
                average_reward=data.get('average_reward'),
                policy_loss=data.get('actor_loss'),
                value_loss=data.get('critic_loss'),
                entropy=data.get('entropy'),
                duration=data.get('duration'),
                num_steps=data['episode_length'],
                stage='training' if data.get('train', True) else 'testing'
            )
            episode.save()
            self.get_logger().info(f'Saved episode metrics for episode {data["episode"]}')
        except Exception as e:
            self.get_logger().error(f'Failed to save episode metrics: {e}')

    def performance_callback(self, msg):
        try:
            data = json.loads(msg.data)
            performance = Performance(
                performance_id=f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                race_id=data['episode_id'],
                agent_type='ppo',
                lap_times=data.get('lap_times', []),
                total_time=data.get('total_time'),
                path_deviation=data.get('track_progress', 0),
                collisions=data.get('collisions', 0),
                lap_progress=data.get('lap_progress', 0)
            )
            performance.save()
            self.get_logger().info(f'Saved performance metrics for episode {data["episode_id"]}')
        except Exception as e:
            self.get_logger().error(f'Failed to save performance metrics: {e}')

    def error_callback(self, msg):
        try:
            data = json.loads(msg.data)
            error = ErrorLog(
                error_id=f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                component=data.get('component', 'unknown'),
                message=data.get('message'),
                resolution_status=False
            )
            error.save()
            self.get_logger().info(f'Saved error log: {data.get("message")}')
        except Exception as e:
            self.get_logger().error(f'Failed to save error log: {e}')