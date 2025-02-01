import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='localhost',
        description='Host address for CARLA server'
    )

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='2000',
        description='Port number for CARLA server'
    )

    sensor_config_arg = DeclareLaunchArgument(
        'sensor_config',
        default_value=os.path.join(
            get_package_share_directory('client_node'),
            'client_node',
            'sensors_config.json'
        ),
        description='Path to the sensors_config.json file'
    )

    spawn_vehicle_node = Node(
        package='client_node',
        executable='spawn_vehicle_node',
        name='spawn_vehicle_node',
        output='screen',
        parameters=[
            {'host': LaunchConfiguration('host')},
            {'port': LaunchConfiguration('port')},
            {'sensor_config_file': LaunchConfiguration('sensor_config')}
        ]
    )

    return LaunchDescription([
        host_arg,
        port_arg,
        sensor_config_arg,
        spawn_vehicle_node
    ])
