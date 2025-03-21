# ros_bridge/src/client_node/client_node/run_load_map.py

import rclpy
from rclpy.node import Node
import carla
from time import sleep
import open3d as o3d
import os
import xml.etree.ElementTree as ET
from std_msgs.msg import String


class LoadMapNode(Node):
    def __init__(self):
        super().__init__("load_map_node")
        self.get_logger().info("LoadMapNode started, managing the map...")

        # Declare ROS2 parameters with default values
        self.declare_parameter("host", "")
        self.declare_parameter("TRACK_LINE", "")
        self.declare_parameter("TRACK_XODR", "")
        self.declare_parameter("CARLA_SERVER_PORT", 2000)

        # Retrieve parameter values
        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.TRACK_LINE = (
            self.get_parameter("TRACK_LINE").get_parameter_value().string_value
        )
        self.TRACK_XODR = (
            self.get_parameter("TRACK_XODR").get_parameter_value().string_value
        )
        self.CARLA_SERVER_PORT = (
            self.get_parameter("CARLA_SERVER_PORT").get_parameter_value().integer_value
        )

        # Start to deploy vehicles after map is loaded
        self.start_vehicle_manager = self.create_publisher(
            String, "/start_vehicle_manager", 10
        )

        # Validate parameters
        if not all([self.host, self.TRACK_LINE, self.TRACK_XODR]):
            self.get_logger().error("One or more required parameters are not set.")
            self.destroy_node()
            return

        # Initialize CARLA client and setup map
        try:
            self.client = carla.Client(self.host, self.CARLA_SERVER_PORT)
            self.client.set_timeout(10.0)
            self.get_logger().info(
                f"Connected to CARLA: {self.client.get_server_version()}"
            )
            self.setup_map()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CARLA client: {e}")
            self.destroy_node()

    def setup_map(self):
        try:
            with open(self.TRACK_XODR, "r") as f:
                opendrive_data = f.read()
            self.get_logger().info(
                "Raw OpenDRIVE data loaded (length: {})".format(len(opendrive_data))
            )

            opendrive_params = carla.OpendriveGenerationParameters(
                # vertex_distance=2.0,
                # max_road_length=500.0,
                wall_height=2.5,
                # additional_width=20.0,
                smooth_junctions=True,
                enable_mesh_visibility=True,
                enable_pedestrian_navigation=True,
            )

            self.get_logger().info(f"Using improved OpenDRIVE generation parameters")
            self.client.generate_opendrive_world(opendrive_data, opendrive_params)
            self.world = self.client.get_world()
            sleep(2)
            self.map = self.world.get_map()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

            bounds = self.get_map_bounds()
            self.get_logger().info(f"Map bounds: {bounds}")

            request_msg = String()
            request_msg.data = "Map is loaded"
            self.start_vehicle_manager.publish(request_msg)
            self.get_logger().info("Map setup complete.")

            # Visualize waypoints with better information
            self.visualize_track()

        except Exception as e:
            self.get_logger().error(f"Error during map setup: {e}")
            self.destroy_node()

    def visualize_track(self):
        """Create detailed track visualization with waypoint information"""
        waypoints = self.map.generate_waypoints(2.0)  # Get waypoints every 1 meter

        # Draw all waypoints
        for wp in waypoints:
            # Draw point for each waypoint
            self.world.debug.draw_point(
                wp.transform.location,
                size=0.05,
                color=carla.Color(0, 0, 255),  # Blue for regular waypoints
                life_time=0.0,  # Persistent
            )

    def get_map_bounds(self):
        waypoints = self.map.generate_waypoints(1.0)  # Waypoints every 2 meters
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        for waypoint in waypoints:
            location = waypoint.transform.location
            min_x, max_x = min(min_x, location.x), max(max_x, location.x)
            min_y, max_y = min(min_y, location.y), max(max_y, location.y)
        return min_x, max_x, min_y, max_y


def main(args=None):
    rclpy.init(args=args)
    node = LoadMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LoadMapNode interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
