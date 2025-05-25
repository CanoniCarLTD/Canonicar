# ros_bridge/src/client_node/client_node/run_load_map.py

import rclpy
from rclpy.node import Node
import carla
from time import sleep
import open3d as o3d
import os
import xml.etree.ElementTree as ET
from std_msgs.msg import String
from ros_interfaces.srv import GetTrackWaypoints, SwapMap
import random
import glob


class LoadMapNode(Node):
    def __init__(self):
        super().__init__("load_map_node")
        self.get_logger().info("LoadMapNode started, managing the map...")

        # Declare ROS2 parameters with default values
        self.declare_parameter("host", "")
        self.declare_parameter("TRACK_LINE", "")
        self.declare_parameter("TRACK_XODR", "")
        self.declare_parameter("port", 2000)

        # Retrieve parameter values
        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.TRACK_LINE = (
            self.get_parameter("TRACK_LINE").get_parameter_value().string_value
        )
        self.TRACK_XODR = (
            self.get_parameter("TRACK_XODR").get_parameter_value().string_value
        )
        self.CARLA_SERVER_PORT = (
            self.get_parameter("port").get_parameter_value().integer_value
        )

        # Validate parameters
        if not all([self.host, self.TRACK_LINE, self.TRACK_XODR]):
            self.get_logger().error("One or more required parameters are not set.")
            self.destroy_node()
            return
        
        # Add waypoints service
        self.waypoints_service = self.create_service(
            GetTrackWaypoints, 'get_track_waypoints', self.handle_waypoints_request
        )
        
        self.track_waypoints = []  
        self.track_length = 0.0 

        self.available_maps = []
        self.current_map_index = 0
        map_directory = os.path.dirname(self.TRACK_XODR)
        self.available_maps = glob.glob(f"{map_directory}/*.xodr")
        self.get_logger().info(f"Found {len(self.available_maps)} available maps: {[os.path.basename(m) for m in self.available_maps]}")

        # self.map_swap_service = self.create_service(
        #     SwapMap, 'map_swap', self.handle_map_swap
        # )

        self.state_publisher = self.create_publisher(
            String, '/map/state', 10
        )

        # if self.TRACK_XODR in self.available_maps:
        #     self.current_map_index = self.available_maps.index(self.TRACK_XODR)
        # else:
        #     self.current_map_index = 0

        # Initialize CARLA client and setup map
        try:
            self.client = carla.Client(self.host, self.CARLA_SERVER_PORT)
            self.client.set_timeout(10.0)
            self.get_logger().info(
                f"Connected to CARLA: {self.client.get_server_version()}"
            )
            self.setup_map()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CARLA client")
            self.destroy_node()

    def setup_map(self):
        try:
            self.publish_state("LOADING")
            # with open(self.TRACK_XODR, "r") as f:
            #     opendrive_data = f.read()
            # self.get_logger().info(
            #     "Raw OpenDRIVE data loaded (length: {})".format(len(opendrive_data))
            # )

            # opendrive_params = carla.OpendriveGenerationParameters(
            #     # vertex_distance=2.0,
            #     # max_road_length=500.0,
            #     wall_height=2.5,
            #     # additional_width=20.0,
            #     smooth_junctions=True,
            #     enable_mesh_visibility=True,
            #     enable_pedestrian_navigation=True,
            # )

            # self.get_logger().info(f"Using improved OpenDRIVE generation parameters")
            # self.client.generate_opendrive_world(opendrive_data, opendrive_params)
            self.world = self.client.get_world()

            settings = self.world.get_settings()
            settings.synchronous_mode = False
            # settings.fixed_delta_seconds = 0.025  # 40Hz simulation (4× sensor frequency)
            # settings.substepping = True
            # settings.max_substep_delta_time = 0.01  # 100Hz physics calculations
            # settings.max_substeps = 3  # Ensures 0.03 <= 0.01×3
            self.world.apply_settings(settings)
            
            spectator = self.world.get_spectator()
            spectator.set_transform(carla.Transform(carla.Location(x=-20, y=-4, z=9), carla.Rotation(pitch=-20, yaw=-80, roll=0)))
            
            sleep(2)
            self.map = self.world.get_map()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

            request_msg = String()
            request_msg.data = "Map is loaded"

            self.get_logger().info("Map setup complete.")

            # Visualize waypoints with better information
            self.visualize_track()
            self.extract_track_waypoints()
            self.get_logger().info(f"Extracted {len(self.track_waypoints)} waypoints from track")
            self.get_logger().info(f"Track length: {self.track_length:.2f} meters")

            sleep(1.0)
            self.publish_state("READY", f"Loaded {os.path.basename(self.TRACK_XODR)}")
        except Exception as e:
            self.publish_state("ERROR")
            self.get_logger().error(f"Error during map setup")
            self.destroy_node()

    def publish_state(self, state, details=""):
        msg = String()
        msg.data = f"MAP_{state.upper()}:{details}" if details else f"MAP_{state.upper()}"
        self.state_publisher.publish(msg)

    # Handle multiple quick map changes better:
    # def handle_map_swap(self, request, response):
    #     """Service handler for map swap requests"""
    #     self.publish_state("SWAPPING", "Preparing for map swap")
        
    #     # Add cleanup for existing map resources
    #     try:
    #         # Clean any existing visualizations
    #         self.world.debug.draw_point(
    #             carla.Location(0,0,0), size=0.1, 
    #             color=carla.Color(0,0,0,0), life_time=0.0
    #         )
    #         self.get_logger().info("Cleaned up previous map visualizations")
    #         self.world = self.client.reload_world()
    #     except:
    #         pass
        
    #     try:
    #         if request.map_file_path and os.path.exists(request.map_file_path):
    #             new_map_path = request.map_file_path
    #         else:
    #             current_idx = self.current_map_index
    #             next_idx = (current_idx + 1) % len(self.available_maps)
    #             new_map_path = self.available_maps[next_idx]
            
    #         self.TRACK_XODR = new_map_path
    #         self.current_map_index = self.available_maps.index(new_map_path)
            
    #         self.setup_map()
            
    #         self.extract_track_waypoints()
            
    #         self.publish_state("READY", f"Loaded {os.path.basename(new_map_path)}")
            
    #         response.success = True
    #         response.message = f"Successfully loaded map: {os.path.basename(new_map_path)}"
    #         return response
                
    #     except Exception as e:
    #         self.publish_state("ERROR", f"Map swap failed: {str(e)}")
    #         response.success = False
    #         response.message = f"Failed to swap map: {str(e)}"
    #         return response

    def extract_track_waypoints(self):
        """Extract waypoints and calculate track length"""
        waypoints = self.map.generate_waypoints(2.0)  # Get waypoints every 2 meters
        
        waypoints.sort(key=lambda wp: wp.s)
        
        self.track_waypoints = []
        for wp in waypoints:
            loc = wp.transform.location
            self.track_waypoints.append((loc.x, loc.y))
        
        # Calculate track length from waypoints
        self.track_length = 0.0
        for i in range(len(self.track_waypoints)):
            next_idx = (i + 1) % len(self.track_waypoints)
            x1, y1 = self.track_waypoints[i]
            x2, y2 = self.track_waypoints[next_idx]
            segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            self.track_length += segment_length

    def handle_waypoints_request(self, request, response):
        """Service handler for track waypoints requests"""
        try:
            if not self.track_waypoints:
                self.extract_track_waypoints()
                    
            # Fill response with waypoints
            response.waypoints_x = [p[0] for p in self.track_waypoints]
            response.waypoints_y = [p[1] for p in self.track_waypoints]
            response.track_length = self.track_length
            
            return response
                
        except Exception as e:
            self.get_logger().error(f"Error handling waypoints request: {str(e)}")
            return response

    def visualize_track(self):
        """Create detailed track visualization with waypoint information"""
        waypoints = self.map.generate_waypoints(2.0) 

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
        waypoints = self.map.generate_waypoints(2.0)  # Waypoints every 2 meters
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
