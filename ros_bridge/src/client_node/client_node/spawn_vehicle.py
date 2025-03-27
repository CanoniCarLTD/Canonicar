import rclpy
from rclpy.node import Node
import carla
from carla import Client, Transform, Location, Rotation, TrafficManager
import json
import os
from std_msgs.msg import Float32MultiArray
from ament_index_python.packages import get_package_share_directory
import time, random
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
from  ros_interfaces.srv import VehicleReady, RespawnVehicle, SwapMap

# For ROS2 messages
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from std_msgs.msg import Header

# Import conversion functions from sensors_data
from sensors_data import (
    carla_image_to_ros_image,
    carla_lidar_to_ros_pointcloud2,
    carla_imu_to_ros_imu,
)

class SpawnVehicleNode(Node):
    def __init__(self):
        super().__init__("spawn_vehicle_node")

        self.declare_parameter("host", "")
        self.declare_parameter("port", 2000)
        self.declare_parameter("vehicle_type", "")

        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.vehicle_type = self.get_parameter("vehicle_type").value

        self.get_logger().info(
            f"Connecting to CARLA server at host x:{self.port}"
        )

        try:
            self.client = Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.world.apply_settings(settings)
            self.spawned_sensors = []
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server")
            return

        self.sensors_publishers = {}

        self.sensor_config_file = os.path.join(
            get_package_share_directory("client_node"),
            "client_node",
            "sensors_config.json",
        )

        self.vehicle = None
        self.respawn_in_progress = False

        self.collision_publisher = self.create_publisher(
            String,
            "/collision_detected",
            10
        )

        self.vehicle_ready_pub = self.create_publisher(
            String, '/vehicle_ready', 10
        )

        self.location_publisher = self.create_publisher(
            Float32MultiArray, "/carla/vehicle/location", 10
        ) 
        self.timer = self.create_timer(0.1, self.publish_vehicle_location)

        self.vehicle_ready_client = self.create_client(VehicleReady, 'vehicle_ready')

        while not self.vehicle_ready_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Vehicle ready service not available, waiting...')

        self.state_subscription = self.create_subscription(
            String,
            '/simulation/state',
            self.handle_system_state,
            10
        )
 
    
        self.respawn_service = self.create_service(
            RespawnVehicle, 
            'respawn_vehicle', 
            self.respawn_vehicle_callback
        )


        self.lap_subscription = self.create_subscription(
            String,
            "/lap_completed",
            self.lap_callback,
            10,
        )
        
        self.spawn_objects_from_config()


    def lap_callback(self, msg):
        self.get_logger().info(f"Lap completed! Destroy vehicle {self.vehicle_type}")
        self.destroy_actors()


    def spawn_objects_from_config(self):

        self.get_logger().info("Waiting for map to fully load...")
        time.sleep(2.0)

        try:
            with open(self.sensor_config_file, "r") as f:
                config = json.load(f)
                self.get_logger().info("Loaded JSON config")

            objects = config.get("objects", [])
            if not objects:
                self.get_logger().error("No objects found in sensors_config.json")
                return

            ego_object = objects[0]
            vehicle_type = ego_object.get("type", self.vehicle_type)
            if not vehicle_type.startswith("vehicle."):
                self.get_logger().error("No valid vehicle object found in JSON.")
                return

            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(vehicle_type)

            carla_map = self.world.get_map()
            waypoints = carla_map.generate_waypoints(2.0)

            road_ids = set(wp.road_id for wp in waypoints)
            if len(road_ids) == 0:
                self.get_logger().error("No roads found in the map!")
                return

            main_road_id = list(road_ids)[0] if len(road_ids) == 1 else min(road_ids)

            lane_ids = set(wp.lane_id for wp in waypoints if wp.road_id == main_road_id)
            driving_lane_id = next(
                (id for id in lane_ids if id < 0),
                next((id for id in lane_ids if id > 0), 0),
            )

            if driving_lane_id == 0:
                self.get_logger().error("No valid driving lane found!")
                return

            road_waypoints = [
                wp
                for wp in waypoints
                if wp.road_id == main_road_id and wp.lane_id == driving_lane_id
            ]
            if not road_waypoints:
                self.get_logger().error(
                    "No waypoints found for the selected road/lane!"
                )
                return

            road_waypoints.sort(key=lambda wp: wp.s)

            spawn_waypoint = road_waypoints[3]
            self.get_logger().info(
                f"Using waypoint at s={spawn_waypoint.s:.1f} for spawn"
            )

            spawn_transform = spawn_waypoint.transform
            spawn_transform.location.z += 0.3

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            if not self.vehicle:
                self.get_logger().error("Failed to spawn at waypoint")
                return
            

            location = self.vehicle.get_location()
            self.get_logger().info(f"Spawned vehicle at {location}")

            collision_bp = blueprint_library.find("sensor.other.collision")
            if collision_bp:
                collision_sensor = self.world.spawn_actor(
                    collision_bp, carla.Transform(), attach_to=self.vehicle
                )
                collision_sensor.listen(lambda event: self._on_collision(event))
                self.spawned_sensors.append(collision_sensor)

            physics_control = self.vehicle.get_physics_control()
            physics_control.mass = physics_control.mass * 1.5
            self.vehicle.apply_physics_control(physics_control)
            sensors = ego_object.get("sensors", [])
            if sensors:
                self.spawn_sensors(sensors)
                
            self.notify_vehicle_ready()
            self.world.tick()

            '''
                AUTOPILOT
            '''
            # self.traffic_manager.global_percentage_speed_difference(0)
            # self.traffic_manager.auto_lane_change(self.vehicle, False)
            # self.traffic_manager.random_left_lanechange_percentage(self.vehicle, 0)
            # self.traffic_manager.random_right_lanechange_percentage(self.vehicle, 0)
            # self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
        except Exception as e:
            self.get_logger().error(f"Error spawning vehicle: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())

    def publish_vehicle_location(self):
        if self.vehicle is not None and self.vehicle.is_alive:
            location = self.vehicle.get_location()
            msg = Float32MultiArray()
            msg.data = [location.x, location.y, location.z]
            self.location_publisher.publish(msg)

    def handle_system_state(self, msg):
        """Handle simulation state updates"""
        state_msg = msg.data
        
        if ':' in state_msg:
            state_name, details = state_msg.split(':', 1)
        else:
            state_name = state_msg
        
        if state_name == "VEHICLE_SPAWNING":
            if (not self.vehicle or not self.vehicle.is_alive) and not self.respawn_in_progress:
                self.get_logger().info("Vehicle spawn state detected, spawning new vehicle...")
                self.respawn_in_progress = True
                try:
                    self.destroy_actors()
                    self.spawn_objects_from_config()
                finally:
                    self.respawn_in_progress = False


    def respawn_vehicle_callback(self, request, response):
        """Handle respawn vehicle service calls"""
        self.get_logger().info(f"Respawn vehicle requested: {request.reason}")
        
        try:
            if self.respawn_in_progress:
                response.success = False
                response.message = "Respawn already in progress"
                return response
            
            self.respawn_in_progress = True
            self.destroy_actors()
            
            time.sleep(1.0)
            
            self.spawn_objects_from_config()

            # Notify that vehicle is ready
            if self.vehicle and self.vehicle.is_alive:
                response.success = True
                response.message = f"Vehicle successfully respawned after: {request.reason}"
            else:
                response.success = False
                response.message = "Respawn failed: no valid vehicle created"
            
            self.respawn_in_progress = False
            return response
        except Exception as e:
            self.respawn_in_progress = False
            response.success = False
            response.message = f"Respawn failed: {str(e)}"
            return response
        

    def notify_vehicle_ready(self):
        """Call the vehicle_ready service to notify the control node"""
        if not self.vehicle or not self.vehicle.is_alive:
            return
            
        request = VehicleReady.Request()
        request.vehicle_id = self.vehicle.id
        
        self.get_logger().info(f"Notifying vehicle_control that vehicle {self.vehicle.id} is ready")
        future = self.vehicle_ready_client.call_async(request)

        ready_msg = String()
        ready_msg.data = str(self.vehicle.id)
        self.vehicle_ready_pub.publish(ready_msg)


    def _on_collision(self, event):
        """Callback for collision sensor"""
        collision_type = event.other_actor.type_id if event.other_actor else "unknown"
        collision_location = event.transform.location

        self.get_logger().warn(
            f"COLLISION: vehicle hit {collision_type} at "
            f"({collision_location.x:.1f}, {collision_location.y:.1f}, {collision_location.z:.1f})"
        )

        collision_msg = String()
        collision_msg.data = f"collision_with_{collision_type}"
        self.collision_publisher.publish(collision_msg)


    def spawn_sensors(self, sensors):
        """
        Given a list of sensor definitions (dict), spawn them and attach to self.vehicle.
        Also sets up a listener callback to publish data via ROS2.
        """
        blueprint_library = self.world.get_blueprint_library()

        for sensor_def in sensors:
            try:
                sensor_type = sensor_def.get("type")
                sensor_id = sensor_def.get("id", "unknown_sensor")

                if not sensor_type:
                    self.get_logger().error(
                        f"Sensor definition is missing 'type'. Skipping {sensor_def}."
                    )
                    continue

                sensor_bp = blueprint_library.find(sensor_type)
                if not sensor_bp:
                    self.get_logger().error(
                        f"Sensor blueprint '{sensor_type}' not found. Skipping."
                    )
                    continue

                for attribute, value in sensor_def.items():
                    if attribute in ["type", "id", "spawn_point", "attached_objects"]:
                        continue
                    if sensor_bp.has_attribute(attribute):
                        try:
                            sensor_bp.set_attribute(attribute, str(value))
                        except RuntimeError as e:
                            self.get_logger().error(
                                f"Error setting attribute '{attribute}' to '{value}' for '{sensor_type}': {e}"
                            )
                    else:
                        self.get_logger().warn(
                            f"Blueprint '{sensor_type}' does NOT have an attribute '{attribute}'. Skipping."
                        )

                sp = sensor_def.get("spawn_point", {})
                spawn_transform = Transform(
                    Location(
                        x=sp.get("x", 0.0),
                        y=sp.get("y", 0.0),
                        z=sp.get("z", 0.0),
                    ),
                    Rotation(
                        roll=sp.get("roll", 0.0),
                        pitch=sp.get("pitch", 0.0),
                        yaw=sp.get("yaw", 0.0),
                    ),
                )

                sensor_actor = self.world.try_spawn_actor(
                    sensor_bp, spawn_transform, attach_to=self.vehicle
                )
                if sensor_actor is None:
                    self.get_logger().error(f"Failed to spawn sensor '{sensor_id}'.")
                    continue
                else:
                    self.spawned_sensors.append(sensor_actor)
                self.get_logger().info(
                    f"Spawned sensor '{sensor_id}' ({sensor_type}) at {spawn_transform}"
                )

                topic_name, msg_type = self.get_ros_topic_and_type(sensor_type, sensor_id)
                if topic_name and msg_type:
                    publisher = self.create_publisher(msg_type, topic_name, 10)
                    self.sensors_publishers[sensor_actor.id] = {
                        "sensor_id": sensor_id,
                        "sensor_type": sensor_type,
                        "publisher": publisher,
                        "msg_type": msg_type,
                        "topic": topic_name 
                    }

                    def debug_listener(data, actor_id=sensor_actor.id):
                        self.sensor_data_callback(actor_id, data)

                    sensor_actor.listen(debug_listener)
                else:
                    self.get_logger().warn(
                        f"No recognized ROS message type for sensor '{sensor_type}'. Not publishing."
                    )

                attached_objects = sensor_def.get("attached_objects", [])
                for ao in attached_objects:
                    self.get_logger().info(
                        f"Detected attached object (pseudo) {ao['type']} with id '{ao['id']}'"
                    )
                

            except Exception as e:
                self.get_logger().error(f"Error spawning sensor '{sensor_def}': {e}")

    def get_ros_topic_and_type(self, sensor_type, sensor_id):
        """
        Returns a (topic_name, message_type) for the given CARLA sensor type.
        Modify naming as you prefer for your Foxglove setup.
        """
        if sensor_type.startswith("sensor.camera"):
            return (f"/carla/{sensor_id}/image_raw", Image)
        elif sensor_type.startswith("sensor.lidar"):
            return (f"/carla/{sensor_id}/points", PointCloud2)
        elif sensor_type.startswith("sensor.other.imu"):
            return (f"/carla/{sensor_id}/imu", Imu)
        else:
            return (None, None)

    def sensor_data_callback(self, actor_id, data):
        """Handle sensor data from Carla and publish to ROS"""
        try:
            matching_sensor = None
            for sensor in self.spawned_sensors:
                if sensor.id == actor_id:
                    matching_sensor = sensor
                    break
                    
            if not matching_sensor:
                return
                
            # Find the publisher for this sensor
            if actor_id in self.sensors_publishers:
                sensor_info = self.sensors_publishers[actor_id]
                sensor_type = sensor_info["sensor_type"]
                publisher = sensor_info["publisher"]
                sensor_id = sensor_info["sensor_id"]
                
                # Create a ROS header
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = f"{sensor_id}"
                
                # Convert the data to ROS format based on sensor type
                message = None
                if sensor_type.startswith('sensor.camera'):
                    message = carla_image_to_ros_image(data, header)
                elif sensor_type.startswith('sensor.lidar'):
                    message = carla_lidar_to_ros_pointcloud2(data, header)
                elif sensor_type.startswith("sensor.other.imu"):
                    message = carla_imu_to_ros_imu(data, header)
                else:
                    message = None
                    
                if message:
                    publisher.publish(message)

                    
        except Exception as e:
            self.get_logger().error(f"Error in sensor callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())  # Print full stack trace for debugging

            
    def destroy_actors(self):
        """
        Clean up all spawned actors (vehicle and sensors) when node shuts down.
        """
        self.get_logger().info("Destroying all spawned actors...")
        try:
            # Destroy all sensors first
            for i, sensor in enumerate(self.spawned_sensors):
                if sensor is not None:
                    try:
                        sensor.stop()
                        sensor.destroy()
                    except Exception as e:
                        self.get_logger().error(f"Failed to destroy sensor: {e}")
            
            self.spawned_sensors.clear()

            # Then destroy the vehicle
            if self.vehicle is not None and self.vehicle.is_alive:
                # Disable autopilot before destroying
                try:
                    self.vehicle.set_autopilot(False)
                except:
                    pass

                self.vehicle.destroy()
                self.get_logger().info(f"Destroyed vehicle {self.vehicle.id}")
                self.vehicle = None

            self.get_logger().info("All actors destroyed successfully")

        except Exception as e:
            self.get_logger().error(f"Error during actor cleanup: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = SpawnVehicleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_actors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
