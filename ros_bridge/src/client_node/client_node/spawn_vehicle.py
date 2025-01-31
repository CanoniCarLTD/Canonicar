import rclpy
from rclpy.node import Node
import carla
from carla import Client, Transform, Location, Rotation, TrafficManager
import json
import dotenv
import os
from ament_index_python.packages import get_package_share_directory

# For ROS2 messages
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from std_msgs.msg import Header

import numpy as np

# Import conversion functions from sensors_data
from sensors_data import (
    carla_image_to_ros_image,
    carla_lidar_to_ros_pointcloud2,
    carla_imu_to_ros_imu,
    carla_gnss_to_ros_navsatfix,
)

class SpawnVehicleNode(Node):
    def __init__(self):
        super().__init__('spawn_vehicle_node')

        self.declare_parameter('host', '')
        self.declare_parameter('port', 2000)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value

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
            self.get_logger().error(f"Error connecting to CARLA server: {e}")
            return

        self.sensors_publishers = {}

        self.sensor_config_file = os.path.join(
            get_package_share_directory('client_node'), 'client_node', 'sensors_config.json'
        )
        self.vehicle = None
        self.spawn_objects_from_config()

    def spawn_objects_from_config(self):
        """
        Reads the 'objects' array from sensors_config.json
        Spawns the first 'vehicle.*' it finds using the JSON's spawn_point
        Then spawns sensors attached to that vehicle.
        """
        try:
            with open(self.sensor_config_file, 'r') as f:
                config = json.load(f)
                self.get_logger().info("Loaded JSON config from sensors_config.json")

            objects = config.get("objects", [])
            if not objects:
                self.get_logger().error("No objects found in sensors_config.json")
                return

            ego_object = objects[0]
            vehicle_type = ego_object.get("type", "")
            if not vehicle_type.startswith("vehicle."):
                self.get_logger().error("No valid vehicle object found in JSON.")
                return

            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(vehicle_type)
            if not vehicle_bp:
                self.get_logger().error(f"Vehicle blueprint '{vehicle_type}' not found in CARLA.")
                return
            self.get_logger().info(f"Found vehicle blueprint '{vehicle_type}'")

            vehicle_id = ego_object.get("id", "ego_vehicle")
            self.get_logger().info(f"Setting role_name to '{vehicle_id}'")

            spawn_point_data = ego_object.get("spawn_point", {})
            self.get_logger().info(f"Found spawn point data: {spawn_point_data}")
            vehicle_transform = Transform(
                Location(
                    x=spawn_point_data.get("x", 0.0),
                    y=spawn_point_data.get("y", 0.0),
                    z=spawn_point_data.get("z", 0.0),
                ),
                Rotation(
                    roll=spawn_point_data.get("roll", 0.0),
                    pitch=spawn_point_data.get("pitch", 0.0),
                    yaw=spawn_point_data.get("yaw", 0.0),
                ),
            )
            self.get_logger().info(f"Spawning vehicle '{vehicle_id}' at {vehicle_transform}")

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, vehicle_transform)
            if self.vehicle:
                self.get_logger().info(
                    f"Spawned vehicle '{vehicle_id}' with ID={self.vehicle.id} at {vehicle_transform}"
                )
            else:
                self.get_logger().error("Failed to spawn the vehicle. Check collisions or map issues.")
                return

            sensors = ego_object.get("sensors", [])
            if not sensors:
                self.get_logger().warn("No sensors found for the vehicle in JSON.")
            else:
                self.spawn_sensors(sensors)

            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, -1000)

        except Exception as e:
            self.get_logger().error(f"Error parsing JSON or spawning objects: {e}")

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
                    self.get_logger().error(f"Sensor blueprint '{sensor_type}' not found. Skipping.")
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

                sensor_actor = self.world.try_spawn_actor(sensor_bp, spawn_transform, attach_to=self.vehicle)
                if sensor_actor is None:
                    self.get_logger().error(f"Failed to spawn sensor '{sensor_id}'.")
                    continue
                else:
                    self.spawned_sensors.append(sensor_actor)

                self.get_logger().info(f"Spawned sensor '{sensor_id}' ({sensor_type}) at {spawn_transform}")

                # Create a ROS2 publisher for this sensor
                topic_name, msg_type = self.get_ros_topic_and_type(sensor_type, sensor_id)
                if topic_name and msg_type:
                    publisher = self.create_publisher(msg_type, topic_name, 10)
                    self.sensors_publishers[sensor_actor.id] = {
                        "sensor_id": sensor_id,
                        "sensor_type": sensor_type,
                        "publisher": publisher,
                        "msg_type": msg_type,
                    }

                    # Attach a listener callback to the CARLA sensor
                    def debug_listener(data, actor_id=sensor_actor.id):
                        # self.get_logger().debug(f"Received data from sensor {actor_id}")
                        self.sensor_data_callback(actor_id, data)

                    sensor_actor.listen(debug_listener)
                else:
                    self.get_logger().warn(
                        f"No recognized ROS message type for sensor '{sensor_type}'. Not publishing."
                    )

                # Handle attached pseudo-actors if any
                attached_objects = sensor_def.get("attached_objects", [])
                for ao in attached_objects:
                    self.get_logger().info(f"Detected attached object (pseudo) {ao['type']} with id '{ao['id']}'")
                    # Implement handling of attached objects if necessary.

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
        elif sensor_type.startswith("sensor.other.gnss"):
            return (f"/carla/{sensor_id}/gnss", NavSatFix)
        else:
            return (None, None)

    def sensor_data_callback(self, actor_id, data):
        """
        Single callback for all sensors. We look up the sensor_type to figure out how to convert.
        """
        if actor_id not in self.sensors_publishers:
            return

        pub_info = self.sensors_publishers[actor_id]
        sensor_type = pub_info["sensor_type"]
        publisher = pub_info["publisher"]
        msg_type = pub_info["msg_type"]

        # Create a header for the message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = pub_info["sensor_id"] 

        # Convert CARLA data to ROS message using sensors_data module
        if sensor_type.startswith("sensor.camera"):
            msg = carla_image_to_ros_image(data, header)
        elif sensor_type.startswith("sensor.lidar"):
            msg = carla_lidar_to_ros_pointcloud2(data, header)
        elif sensor_type.startswith("sensor.other.imu"):
            msg = carla_imu_to_ros_imu(data, header)
        elif sensor_type.startswith("sensor.other.gnss"):
            msg = carla_gnss_to_ros_navsatfix(data, header)
        else:
            msg = None

        if msg:
            publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpawnVehicleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
