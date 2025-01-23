import rclpy
from rclpy.node import Node
import carla
from carla import Client, Transform, Location, Rotation
import json
import dotenv
import os
from ament_index_python.packages import get_package_share_directory

# For ROS2 messages
from sensor_msgs.msg import Image, PointCloud2, PointField, Imu, NavSatFix
from std_msgs.msg import Header

import numpy as np
import struct

dotenv.load_dotenv()

ETAI = os.getenv("ETAI_IP")
KFIR = os.getenv("KFIR_IP")

# Load sensors_config.json from your ROS package share directory
share_dir = get_package_share_directory('client_node')
sensors_file_path = os.path.join(share_dir, 'client_node', 'sensors_config.json')

class SpawnVehicleNode(Node):
    def __init__(self):
        super().__init__('spawn_vehicle_node')

        # Use ROS2 params primarily for host/port
        self.declare_parameter('host', KFIR)
        self.declare_parameter('port', 2000)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value

        # Connect to CARLA
        try:
            self.client = Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.spawned_sensors = []
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server: {e}")
            return

        # We keep a map from actor_id -> publisher info
        # Example: self.sensors_publishers[actor_id] = {"type": <sensor_type>, "publisher": <publisher_obj>, "cb": <callback>}
        self.sensors_publishers = {}

        # Read JSON config and spawn objects
        self.sensor_config_file = sensors_file_path
        self.vehicle = None
        self.spawn_objects_from_config()

    # ----------------------------------------------------------------------
    # 1) SPAWN THE VEHICLE & SENSORS FROM JSON
    # ----------------------------------------------------------------------
    def spawn_objects_from_config(self):
        """
        Reads the 'objects' array from sensors_config.json
        Spawns the first 'vehicle.*' it finds using the JSON's spawn_point
        Then spawns sensors attached to that vehicle.
        """
        try:
            with open(self.sensor_config_file, 'r') as f:
                config = json.load(f)
                print("Loaded JSON config from sensors_config.json")

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
            print(f"Found vehicle blueprint '{vehicle_type}'")
            print(f"vehicle_bp: {vehicle_bp}")
            # Set the role_name from JSON "id" => e.g. "ego_vehicle"
            vehicle_id = ego_object.get("id", "ego_vehicle")
            print(f"Setting role_name to '{vehicle_id}'")
            # Get transform from JSON
            try:
                spawn_point_data = ego_object.get("spawn_point", {})
            except Exception as e:
                self.get_logger().error(f"Error parsing spawn_point data: {e}")
            print(f"Found spawn point data: {spawn_point_data}")
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
            print(f"Spawning vehicle '{vehicle_id}' at {vehicle_transform}")

            # Attempt to spawn the vehicle
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, vehicle_transform)
            if self.vehicle:
                self.get_logger().info(
                    f"Spawned vehicle '{vehicle_id}' with ID={self.vehicle.id} at {vehicle_transform}"
                )
            else:
                self.get_logger().error("Failed to spawn the vehicle. Check collisions or map issues.")
                return

            # Spawn the sensors for this vehicle
            sensors = ego_object.get("sensors", [])
            if not sensors:
                self.get_logger().warn("No sensors found for the vehicle in JSON.")
            else:
                self.spawn_sensors(sensors)

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

                # Set blueprint attributes (skip 'type', 'id', 'spawn_point', 'attached_objects')
                for attribute, value in sensor_def.items():
                    if attribute in ["type", "id", "spawn_point", "attached_objects"]:
                        continue
                    try:
                        sensor_bp.set_attribute(attribute, str(value))
                    except RuntimeError as e:
                        self.get_logger().error(
                            f"Error for line 157 setting str attribute '{attribute}' to '{value}' for '{sensor_type}': {e}"
                        )

                     # Check if the blueprint actually has this attribute
                    if not sensor_bp.has_attribute(attribute):
                        self.get_logger().warn(
                            f"Blueprint '{sensor_type}' does NOT have an attribute '{attribute}'. Skipping."
                        )
                        continue

                    try:
                        sensor_bp.set_attribute(attribute, str(value))
                    except RuntimeError as e:
                        self.get_logger().error(
                            f"Error setting attribute '{attribute}' to '{value}' for '{sensor_type}': {e}"
                        )

                # Build the transform
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

                # Attach sensor to the ego vehicle
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

                    # # Attach a listener callback to the CARLA sensor
                    # sensor_actor.listen(
                    #     lambda data, actor_id=sensor_actor.id: self.sensor_data_callback(actor_id, data)
                    # )
                else:
                    self.get_logger().warn(
                        f"No recognized ROS message type for sensor '{sensor_type}'. Not publishing."
                    )

                # If you have attached pseudo-actors, handle them similarly
                attached_objects = sensor_def.get("attached_objects", [])
                for ao in attached_objects:
                    self.get_logger().info(f"Detected attached object (pseudo) {ao['type']} with id '{ao['id']}'")
                    # ...
                    # You could spawn/attach these similarly if needed.

            except Exception as e:
                self.get_logger().error(f"Error spawning sensor '{sensor_def}': {e}")

    # ----------------------------------------------------------------------
    # 2) SET UP PUBLISHING LOGIC
    # ----------------------------------------------------------------------

    def get_ros_topic_and_type(self, sensor_type, sensor_id):
        """
        Returns a (topic_name, message_type) for the given CARLA sensor type.
        Modify naming as you prefer for your Foxglove setup.
        """
        if sensor_type.startswith("sensor.camera"):
            # E.g. "/carla/ego_vehicle/rgb_front/image_raw"
            return (f"/carla/{sensor_id}/image_raw", Image)
        elif sensor_type.startswith("sensor.lidar"):
            # E.g. "/carla/ego_vehicle/lidar/points"
            return (f"/carla/{sensor_id}/points", PointCloud2)
        elif sensor_type.startswith("sensor.other.imu"):
            return (f"/carla/{sensor_id}/imu", Imu)
        elif sensor_type.startswith("sensor.other.gnss"):
            return (f"/carla/{sensor_id}/gnss", NavSatFix)
        else:
            return (None, None)

    # def sensor_data_callback(self, actor_id, data):
    #     """
    #     Single callback for all sensors. We look up the sensor_type to figure out how to convert.
    #     """
    #     if actor_id not in self.sensors_publishers:
    #         return

    #     pub_info = self.sensors_publishers[actor_id]
    #     sensor_type = pub_info["sensor_type"]
    #     publisher = pub_info["publisher"]
    #     msg_type = pub_info["msg_type"]

    #     # We create a header for the message
    #     # (In a real app, you might want to handle sim vs. wall clock time, frames, etc.)
    #     header = Header()
    #     header.stamp = self.get_clock().now().to_msg()
    #     header.frame_id = pub_info["sensor_id"]  # or "map", "base_link", etc.

    #     if sensor_type.startswith("sensor.camera"):
    #         msg = self.carla_image_to_ros_image(data, header)
    #     elif sensor_type.startswith("sensor.lidar"):
    #         msg = self.carla_lidar_to_ros_pointcloud2(data, header)
    #     elif sensor_type.startswith("sensor.other.imu"):
    #         msg = self.carla_imu_to_ros_imu(data, header)
    #     elif sensor_type.startswith("sensor.other.gnss"):
    #         msg = self.carla_gnss_to_ros_navsatfix(data, header)
    #     else:
    #         # Unhandled type
    #         msg = None

    #     if msg:
    #         publisher.publish(msg)

    # ----------------------------------------------------------------------
    # 3) CONVERSION HELPERS
    # ----------------------------------------------------------------------

    # def carla_image_to_ros_image(self, carla_image, header):
    #     """
    #     Convert a carla.Image (BGRA) to a sensor_msgs/Image (e.g. in 'bgra8' or 'rgb8').
    #     By default, CARLA camera is in BGRA byte order.
    #     """
    #     img_msg = Image()
    #     img_msg.header = header
    #     img_msg.height = carla_image.height
    #     img_msg.width = carla_image.width
    #     img_msg.encoding = "bgra8"  # or "rgba8" / "bgr8" / etc. depending on your use
    #     img_msg.step = 4 * carla_image.width  # BGRA -> 4 bytes per pixel

    #     # raw_data is a bytes() object containing BGRA
    #     img_msg.data = bytes(carla_image.raw_data)

    #     return img_msg

    # def carla_lidar_to_ros_pointcloud2(self, carla_lidar_data, header):
    #     """
    #     Convert a carla.LidarMeasurement to sensor_msgs/PointCloud2.
    #     Each point in carla_lidar_data is [x, y, z, intensity].
    #     """
    #     pc_msg = PointCloud2()
    #     pc_msg.header = header

    #     # Define fields: x, y, z, intensity
    #     pc_msg.height = 1  # unorganized point cloud
    #     pc_msg.width = len(carla_lidar_data)
    #     pc_msg.is_bigendian = False
    #     pc_msg.is_dense = False

    #     pc_msg.fields = [
    #         PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    #         PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    #     ]
    #     pc_msg.point_step = 16  # 4 floats * 4 bytes each = 16

    #     # Build raw data array
    #     # carla_lidar_data is an iterable of carla.LidarDetection
    #     # [d.point.x, d.point.y, d.point.z, d.object_idx (or intensity?)]
    #     # In newer CARLA versions, it's x,y,z, cos_inc_angle, etc. 
    #     # We'll treat 4th value as 'intensity' if your sensor is configured that way.

    #     points = []
    #     for d in carla_lidar_data:
    #         # d has d.point.x, d.point.y, d.point.z, d.object_idx
    #         # Or in older versions: d.x, d.y, d.z, d.intensity
    #         # Adjust accordingly. Suppose we treat d.object_idx as intensity = 0.0 for demonstration
    #         x = float(d.point.x)
    #         y = float(d.point.y)
    #         z = float(d.point.z)
    #         intensity = 0.0  # or e.g. float(d.object_idx)
    #         points.append(struct.pack('ffff', x, y, z, intensity))

    #     pc_msg.data = b''.join(points)

    #     return pc_msg

    # def carla_imu_to_ros_imu(self, carla_imu_data, header):
    #     """
    #     Convert a carla.IMUMeasurement to sensor_msgs/Imu.
    #     carla_imu_data.accelerometer, .gyroscope => Vector3D
    #     """
    #     imu_msg = Imu()
    #     imu_msg.header = header

    #     # Carla IMU data fields
    #     #  - accelerometer: (x, y, z) in m/s^2
    #     #  - gyroscope: (x, y, z) in rad/s
    #     #  - compass (float) in radians, optional

    #     # Fill linear_acceleration
    #     imu_msg.linear_acceleration.x = float(carla_imu_data.accelerometer.x)
    #     imu_msg.linear_acceleration.y = float(carla_imu_data.accelerometer.y)
    #     imu_msg.linear_acceleration.z = float(carla_imu_data.accelerometer.z)

    #     # Fill angular_velocity
    #     imu_msg.angular_velocity.x = float(carla_imu_data.gyroscope.x)
    #     imu_msg.angular_velocity.y = float(carla_imu_data.gyroscope.y)
    #     imu_msg.angular_velocity.z = float(carla_imu_data.gyroscope.z)

    #     # Carla doesn’t provide a direct orientation quaternion from sensor.other.imu
    #     # We could fill with identity or compute from compass:
    #     # Suppose we have a 'compass' field in radians (0 = North, pi/2 = East, etc.)
    #     # You can do a partial orientation if you want. We'll skip it or set to 0.
    #     # orientation is left at default (x=0, y=0, z=0, w=1).

    #     return imu_msg

    # def carla_gnss_to_ros_navsatfix(self, carla_gnss_data, header):
    #     """
    #     Convert a carla.GnssMeasurement to sensor_msgs/NavSatFix.
    #     carla_gnss_data.latitude, .longitude, .altitude
    #     """
    #     navsat_msg = NavSatFix()
    #     navsat_msg.header = header
    #     navsat_msg.latitude = float(carla_gnss_data.latitude)
    #     navsat_msg.longitude = float(carla_gnss_data.longitude)
    #     navsat_msg.altitude = float(carla_gnss_data.altitude)

    #     # Typically, you'd set position_covariance, status, etc. as needed.
    #     navsat_msg.status.service = 1  # GPS
    #     navsat_msg.status.status = 0   # STATUS_FIX
    #     navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    #     return navsat_msg

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
