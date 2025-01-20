import rclpy
from rclpy.node import Node
from carla import Client, Transform, Location, Rotation
import json
import dotenv
import os

dotenv.load_dotenv()

ETAI = os.getenv("ETAI_IP")
KFIR = os.getenv("KFIR_IP")


class SpawnVehicleNode(Node):
    def __init__(self):
        super().__init__('spawn_vehicle_node')
        self.declare_parameter('host', 'KFIR_IP')
        self.declare_parameter('port', 2000)
        self.declare_parameter('vehicle_blueprint', 'vehicle.tesla.model3')
        self.declare_parameter('sensors_config_file', 'sensors_config.json')

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.vehicle_blueprint = self.get_parameter('vehicle_blueprint').value
        self.sensor_config_file = self.get_parameter('sensors_config_file').value

        self.client = None
        self.world = None
        self.vehicle = None

        try:
            # Initialize the CARLA client and world
            self.client = Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            # Spawn the vehicle
            self.spawn_vehicle()

            # Add sensors from JSON configuration
            self.add_sensors_from_config()
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server or fetching map: {e}")

    def spawn_vehicle(self):
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(self.vehicle_blueprint)

            if not vehicle_bp:
                self.get_logger().error(f"Vehicle blueprint '{self.vehicle_blueprint}' not found.")
                return

            # Use a predefined spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if len(spawn_points) < 1:
                self.get_logger().error("No spawn points available in the map.")
                return

            transform = spawn_points[2]  # Use the first spawn point
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)

            if self.vehicle:
                self.get_logger().info(f"Successfully spawned vehicle: {self.vehicle.id}")
            else:
                self.get_logger().error("Failed to spawn vehicle. Possibly a collision at the spawn location.")
        except Exception as e:
            self.get_logger().error(f"Error spawning vehicle: {e}")

    def add_sensors_from_config(self):
        try:
            with open(self.sensor_config_file, 'r') as f:
                config = json.load(f)

            sensors = config.get("sensors", [])
            for sensor in sensors:
                sensor_bp = self.world.get_blueprint_library().find(sensor["type"])
                if not sensor_bp:
                    self.get_logger().error(f"Sensor blueprint '{sensor['type']}' not found.")
                    continue

                # Set sensor attributes from JSON
                for attribute, value in sensor.items():
                    if attribute not in ["type", "id", "spawn_point"]:
                        sensor_bp.set_attribute(attribute, str(value))

                # Get the spawn point for the sensor
                spawn_point_config = sensor["spawn_point"]
                spawn_transform = Transform(
                    Location(x=spawn_point_config["x"], y=spawn_point_config["y"], z=spawn_point_config["z"]),
                    Rotation(roll=spawn_point_config["roll"], pitch=spawn_point_config["pitch"], yaw=spawn_point_config["yaw"])
                )

                # Attach sensor to the vehicle
                sensor_actor = self.world.try_spawn_actor(sensor_bp, spawn_transform, attach_to=self.vehicle)
                if sensor_actor:
                    self.get_logger().info(f"Successfully spawned sensor '{sensor['id']}' of type '{sensor['type']}'")
                else:
                    self.get_logger().error(f"Failed to spawn sensor '{sensor['id']}'")
        except Exception as e:
            self.get_logger().error(f"Error adding sensors: {e}")


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
