import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from carla import Client, VehicleControl

class VehicleControlNode(Node):
    def __init__(self):
        super().__init__('vehicle_control_node')
                
        # # Set up CARLA client
        # self.client = carla.Client('localhost', 2000)
        # self.client.set_timeout(2.0)
        # self.world = self.client.get_world()
        self.declare_parameter("host", "")
        self.declare_parameter("port", 2000)
        self.declare_parameter("vehicle_type", "vehicle.tesla.model3")

        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.vehicle_type = self.get_parameter("vehicle_type").value

        self.get_logger().info(
            f"Connecting to CARLA server at {self.host}:{self.port} with vehicle {self.vehicle_type}"
        )

        try:
            self.client = Client(self.host, 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server: {e}")
            return
        
       
        
        self.vehicle = None

        self.get_logger().info("Creating start vehicle manager subscription")
        self.start_subscription = self.create_subscription(
            String, "/start_vehicle_manager", self.start_driving, 10  # QoS
        )
        
         # Subscribe to control commands
        self.control_sub = self.create_subscription(
            Float32MultiArray,
            '/carla/vehicle_control',
            self.control_callback,
            10
        )
        
        self.lap_subscription = self.create_subscription(
            String,
            "/lap_completed",
            self.lap_callback,
            10,
        )
        
        self.data_collector_ready = True
        self.map_loaded = False

        self.current_vehicle_index = 0
                        
        # Find the vehicle already spawned
        # self.vehicle = self.find_vehicle()        
        self.get_logger().info('VehicleControlNode initialized')

    def start_driving(self, msg):
        self.get_logger().info(f"Received start signal: {msg.data}")
        if msg.data == "Map is loaded":
            self.get_logger().info("Map is loaded")
            self.map_loaded = True
        if msg.data == "DataCollector is ready":
            self.get_logger().info("DataCollector is ready")
            self.data_collector_ready = True
        if self.map_loaded and self.data_collector_ready:
            self.get_logger().info("Starting to drive")
            self.spawn_objects_from_config()
            self.get_logger().info("Spanwed objects from config")
        self.get_logger().info(
            f"Is map loaded?: {self.map_loaded}, Is data collector ready {self.data_collector_ready}"
        )
        self.vehicle = self.find_vehicle()
        

    def lap_callback(self, msg):
        self.data_collector_ready = False
        self.get_logger().info(f"Lap completed! Destroy vehicle {self.vehicle_type}")
        self.destroy_actors()
        self.vehicle_type = self.vehicle_types[self.current_vehicle_index]
        self.current_vehicle_index = (self.current_vehicle_index + 1) % len(
            self.vehicle_types
        )
        

    def find_vehicle(self):
        vehicles = self.world.get_actors().filter('vehicle.*')
        if vehicles:
            self.get_logger().info(f'Found {len(vehicles)} vehicles, using the first one.')
            return vehicles[0]
        else:
            self.get_logger().error('No vehicles found in the world.')
            return None

    def control_callback(self, msg):
        self.get_logger().info(f'Received message from ppo node: {msg.data}')
        if len(msg.data) != 3:
            self.get_logger().error('Control message must contain [throttle, steer, brake]')
            return
            
        throttle, steer, brake = msg.data
        self.get_logger().info(f'received: throttle={throttle}, steer={steer}, brake={brake}')
        
        if self.vehicle:
            control = VehicleControl()
            control.throttle = float(throttle)
            control.steer = float(steer)
            control.brake = float(brake)
            self.get_logger().info('Applying control to vehicle')   
            self.vehicle.apply_control(control)

def main(args=None):
    rclpy.init(args=args)
    node = VehicleControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()