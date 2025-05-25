import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from carla import Client, VehicleControl
from  ros_interfaces.srv import VehicleReady
class VehicleControlNode(Node):
    def __init__(self):
        super().__init__('vehicle_control_node')
                
        self.declare_parameter("host", "localhost") 
        self.declare_parameter("port", 2000)        
        self.declare_parameter("vehicle_type", "vehicle.tesla.model3")

        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.vehicle_type = self.get_parameter("vehicle_type").value
        
        try:
            self.client = Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server")
            return
        
        self.get_logger().info(
            f"Using launch parameters - host: x, port: {self.port}"
        )

        
        self.vehicle = None

        self.control_sub = self.create_subscription(
            Float32MultiArray,
            '/carla/vehicle_control',
            self.control_callback,
            10
        )

        self.collision_sub = self.create_subscription(
            String,
            '/collision_detected',
            self.collision_callback,
            10
        )
        
        # Subscribe to simulation state
        self.state_sub = self.create_subscription(
            String,
            '/simulation/state',
            self.state_callback,
            10
        )
        
        # Create vehicle ready service server
        self.vehicle_ready_server = self.create_service(
            VehicleReady,
            'vehicle_ready',
            self.handle_vehicle_ready
        )

        self.get_logger().info('VehicleControlNode initialized')


    def state_callback(self, msg):
        """Handle simulation state updates"""
        state_msg = msg.data
        
        if ':' in state_msg:
            state_name, details = state_msg.split(':', 1)
        else:
            state_name = state_msg
            details = ""
        
        if state_name == "RESPAWNING":
            if "Relocating" in details or "vehicle_relocated" in details:
                # For relocation, we keep the vehicle reference but apply emergency brake
                if self.vehicle and self.vehicle.is_alive:
                    control = VehicleControl()
                    control.steer = 0.0
                    control.throttle = 0.0
                    control.brake = 1.0
                    self.vehicle.apply_control(control)
                    self.get_logger().info("Applied emergency brake for vehicle relocation")
            else:
                self.vehicle = None
                self.get_logger().info(f"Vehicle control suspended: {details}")
        elif state_name == "MAP_SWAPPING":
            # For map swapping, always clear the vehicle
            self.vehicle = None
            self.get_logger().info(f"Vehicle control suspended: {details}")


    def collision_callback(self, msg):
        """Handle collision events - just log, let SimulationCoordinator manage response"""
        collision_info = msg.data
        self.get_logger().warn(f"Collision detected: {collision_info}")
        
        if self.vehicle:
            control = VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            self.get_logger().info("Applied emergency brake after collision")
    
    def handle_vehicle_ready(self, request, response):
        """Service handler when spawn_vehicle_node notifies that a vehicle is ready"""
        try:
            self.vehicle = self.world.get_actor(request.vehicle_id)
            
            if self.vehicle:
                response.success = True
                response.message = "Vehicle control ready"
            else:
                response.success = False
                response.message = f"Vehicle with ID {request.vehicle_id} not found"
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
        
    def control_callback(self, msg):
        if len(msg.data) != 3:
            self.get_logger().error('Control message must contain [throttle, steer, brake]')
            return
            
        steer, throttle, brake = msg.data
        if self.vehicle and self.vehicle.is_alive:
            control = VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            # self.vehicle.apply_control(control)
        else:
            self.get_logger().warn("Control message received but no vehicle available")

def main(args=None):
    rclpy.init(args=args)
    node = VehicleControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()