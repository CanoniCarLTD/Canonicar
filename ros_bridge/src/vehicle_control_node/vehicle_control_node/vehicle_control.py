import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from carla import Client, VehicleControl
from  ros_interfaces.srv import VehicleReady
class VehicleControlNode(Node):
    def __init__(self):
        super().__init__('vehicle_control_node')
                
        self.declare_parameter("host", "localhost")  # Default fallback value
        self.declare_parameter("port", 2000)        # Default fallback value
        self.declare_parameter("vehicle_type", "vehicle.tesla.model3")

        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.vehicle_type = self.get_parameter("vehicle_type").value
        
        try:
            self.client = Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA server: {e}")
            return
        
        self.get_logger().info(
            f"Using launch parameters - host: {self.host}, port: {self.port}"
        )

        
        self.vehicle = None

        self.control_sub = self.create_subscription(
            Float32MultiArray,
            '/carla/vehicle_control',
            self.control_callback,
            10
        )
        
        # Create vehicle ready service server
        self.vehicle_ready_server = self.create_service(
            VehicleReady,
            'vehicle_ready',
            self.handle_vehicle_ready
        )

        self.get_logger().info('VehicleControlNode initialized')

    
    def handle_vehicle_ready(self, request, response):
        """Service handler when spawn_vehicle_node notifies that a vehicle is ready"""
        self.get_logger().info(f'Received vehicle_ready notification with vehicle ID: {request.vehicle_id}')
        
        try:
            # Find the vehicle with the provided ID
            self.vehicle = self.world.get_actor(request.vehicle_id)
            
            if self.vehicle:
                self.get_logger().info(f'Successfully found vehicle with ID: {request.vehicle_id}')
                response.success = True
                response.message = f"Vehicle control started for vehicle {request.vehicle_id}"
            else:
                self.get_logger().error(f'Could not find vehicle with ID: {request.vehicle_id}')
                response.success = False
                response.message = f"Vehicle with ID {request.vehicle_id} not found"
        except Exception as e:
            self.get_logger().error(f'Error finding vehicle: {str(e)}')
            response.success = False
            response.message = f"Error: {str(e)}"
            
        return response
        
    def control_callback(self, msg):
        if len(msg.data) != 3:
            self.get_logger().error('Control message must contain [throttle, steer, brake]')
            return
            
        steer, throttle, brake = msg.data
        
        if self.vehicle:
            control = VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            self.vehicle.apply_control(control)

def main(args=None):
    rclpy.init(args=args)
    node = VehicleControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()