import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros_interfaces.srv import SwapMap, RespawnVehicle, VehicleReady
from enum import Enum, auto
import time

class SimState(Enum):
    INITIALIZING = auto()
    MAP_LOADING = auto()
    VEHICLE_SPAWNING = auto()
    RUNNING = auto()
    RESPAWNING = auto()
    MAP_SWAPPING = auto()
    ERROR = auto()

class SimulationCoordinator(Node):
    def __init__(self):
        super().__init__('simulation_coordinator')
        
        self.declare_parameter('max_collisions', 3)
        self.max_collisions = self.get_parameter('max_collisions').value
        
        self.state = SimState.INITIALIZING
        self.collision_count = 0
        
        self.state_pub = self.create_publisher(
            String, '/simulation/state', 10)
            
        self.map_swap_client = self.create_client(SwapMap, 'map_swap')
        self.respawn_client = self.create_client(RespawnVehicle, 'respawn_vehicle')
        
        self.vehicle_ready_server = self.create_service(
            VehicleReady, 
            'vehicle_ready', 
            self.handle_vehicle_ready_service
        )
        
        self.collision_sub = self.create_subscription(
            String, '/collision_detected', self.handle_collision, 10)
        
        self.map_state_sub = self.create_subscription(
            String, '/map/state', self.handle_map_state, 10)
        
        self.episode_complete_sub = self.create_subscription(
            String, '/episode_complete', self.handle_episode_complete, 10)
        
        self.wait_for_services_timer = self.create_timer(2.0, self.check_service_availability)

        # Create timer for periodic state checks
        self.timer = self.create_timer(0.1, self.state_check)

        self.last_collision_time = 0.0
        self.last_respawn_time = 0.0
        self.collision_cooldown = 1.0  # seconds between collision handling
        self.respawn_in_progress = False
        self.vehicle_id = None

        self.map_state_translation = {
            "MAP_LOADING": SimState.MAP_LOADING,
            "MAP_READY": SimState.VEHICLE_SPAWNING,
            "MAP_ERROR": SimState.ERROR,
            "MAP_SWAPPING": SimState.MAP_SWAPPING,
        }
        
        self.get_logger().info('Simulation Coordinator started')
        self.publish_state('Initializing simulation')

    def state_check(self):
        """Periodic state check with health monitoring"""
        # Initial setup check
        if self.state == SimState.INITIALIZING:
            if self.map_swap_client.service_is_ready() and self.respawn_client.service_is_ready():
                self.state = SimState.RUNNING
                self.publish_state('Simulation running')
        
        # Check for stuck respawn 
        elif self.state == SimState.RESPAWNING and self.respawn_in_progress:
            current_time = time.time()
            if current_time - self.last_respawn_time > 10.0:
                self.get_logger().warn(f"Potential stuck respawn detected after {current_time - self.last_respawn_time:.1f} seconds")
                self.respawn_in_progress = False
                self.trigger_respawn("retry_stuck_respawn")
        
        
        elif not self.map_swap_client.service_is_ready() or not self.respawn_client.service_is_ready():
            self.get_logger().warn("Waiting for critical services to become available...")
    
    def check_service_availability(self):
        """Check if critical services are available"""
        if self.map_swap_client.service_is_ready() and self.respawn_client.service_is_ready():
            self.get_logger().info("All critical services are now available")
            self.wait_for_services_timer.cancel()
            
            # If we're in INITIALIZING state, move to next state
            if self.state == SimState.INITIALIZING:
                self.state = SimState.MAP_LOADING
                self.publish_state("Waiting for map to load")


    def handle_episode_complete(self, msg):
        """Handle episode completion events"""
        # Only handle in RUNNING state
        if self.state != SimState.RUNNING:
            return
            
        completion_reason = msg.data
        self.get_logger().info(f"Episode completed: {completion_reason}")
        self.trigger_respawn(f"episode_complete:{completion_reason}")
    
    def handle_collision(self, msg):
        """Handle collision events with cooldown protection"""
        # Only handle collisions in RUNNING state
        if self.state != SimState.RUNNING:
            return
            
        # Implement cooldown to prevent multiple rapid collisions
        current_time = time.time()
        if current_time - self.last_collision_time < self.collision_cooldown:
            return
            
        self.last_collision_time = current_time
        self.collision_count += 1
        self.get_logger().info(f'Processing collision: {self.collision_count}/{self.max_collisions}')
        
        # Check if we need to swap maps
        if self.collision_count >= self.max_collisions:
            self.trigger_map_swap()
        else:
            self.trigger_respawn(f"Collision: {msg.data}")
    
    def handle_map_state(self, msg):
        """Translate map state messages to simulation states"""
        map_state = msg.data.split(':')[0] if ':' in msg.data else msg.data
        
        if map_state in self.map_state_translation:
            new_state = self.map_state_translation[map_state]
            details = msg.data.split(':')[1] if ':' in msg.data else ""
            
            # Handle MAP_READY specially to trigger respawn after map load
            if map_state == "MAP_READY":
                self.state = SimState.VEHICLE_SPAWNING
                self.publish_state(details)
                
                # Trigger vehicle respawn now that map is ready
                self.trigger_respawn("after_map_swap")
            else:
                if new_state != self.state:
                    self.state = new_state
                    self.publish_state(details)
    
    def handle_vehicle_ready_service(self, request, response):
        """Service handler for vehicle ready notifications"""
        vehicle_id = request.vehicle_id
        self.get_logger().info(f'Vehicle ready notification received for ID: {vehicle_id}')
        
        self.vehicle_id = vehicle_id
        
        if self.state in [SimState.VEHICLE_SPAWNING, SimState.RESPAWNING, SimState.ERROR]:
            self.respawn_in_progress = False
            self.state = SimState.RUNNING
            self.publish_state(f'RUNNING:vehicle_{vehicle_id}_ready')
            
            response.success = True
            response.message = f"Simulation coordinator acknowledged vehicle {vehicle_id}"
        else:
            # Even if already in RUNNING state, still publish the vehicle ready notification
            self.publish_state(f'RUNNING:vehicle_{vehicle_id}_ready')
            response.success = True
            response.message = f"Acknowledging vehicle {vehicle_id}"
            
        return response
    
    def trigger_respawn(self, reason):
        """Trigger vehicle respawn"""
        if self.respawn_in_progress:
            self.get_logger().info(f"Respawn already in progress, ignoring new request: {reason}")
            return
            
        if not self.respawn_client.service_is_ready():
            self.get_logger().error('Cannot respawn - service not available')
            return
                
        self.respawn_in_progress = True
        self.state = SimState.RESPAWNING
        if "collision" in reason:
            self.publish_state('Respawning vehicle due to collision')
        elif "episode_complete" in reason:
            self.publish_state('Respawning vehicle due to episode_complete')
        self.last_respawn_time = time.time()
        request = RespawnVehicle.Request()
        request.reason = reason
        
        future = self.respawn_client.call_async(request)
        future.add_done_callback(self.respawn_done)
    
    def respawn_done(self, future):
        """Handle respawn completion"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Respawn successful: {response.message}")
                self.last_collision_time = time.time()
                self.respawn_in_progress = False
            else:
                self.respawn_in_progress = False
                self.retry_timer = self.create_timer(2.0, self.retry_respawn)
        except Exception as e:
            self.respawn_in_progress = False
            self.retry_timer = self.create_timer(2.0, self.retry_respawn)
    
    def retry_respawn(self):
        """Retry the respawn after a failure"""
        self.retry_timer.cancel()
        self.trigger_respawn("retry_after_failure")
        
    def trigger_map_swap(self):
        """Trigger map swap"""
        self.state = SimState.MAP_SWAPPING
        self.publish_state('Swapping map')
        
        request = SwapMap.Request()
        request.map_file_path = "" 
        
        future = self.map_swap_client.call_async(request)
        future.add_done_callback(self.map_swap_done)
    
    def map_swap_done(self, future):
        """Handle map swap completion"""
        try:
            response = future.result()
            if response.success:
                self.collision_count = 0  
            else:
                self.state = SimState.ERROR
                self.publish_state(f'Map swap failed')
        except Exception as e:
            self.state = SimState.ERROR
            self.publish_state(f'Map swap service error')
    
    def publish_state(self, details=""):
        """Publish current simulation state"""
        msg = String()
        if ":" in details:
            msg.data = details
        else:
            msg.data = f"{self.state.name}:{details}"
        self.state_pub.publish(msg)
        self.get_logger().info(f'State: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = SimulationCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()