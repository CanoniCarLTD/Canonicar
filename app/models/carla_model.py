import time
import carla

ETAI = "5.29.228.0"
KFIR = "109.67.132.31"
KFIR_LOCAL = "10.0.0.16"
class CarlaModel:
    def __init__(self, host=KFIR_LOCAL, port=2000, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.cameras = []
        self.vehicles = []

    def connect(self):
        """Connect to the CARLA server and set up the world."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        print("Connected to CARLA:", self.client.get_server_version())
        # self.destroy_all_actors()  # Destroy all actors in the simulation
        time.sleep(2)  # Wait for the world to be ready

    def destroy_all_vehicles_and_cameras(self):
        """Destroy all actors in the simulation."""
        for vehicle in self.vehicles:
            vehicle.vehicle_actor.destroy()
        for camera in self.cameras:
            camera.camera.destroy()
        self.vehicles = []
        self.cameras = []


    
