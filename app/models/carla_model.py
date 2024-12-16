import time
import carla

ITAY = "5.29.228.0"
KFIR = "109.67.132.31"

class CarlaModel:
    def __init__(self, host=ITAY, port=2000, timeout=2.0):
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

    def destroy_all_vehicles(self):
        """Destroy all actors in the simulation."""
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            actor.destroy()
