import time
import carla
import dotenv
import os

dotenv.load_dotenv()

ETAI = os.getenv("ETAI_IP")
KFIR = os.getenv("KFIR_IP")


class CarlaModel:
    def __init__(self, host=KFIR, port=2000, timeout=4.0):
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

    def load_map(self, map_name):
        if self.client is None:
            self.connect()  # Ensure connection is established
        if self.client is not None:
            try:
                self.world = self.client.load_world(map_name)
                print("Loaded map:", map_name)
            except Exception as e:
                print(f"Failed to load map: {e}")
        else:
            raise ValueError("Client is not initialized")
