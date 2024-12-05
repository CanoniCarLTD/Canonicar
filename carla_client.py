# carla_client.py
import carla
import numpy as np
import cv2


class CarlaClient:
    def __init__(self, host, port, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        print("Connected to CARLA: ", self.client.get_server_version())
        self.destroy_all_actors()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]
        transform = carla.Transform(
            carla.Location(x=-6, y=84, z=0.3), carla.Rotation(yaw=180)
        )
        vehicle = self.world.spawn_actor(vehicle_bp, transform)
        vehicle.set_autopilot(True)

    def setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust camera position as needed

        # Attach camera to a vehicle or standalone
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print("Camera attached")

    def start_stream(self, frame_callback):
        if not self.camera:
            raise RuntimeError("Camera is not set up. Call setup_camera() first.")
        self.camera.listen(frame_callback)

    def stop_camera(self):
        if self.camera:
            self.camera.destroy()

    def destroy_all_actors(self):
        actors = self.world.get_actors()
        for actor in actors:
            actor.destroy()
