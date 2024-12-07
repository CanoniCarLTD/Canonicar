import carla
import numpy as np
import cv2

class CameraModel:
    def __init__(self, carla_model, vehicle=None):
        self.carla_model = carla_model
        self.vehicle = vehicle
        self.camera = None
        self.frame = None

    def setup_camera(self, camera_location=(1.5, 2.4, 0), camera_type='sensor.camera.rgb'):
        blueprint_library = self.carla_model.world.get_blueprint_library()
        camera_bp = blueprint_library.find(camera_type)
        camera_transform = carla.Transform(carla.Location(*camera_location))
        
        # Attach camera to vehicle if provided, else spawn standalone
        if self.vehicle:
            self.camera = self.carla_model.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        else:
            self.camera = self.carla_model.world.spawn_actor(camera_bp, camera_transform)

        # Listen for the camera frames
        self.camera.listen(self.process_frame)

    def process_frame(self, image):
        """Process the frame from the camera."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA
        frame_bgr = array[:, :, :3]  # Convert to BGR
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        self.frame = buffer.tobytes()

    def get_frame(self):
        """Return the current camera frame."""
        return self.frame
