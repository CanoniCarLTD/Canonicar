from app.models.camera_model import CameraModel
from app.models.vehicle_model import VehicleModel


class CarlaFactory:
    def __init__(self, carla_model):
        self.carla_model = carla_model
        self.blueprint_library = None

    def create_vehicle(self):
        if not self.blueprint_library:
            self.blueprint_library = self.carla_model.world.get_blueprint_library()
        vehicle = VehicleModel(self.blueprint_library)
        self.carla_model.vehicles.append(vehicle)
        return vehicle

    def create_camera(self, vehicle=None):
        if not self.blueprint_library:
            self.blueprint_library = self.carla_model.world.get_blueprint_library()
        camera = CameraModel(self.carla_model, vehicle)
        self.carla_model.cameras.append(camera)
        return camera
        