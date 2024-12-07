import carla

class VehicleModel:
    def __init__(self, blueprint_library):
        self.blueprint_library = blueprint_library
        self.vehicle = None

    def spawn_vehicle(self, vehicle_bp="model3", x=0, y=0, z=0, yaw=180):
        vehicle_bp = self.blueprint_library.filter(vehicle_bp)[0]
        transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
        self.vehicle = self.carla_model.world.spawn_actor(vehicle_bp, transform)
        self.vehicle.set_autopilot(True)
        return self.vehicle
