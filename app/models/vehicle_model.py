import random

class VehicleModel:
    def __init__(self, blueprint_library, world):
        self.blueprint_library = blueprint_library
        self.world = world
        self.vehicle = None

    def random_point_spawn_vehicle(self, vehicle_bp="model3", x=42, y=-100, z=1, yaw=180):
        map = self.world.get_map()
        # Get all predefined spawn points
        spawn_points = map.get_spawn_points()
        random.shuffle(spawn_points)
        spawn_point = spawn_points[0]
        vehicle_bp = self.blueprint_library.filter(vehicle_bp)[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        return self.vehicle
