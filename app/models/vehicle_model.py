import random
import time
import carla


class VehicleModel:
    def __init__(self, blueprint_library, world):
        self.blueprint_library = blueprint_library
        self.world = world
        self.vehicle_actor = None

    def random_point_spawn_vehicle(
        self, vehicle_bp="model3", x=42, y=-100, z=1, yaw=180
    ):
        map = self.world.get_map()
        # Get all predefined spawn points
        spawn_points = map.get_spawn_points()
        random.shuffle(spawn_points)
        spawn_point = spawn_points[0]
        vehicle_bp = self.blueprint_library.filter(vehicle_bp)[0]
        self.vehicle_actor = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle_actor.set_autopilot(False)
        return self.vehicle_actor

    def control_vehicle(self, throttle, steer, brake):
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        print(control.throttle, control.steer)
        control.hand_brake = False
        self.vehicle_actor.apply_control(control)
        time.sleep(0.03)
