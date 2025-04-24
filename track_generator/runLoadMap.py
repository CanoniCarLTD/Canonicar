import random
import carla
from time import sleep
import open3d as o3d
import glob
import dotenv
import os
import xml.etree.ElementTree as ET
import math

dotenv.load_dotenv()

ETAI = os.getenv("ETAI_IP")
KFIR = os.getenv("KFIR_IP")
TRACK_XODR = os.getenv("TRACK_XODR")
CARLA_SERVER_PORT = 2000


def draw_vehicle_bounding_box(world, vehicle, life_time=1.0):
    """
    Draws a bounding box around a vehicle in the CARLA world.
    """
    bounding_box = vehicle.bounding_box
    vehicle_transform = vehicle.get_transform()
    world_location = vehicle_transform.transform(bounding_box.location)
    world_bounding_box = carla.BoundingBox(world_location, bounding_box.extent)
    world.debug.draw_box(
        box=world_bounding_box,
        rotation=vehicle_transform.rotation,
        thickness=0.1,
        color=carla.Color(0, 255, 0),
        life_time=life_time,
        persistent_lines=True,
    )
    print(f"Bounding box drawn around vehicle {vehicle.id}")


def is_out_of_bounds(location, bounds):
    """
    Check if a given location is outside the defined bounds.
    """
    x, y, z = location.x, location.y, location.z
    min_x, max_x, min_y, max_y = bounds
    return not (min_x <= x <= max_x and min_y <= y <= max_y)


def get_map_bounds(world):
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(2.0)
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")
    for waypoint in waypoints:
        location = waypoint.transform.location
        min_x, max_x = min(min_x, location.x), max(max_x, location.x)
        min_y, max_y = min(min_y, location.y), max(max_y, location.y)
    return min_x, max_x, min_y, max_y


try:
    client = carla.Client(ETAI, CARLA_SERVER_PORT)
    client.set_timeout(10.0)
    print("Connected to carla: ", client.get_server_version())
    print(f"Loading track: {TRACK_XODR}")
    with open(TRACK_XODR, "r") as f:
        opendrive_data = f.read()
    opendrive_params = carla.OpendriveGenerationParameters(
        # vertex_distance=2.0,
        # max_road_length=500.0,
        wall_height=1.0,
        # additional_width=20.0,
        smooth_junctions=True,
        enable_mesh_visibility=True,
        enable_pedestrian_navigation=True,
    )
    world = client.generate_opendrive_world(opendrive_data, opendrive_params)
    world = client.get_world()
    map = world.get_map()
    world.set_weather(carla.WeatherParameters.CloudyNoon)
    bounds = get_map_bounds(world)
    print("Map bounds: ", bounds)
    waypoints = map.generate_waypoints(1)
    for waypoint in waypoints:
        world.debug.draw_point(
            waypoint.transform.location,
            size=0.1,
            color=carla.Color(0, 0, 255),
            life_time=1000.0,
        )
    blueprint_library = world.get_blueprint_library()
    spawn_points = map.get_spawn_points()
    vehicles = []
    """ need to figure out why the vehicles disappear after almost completing the track """
    # collision_bp = blueprint_library.find("sensor.other.collision")
    # collision_sensor = world.spawn_actor(
    #     collision_bp, carla.Transform(), attach_to=vehicle
    # )
    # vehicles.append(vehicle)
    # print(
    #     "Vehicle spawned: ",
    #     vehicle.type_id,
    # )
    # print("Setting autopilot...")
    # vehicle.set_autopilot(False)
    # print("Autopilot set")
    # traffic_manager = client.get_trafficmanager()
    # traffic_manager.auto_lane_change(vehicle, True)
    # traffic_manager.force_lane_change(vehicle, False)
    # traffic_manager.ignore_vehicles_percentage(vehicle, 5.0)
    # traffic_manager.global_percentage_speed_difference(-100.0)
    while True:
        world.tick()
        # for vehicle in vehicles:
        #     location = vehicle.get_location()
        #     if is_out_of_bounds(location, bounds):
        #         print(f"Vehicle {vehicle.id} is out of bounds at {location}.")
        # sleep(0.05)
except Exception as e:
    # for vehicle in vehicles:
    #     if vehicle is not None:
    #         vehicle.destroy()
    print("error: ", e)
finally:
    world = client.reload_world()
    print("World cleaned")
    sleep(0.5)
