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
TRACK_LINE = os.getenv("TRACK_LINE")
TRACK_XODR = os.getenv("TRACK_XODR")
CARLA_SERVER_PORT = 2000


def draw_vehicle_bounding_box(world, vehicle, life_time=1.0):
    """
    Draws a bounding box around a vehicle in the CARLA world.
    """
    # Get the vehicle's bounding box and its transform
    bounding_box = vehicle.bounding_box
    vehicle_transform = vehicle.get_transform()

    # Apply the vehicle's transform to the bounding box location
    world_location = vehicle_transform.transform(bounding_box.location)

    # Create a new bounding box with the world space location
    world_bounding_box = carla.BoundingBox(world_location, bounding_box.extent)

    # Draw the bounding box in the world
    world.debug.draw_box(
        box=world_bounding_box,
        rotation=vehicle_transform.rotation,
        thickness=0.1,
        color=carla.Color(0, 255, 0),  # Red color for the box
        life_time=life_time,  # Duration for the box to be visible
        persistent_lines=True,  # Keep the box visible across ticks
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
    waypoints = carla_map.generate_waypoints(2.0)  # Waypoints every 2 meters

    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    for waypoint in waypoints:
        location = waypoint.transform.location
        min_x, max_x = min(min_x, location.x), max(max_x, location.x)
        min_y, max_y = min(min_y, location.y), max(max_y, location.y)

    return min_x, max_x, min_y, max_y


def read_line_file(file_path):
    """
    Reads the .line file and extracts x, y, and heading (rad).
    """
    centerline = []
    with open(file_path, "r") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                x, y, rad = map(float, line.strip().split(","))
                centerline.append((x, y, rad))
    return centerline


def calculate_segment_lengths(centerline):
    """
    Calculates the length of each segment and the cumulative road length.
    """
    segment_lengths = []
    for i in range(len(centerline) - 1):
        x1, y1, _ = centerline[i]
        x2, y2, _ = centerline[i + 1]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        segment_lengths.append(length)
    return segment_lengths


def create_xodr_file(
    centerline,
    segment_lengths,
    lane_width=10.5,
    file_name=TRACK_XODR,
):
    """
    Generates an OpenDRIVE file from the centerline.
    """
    # Create the root element
    opendrive = ET.Element("OpenDRIVE")

    # Create the header
    ET.SubElement(
        opendrive,
        "header",
        {
            "revMajor": "1",
            "revMinor": "4",
            "name": "GeneratedTrack",
            "version": "1.00",
            "date": "2024-12-21",
        },
    )

    # Calculate total road length
    total_length = sum(segment_lengths)

    # Create a road element
    road = ET.SubElement(
        opendrive,
        "road",
        {
            "name": "GeneratedRoad",
            "length": f"{total_length:.5f}",
            "id": "1",
            "junction": "-1",
        },
    )

    # Add plan view with geometry
    plan_view = ET.SubElement(road, "planView")
    s = 0.0  # Cumulative road length

    for i in range(len(centerline) - 1):
        x, y, hdg = centerline[i]
        length = segment_lengths[i]
        geometry = ET.SubElement(
            plan_view,
            "geometry",
            {
                "s": f"{s:.5f}",
                "x": f"{x:.5f}",
                "y": f"{y:.5f}",
                "hdg": f"{hdg:.5f}",
                "length": f"{length:.5f}",
            },
        )
        ET.SubElement(geometry, "line")
        s += length

    # Add lanes
    lanes = ET.SubElement(road, "lanes")
    lane_section = ET.SubElement(lanes, "laneSection", {"s": "0.0"})

    left_element = ET.SubElement(lane_section, "left")
    left_lane = ET.SubElement(
        left_element, "lane", {"id": "1", "type": "driving", "level": "false"}
    )
    ET.SubElement(
        left_lane,
        "roadMark",
        {
            "sOffset": "0.0",
            "type": "solid",
            "color": "standard",
            "width": "0.15",
            "laneChange": "none",
        },
    )
    ET.SubElement(
        left_lane,
        "width",
        {"sOffset": "0.0", "a": str(lane_width), "b": "0.0", "c": "0.0", "d": "0.0"},
    )

    center = ET.SubElement(lane_section, "center")
    ET.SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})

    # right_element = ET.SubElement(lane_section, "right")
    # right_lane = ET.SubElement(right_element, "lane", {
    #     "id": "-1",
    #     "type": "driving",
    #     "level": "false"
    # })

    # ET.SubElement(right_lane, "roadMark", {
    #     "sOffset":"0.0",
    #     "type":"solid",
    #     "color":"standard",
    #     "width":"0.15",
    #     "laneChange":"none"
    # })
    # ET.SubElement(right_lane, "width", {
    #     "sOffset": "0.0",
    #     "a": str(lane_width),
    #     "b": "0.0",
    #     "c": "0.0",
    #     "d": "0.0"
    # })

    # Write to file
    tree = ET.ElementTree(opendrive)
    with open(file_name, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print(f"OpenDRIVE file generated: {file_name}")


def trackgen():
    line_file = TRACK_LINE  # Replace with your file name

    # Read the .line file
    centerline = read_line_file(line_file)

    # Calculate segment lengths
    segment_lengths = calculate_segment_lengths(centerline)

    # Create the OpenDRIVE file using the centerline
    create_xodr_file(centerline, segment_lengths)


try:

    #trackgen()
    client = carla.Client(KFIR, CARLA_SERVER_PORT)
    client.set_timeout(10.0)
    print("Connected to carla: ", client.get_server_version())
    print(f"Loading track: {TRACK_LINE}")

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

    # Visualize waypoints
    waypoints = map.generate_waypoints(1)
    for waypoint in waypoints:
        world.debug.draw_point(
            waypoint.transform.location,
            size=0.1,
            color=carla.Color(0, 0, 255),
            life_time=1000.0,
        )

    blueprint_library = world.get_blueprint_library()
    # vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_points = map.get_spawn_points()
    # transform = carla.Transform(
    #     carla.Location(x=-5, y=-2, z=2), carla.Rotation(yaw=180)
    # )
    vehicles = []

    # vehicle = world.spawn_actor(vehicle_bp, transform)

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
    # Get the Traffic Manager
    # traffic_manager = client.get_trafficmanager()
    # traffic_manager.auto_lane_change(vehicle, True)
    # traffic_manager.force_lane_change(vehicle, False)
    # traffic_manager.ignore_vehicles_percentage(vehicle, 5.0)
    # traffic_manager.global_percentage_speed_difference(-100.0)

    # Perform ticks and check bounds
    while True:
        world.tick()  # Step simulation
        for vehicle in vehicles:
            location = vehicle.get_location()
            # draw_vehicle_bounding_box(world, vehicle, life_time=0.05)
            if is_out_of_bounds(location, bounds):
                print(f"Vehicle {vehicle.id} is out of bounds at {location}.")
                # vehicle.set_autopilot(False)  # Disable autopilot for safety
                # vehicles.remove(vehicle)  # Remove the vehicle from the list
                # vehicle.destroy()  # Destroy the out-of-bounds vehicle
        sleep(0.05)  # Adjust tick frequency as needed

except Exception as e:
    for vehicle in vehicles:
        if vehicle is not None:
            vehicle.destroy()
    print("error: ", e)

finally:
    world = client.reload_world()  # for cleaning the world
    print("World cleaned")
    sleep(0.5)
