from utils import *
from track_generator import TrackGenerator
import os
import xml.etree.ElementTree as ET
import math


def generate_tracks(config: dict, generated_track_path):
    import os

    track_cfg = config["track"]
    gen_cfg = config["general"]

    # Read difficulty level from the config (default to "normal")
    diff = track_cfg.get("difficulty_level", "normal").lower()
    adj = track_cfg.get("difficulty", {}).get(diff, {})

    # Append difficulty as a subdirectory and ensure it exists
    output_path = os.path.join(generated_track_path, diff)
    os.makedirs(output_path, exist_ok=True)

    # Adjust base parameters for constructor
    n_points = track_cfg["n_points"] + adj.get("n_points", 0)
    n_regions = track_cfg["n_regions"] + adj.get("n_regions", 0)
    min_bound = track_cfg["min_bound"] + adj.get("min_bound", 0.0)
    max_bound = track_cfg["max_bound"] + adj.get("max_bound", 0.0)
    
    # Convert mode if it's a string, otherwise use as is.
    mode_val = track_cfg["mode"]
    mode = Mode[mode_val.upper()] if isinstance(mode_val, str) else mode_val

    for i in range(gen_cfg["n_tracks"]):
        try:
            track_name = f"{track_cfg['base_track_name']}{i}"
            track_gen = TrackGenerator(
                track_name,
                n_points,
                n_regions,
                min_bound,
                max_bound,
                mode,
                output_path,
                path_points_density_factor=track_cfg["path_points_density_factor"],
                plot_track=gen_cfg["plot_track"],
                lat_offset=track_cfg["lat_offset"],
                lon_offset=track_cfg["lon_offset"],
                visualise_voronoi=gen_cfg["visualise_voronoi"],
            )
            # Update extra attributes based on difficulty adjustments
            for attr in [
                "curvature_threshold",
                "track_width",
                "cone_spacing",
                "length_start_area",
                "misplacement_rate",
                "max_cone_spacing_offset",
                "max_cone_inward_offset",
                "max_cone_outward_offset",
            ]:
                if attr in adj:
                    setattr(track_gen, f"_{attr}", getattr(track_gen, f"_{attr}") + adj[attr])
            track_gen.create_track()
            xodr_path = os.path.join("ros_bridge\src\map_loader\map_loader",f"{track_name}.xodr")
            line_path = os.path.join(output_path,f"{track_name}.line")
            trackgen(line_path, xodr_path)
        except Exception as e:
            print(f"Error encountered: {e}. Retrying...")


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
    file_name,
    lane_width=10.5
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
    total_length = sum(segment_lengths) + segment_lengths[0]

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
    link = ET.SubElement(road, "link")
    ET.SubElement(
        link,
        "predecessor",
        {"elementType": "road", "elementId": "1", "contactPoint": "end"},
    )
    ET.SubElement(
        link,
        "successor",
        {"elementType": "road", "elementId": "1", "contactPoint": "start"},
    )

    # Add plan view with geometry
    plan_view = ET.SubElement(road, "planView")
    s = 0.0  # Cumulative road length

    # Add all geometry segments except the last closing one
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

    # Add the final geometry element that closes the loop
    x, y, hdg = centerline[-1]  # Use the last point from centerline
    close_length = segment_lengths[0]  # Distance back to the starting point
    geometry = ET.SubElement(
        plan_view,
        "geometry",
        {
            "s": f"{s:.5f}",
            "x": f"{x:.5f}",
            "y": f"{y:.5f}",
            "hdg": f"{hdg:.5f}",
            "length": f"{close_length:.5f}",
        },
    )
    ET.SubElement(geometry, "line")  # Don't forget to add the line element!

    # Add lanes
    lanes = ET.SubElement(road, "lanes")
    lane_section = ET.SubElement(lanes, "laneSection", {"s": "0.0"})

    left_element = ET.SubElement(lane_section, "left")
    left_lane = ET.SubElement(
        left_element, "lane", {"id": "-1", "type": "driving", "level": "false"}
    )
    left_link = ET.SubElement(left_lane, "link")
    ET.SubElement(left_link, "predecessor", {"id": "-1"})
    ET.SubElement(left_link, "successor", {"id": "-1"})
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

    # Write to file
    tree = ET.ElementTree(opendrive)
    with open(file_name, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print(f"OpenDRIVE file generated: {file_name}")

def trackgen(line_file, xodr_file):
    centerline = read_line_file(line_file)
    segment_lengths = calculate_segment_lengths(centerline)
    create_xodr_file(centerline, segment_lengths,xodr_file)



def main(args=None):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_folder, "config.yaml")
    generated_tracks_path = os.path.join(current_folder, "generated_tracks")
    config = parse_config_with_mode(config_path)
    print(config)
    generate_tracks(config, generated_tracks_path)


main()
