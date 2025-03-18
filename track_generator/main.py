from utils import *
from track_generator import TrackGenerator
import os


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
        except Exception as e:
            print(f"Error encountered: {e}. Retrying...")




def main(args=None):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_folder, "config.yaml")
    generated_tracks_path = os.path.join(current_folder, "generated_tracks")
    config = parse_config_with_mode(config_path)
    print(config)
    generate_tracks(config, generated_tracks_path)


main()
