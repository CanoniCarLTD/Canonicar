from utils import *
from track_generator import TrackGenerator
import os

def generate_tracks(config:dict, generated_track_path):
    generated_tracks_location = 'generated_tracks'
    track_config = config['track']
    general_config = config['general']
    # Create tracks
    i = 0
    while i < general_config['n_tracks']:
       try:
            track_name = track_config['base_track_name'] + str(i)
            track_gen = TrackGenerator(track_name,
                                    track_config['n_points'],
                                    track_config['n_regions'],
                                    track_config['min_bound'],
                                    track_config['max_bound'], 
                                    track_config['mode'], 
                                    generated_track_path,
                                    path_points_density_factor=track_config['path_points_density_factor'],
                                    plot_track=general_config['plot_track'],
                                    lat_offset=track_config['lat_offset'], 
                                    lon_offset=track_config['lon_offset'],
                                    visualise_voronoi=general_config['visualise_voronoi'])
            track_gen.create_track()
            i += 1 
       except Exception as e:
            print(f"Error encountered: {e}. Retrying...") 

def main(args=None):
 current_folder = os.path.dirname(os.path.abspath(__file__))
 config_path = os.path.join(current_folder,'config.yaml')
 generated_tracks_path = os.path.join(current_folder,'generated_tracks')
 config = parse_config_with_mode(config_path)
 print (config)
 generate_tracks(config, generated_tracks_path)

main()