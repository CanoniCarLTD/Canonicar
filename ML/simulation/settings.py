"""
All the parameters used in the Simulation has been documented here.

Easily modifiable paramters with the quick access in this settings.py file \
    to achieve quick modifications especially during the training sessions.

Names of the parameters are self-explanatory therefore elimating the use of further comments.
"""

import dotenv
import os

dotenv.load_dotenv()


HOST = os.getenv("ETAI_IP")
PORT = 2000
TIMEOUT = 20.0

CAR_NAME = "model3"
# EPISODE_LENGTH = 120 # WHATS THE DIFFERENCE BETWEEN THIS AND THE ONE IN parameters.py ?
# NUMBER_OF_VEHICLES = 30 # IF WE DECIDE TO ADD MORE VEHICLES
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = True

RGB_CAMERA = "sensor.camera.rgb"
SSC_CAMERA = "sensor.camera.semantic_segmentation"

LIDAR = "sensor.lidar.ray_cast"  # ADDED BY ETAI, CHANGE IF NEEDED
