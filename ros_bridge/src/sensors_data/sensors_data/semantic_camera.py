from sensor_msgs.msg import Image
from std_msgs.msg import Header
import carla

def carla_semantic_image_to_ros_image(carla_image, header: Header) -> Image:
    """
    Convert a carla.Image (semantic segmentation) to a sensor_msgs/Image (bgr8 encoding),
    using the CityScapesPalette color conversion.
    """
    # Convert to CityScapesPalette using CARLA's ColorConverter
    carla_image.convert(carla.ColorConverter.CityScapesPalette)

    img_msg = Image()
    img_msg.header = header
    img_msg.height = carla_image.height
    img_msg.width = carla_image.width
    img_msg.encoding = "bgr8"  # 8-bit BGR image for ROS
    img_msg.step = 3 * carla_image.width
    img_data = bytes(carla_image.raw_data)
    # Remove alpha channel (BGRA -> BGR)
    img_msg.data = b"".join([
        img_data[i:i+3] for i in range(0, len(img_data), 4)
    ])
    return img_msg
