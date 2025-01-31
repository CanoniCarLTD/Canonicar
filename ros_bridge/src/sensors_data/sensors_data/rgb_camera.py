from sensor_msgs.msg import Image
from std_msgs.msg import Header

def carla_image_to_ros_image(carla_image, header: Header) -> Image:
    """
    Convert a carla.Image (BGRA) to a sensor_msgs/Image (e.g. in 'bgra8' or 'rgb8').
    By default, CARLA camera is in BGRA byte order.
    """
    img_msg = Image()
    img_msg.header = header
    img_msg.height = carla_image.height
    img_msg.width = carla_image.width
    img_msg.encoding = "bgra8" 
    img_msg.step = 4 * carla_image.width 

    img_msg.data = bytes(carla_image.raw_data)

    return img_msg
