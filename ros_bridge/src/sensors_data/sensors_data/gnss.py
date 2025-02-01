from sensor_msgs.msg import NavSatFix ,NavSatStatus 
from std_msgs.msg import Header

def carla_gnss_to_ros_navsatfix(carla_gnss_data, header: Header) -> NavSatFix:
    """
    Convert a carla.GnssMeasurement to sensor_msgs/NavSatFix.
    carla_gnss_data.latitude, .longitude, .altitude
    """
    navsat_msg = NavSatFix()
    navsat_msg.header = header
    navsat_msg.latitude = float(carla_gnss_data.latitude)
    navsat_msg.longitude = float(carla_gnss_data.longitude)
    navsat_msg.altitude = float(carla_gnss_data.altitude)

    navsat_msg.status.service = NavSatStatus.SERVICE_GPS
    navsat_msg.status.status = NavSatStatus.STATUS_FIX
    navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    return navsat_msg
