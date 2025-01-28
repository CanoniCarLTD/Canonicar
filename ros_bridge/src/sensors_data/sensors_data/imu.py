from sensor_msgs.msg import Imu
from std_msgs.msg import Header

def carla_imu_to_ros_imu(carla_imu_data, header: Header) -> Imu:
    """
    Convert a carla.IMUMeasurement to sensor_msgs/Imu.
    carla_imu_data.accelerometer, .gyroscope => Vector3D
    """
    imu_msg = Imu()
    imu_msg.header = header


    imu_msg.linear_acceleration.x = float(carla_imu_data.accelerometer.x)
    imu_msg.linear_acceleration.y = float(carla_imu_data.accelerometer.y)
    imu_msg.linear_acceleration.z = float(carla_imu_data.accelerometer.z)

    # Fill angular_velocity
    imu_msg.angular_velocity.x = float(carla_imu_data.gyroscope.x)
    imu_msg.angular_velocity.y = float(carla_imu_data.gyroscope.y)
    imu_msg.angular_velocity.z = float(carla_imu_data.gyroscope.z)


    return imu_msg
