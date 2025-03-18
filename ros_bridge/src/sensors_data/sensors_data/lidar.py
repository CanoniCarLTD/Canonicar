import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def carla_lidar_to_ros_pointcloud2(carla_lidar_data, header: Header) -> PointCloud2:
    """
    Convert a carla.LidarMeasurement to sensor_msgs/PointCloud2.
    Each point in carla_lidar_data is [x, y, z, intensity].
    """
    pc_msg = PointCloud2()
    pc_msg.header = header
    pc_msg.height = 1  
    pc_msg.width = len(carla_lidar_data)
    pc_msg.is_bigendian = False
    pc_msg.is_dense = False

    pc_msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    pc_msg.point_step = 16
    pc_msg.row_step = pc_msg.point_step * pc_msg.width

    points = np.array([
        [float(d.point.x), float(d.point.y), float(d.point.z), float(d.intensity)]
        for d in carla_lidar_data
    ], dtype=np.float32)

    pc_msg.data = points.tobytes()

    return pc_msg
