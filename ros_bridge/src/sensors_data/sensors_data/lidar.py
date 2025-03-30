import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import time

def carla_lidar_to_ros_pointcloud2(carla_lidar_data, header: Header, filter_ground=True) -> PointCloud2:
    """
    Convert a carla.LidarMeasurement to sensor_msgs/PointCloud2.
    """
    height_threshold = -1.0  

    points = np.array([
        [float(d.point.x), float(d.point.y), float(d.point.z), float(d.intensity)]
        for d in carla_lidar_data
    ], dtype=np.float32)

    if filter_ground and len(points) > 10:
        points = points[points[:, 2] > height_threshold] 
    
    pc_msg = PointCloud2()
    pc_msg.header = header
    pc_msg.height = 1  
    pc_msg.width = len(points)
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
    pc_msg.data = points.tobytes()

    return pc_msg