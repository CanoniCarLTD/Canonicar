Here's a table summarizing the data passed to the PPO agent from the `DataCollector` class:

| **Data Segment**            | **ROS Topic**                            | **Format**                     | **Description**                                                                                        | **Size (Cells)** |
| --------------------------- | ---------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ | ---------------- |
| **RGB (Image)**             | `/carla/ego_vehicle/rgb_front/image_raw` | 1D array (`np.array`, float32) | 20 high-level features extracted using MobileNetV2, representing visual features like edges, textures. | 20               |
| **LiDAR**                   | `/carla/ego_vehicle/lidar/points`        | 1D array (`list`, float32)     | 15 features: mean height, density, and 13 selected 3D points from the point cloud data.                | 15               |
| **IMU (Acceleration)**      | `/carla/ego_vehicle/imu`                 | List (`float32`)               | Linear acceleration in x, y, z directions (`[ax, ay, az]`).                                            | 3                |
| **IMU (Angular Vel.)**      | `/carla/ego_vehicle/imu`                 | List (`float32`)               | Angular velocity in x, y, z directions (`[wx, wy, wz]`).                                               | 3                |
| **GNSS (Position)**         | `/carla/ego_vehicle/gnss`                | List (`float32`)               | GPS position: latitude, longitude, and altitude (`[lat, lon, alt]`).                                   | 3                |
| **GNSS (Derived Features)** | `/carla/ego_vehicle/gnss`                | List (`float32`)               | Velocity and heading direction derived from GNSS data (`[velocity, heading]`).                         | 2                |

### **Total Size**: 50 Cells

This structure ensures that the PPO agent receives a consistent, compact, and meaningful representation of the environment for decision-making.

### **Breakdown by Segment**

1. **RGB (Image)**:

   - Extracted using a pretrained MobileNetV2 model.
   - Captures critical visual features like object shapes, edges, and textures.

2. **LiDAR**:

   - Includes spatial features like point density, mean height, and raw 3D points.
   - Useful for detecting obstacles and spatial navigation.

3. **IMU**:

   - Provides the vehicle’s dynamic state, including acceleration and rotational velocity.

4. **GNSS**:
   - Offers global positioning data for localization and motion prediction.
   - Derived features (velocity, heading) add context about the vehicle’s movement.

### **Why This Structure Works for PPO**

- **Consistency**: A fixed-size state vector ensures compatibility with the neural network input.
- **Comprehensive**: Covers visual, spatial, and dynamic aspects of the environment.
- **Efficiency**: Reduces raw data to meaningful features, minimizing computational overhead for PPO.

Let me know if you’d like further refinements or adjustments to the data pipeline!
