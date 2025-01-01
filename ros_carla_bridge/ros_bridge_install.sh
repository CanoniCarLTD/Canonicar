#!/bin/bash

# clone the ROS 2 bridge
mkdir -p ~/carla-ros-bridge 
cd ~/carla-ros-bridge
git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

source /opt/ros/foxy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r
colcon build

echo "Skipping installation of Python dependencies; assuming they're installed locally."
echo "activate your virtual python env with all the dependencies."

 source ./install/setup.bash

 echo "CARLA ROS Bridge setup complete!"

# running the container and use the versions installed in the host 
# docker run -it --rm \
#     --network host \
#     --name carla_ros_bridge_container \
#     -v ~/.local/lib/python3.8/site-packages:/root/.local/lib/python3.8/site-packages \
#     carla_ros_bridge:latest