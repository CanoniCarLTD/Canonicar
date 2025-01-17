# Use the official ROS Foxy base image
FROM osrf/ros:foxy-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

# Install necessary dependencies
RUN apt update && apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-foxy-derived-object-msgs

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy pygame
RUN python3 -m pip install \
rclpy \
ros2cli \
ros2pkg \
ros2run \
ros2topic

RUN rosdep update && rosdep install --from-paths src --ignore-src -r -y
# Set up workspace
WORKDIR /ros_bridge

# Copy your local ROS 2 package into the container
COPY ./src ./src

# Install ROS dependencies

# Source ROS 2 setup and build the workspace
RUN source /opt/ros/foxy/setup.bash && colcon build

# Expose necessary ports (if required by your ROS application)
# EXPOSE 2000-2002

# Set default entrypoint to bash (useful for debugging)
CMD ["/bin/bash"]

# Instructions for running:
# 1. Build the Docker image:
#    docker build -t ros_hello_world .
# 2. Run the container interactively:
#    docker run -it ros_hello_world /bin/bash
# 3. Inside the container:
#    - Source ROS 2: source /opt/ros/foxy/setup.bash
#    - Source workspace: source install/setup.bash
#    - Run publisher: ros2 run hello_world publisher
#    - Run subscriber: ros2 run hello_world subscriber
