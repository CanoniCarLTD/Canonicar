# Use the official ROS Foxy base image
FROM osrf/ros:foxy-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y python3-pip

RUN sudo apt install python3-colcon-common-extensions 

WORKDIR /ros_bridge
# Copy your local ROS 2 package into the container
COPY ./src ./src

# RUN rosdep update && rosdep install -i --from-path src --rosdistro foxy -y

RUN rosdep update && rosdep install --from-paths src --ignore-src -r
# Upgrade pip and install Python dependencies
WORKDIR /ros_bridge
# Source ROS 2 setup and build the workspace
RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy pygame
RUN python3 -m pip install 'carla==0.9.13'
RUN apt-get install -y ros-foxy-derived-object-msgs ros-foxy-std-msgs
RUN source /opt/ros/foxy/setup.bash && colcon build 

# RUN sudo apt update
# # RUN sudo apt upgrade
# RUN sudo apt install ros-foxy-desktop python3-argcomplete
# RUN sudo apt install ros-dev-tools


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
