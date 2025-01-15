# First stage sets up the base, ie everything except the active dev repo
FROM osrf/ros:foxy-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

# Install dependencies and tools
RUN apt update && apt install -y python3-pip

RUN sudo apt install python3-colcon-common-extensions

# Clone the carla-ros-bridge repository
RUN git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

# Install dependencies via rosdep
RUN rosdep update && rosdep install --from-paths src --ignore-src -r


# Change directory to src/ros-bridge and build the workspace
WORKDIR /src/ros-bridge
RUN source /opt/ros/foxy/setup.bash && colcon build 
RUN pip install -r requirements.txt  
RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy pygame
RUN python3 -m pip install 'carla==0.9.13'
RUN apt-get install -y ros-foxy-derived-object-msgs

# build image: docker build -t ros_carla_bridge .
# run docker image: docker run -it ros_carla_bridge /bin/bash
# new terminal: docker exec -it <container_name> /bin/bash 

# How to run the docker container with display (X11 forwarding) enabled:
# docker run -it `
# >>   -e DISPLAY=host.docker.internal:0 `
# >>   -v /tmp/.X11-unix:/tmp/.X11-unix `
# >>   ros_carla_bridge /bin/bash


# launch ros2 : source /opt/ros/foxy/setup.bash
# launch the colcon build workspace : source install/setup.bash
# launch the carla-ros-bridge : ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=193.106.55.44 port:=2000 synchronous_mode:=False
# launch ros-spawn-object: ros2 launch carla_spawn_objects carla_spawn_objects.launch.py objects_definition_file:=carla_spawn_objects/config/objects.json