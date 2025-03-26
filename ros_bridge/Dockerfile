# Use the official ROS Foxy base image
FROM osrf/ros:foxy-desktop-focal AS base

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y python3-pip

RUN sudo apt install python3-colcon-common-extensions 

WORKDIR /ros_bridge

RUN mkdir -p src

VOLUME [ "/ros_bridge/src" ]

COPY setup_ros.sh /ros_bridge/

# Initialize ROS workspace

# Upgrade pip and install Python dependencies
# WORKDIR /ros_bridge
# Source ROS 2 setup and build the workspace
RUN python3 -m pip install --upgrade pip && python3 -m pip install numpy pygame python-dotenv torch torchvision

RUN  python3 -m pip install tensorboard

RUN rm -rf /usr/lib/python3/dist-packages/PyYAML-5.3.1.egg-info && rm -rf /usr/lib/python3/dist-packages/yaml && python3 -m pip install pyyaml open3d

RUN python3 -m pip install 'carla==0.9.13'
RUN apt-get install -y ros-foxy-derived-object-msgs ros-foxy-std-msgs ros-foxy-carla-msgs
RUN source /opt/ros/foxy/setup.bash


# Add the fix for typing-extensions without changing previous layers
RUN python3 -m pip install --upgrade typing-extensions>=4.8.0
RUN pip3 install psutil pymongo python-dotenv mongoengine

RUN pip install nvidia-ml-py3

# Set default entrypoint to bash (useful for debugging)
CMD ["/bin/bash"]

# Instructions for running:
# 1. Run the runLoadMap file from track_generator folder:
#    python runLoadMap.py (Adjust the host IP in the file)
# 2. Build the Docker image:
#    docker compose up --build
# 3. Run the container interactively:
#    docker exec -it carla_client /bin/bash
# 4. Inside the container:
#    - Source ROS 2: source setup_ros.sh
#    - Export environment IP: export CARLA_HOST="" (IP of the host machine)
#    - Run the carla_client: ros2 run client_node spawn_vehicle_node
