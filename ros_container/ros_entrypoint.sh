#!/bin/bash
set -e

source /opt/ros/foxy/setup.bash
source /ros2_ws/install/setup.bash

# exec "$@"
roslaunch carla_ros_bridge carla_ros_bridge.launch host:=192.168.1.50 port:=2000
