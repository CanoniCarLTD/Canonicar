# Canonicar

# CARLA ROS 2 Client Setup

This repository contains the setup and instructions to run the CARLA simulator with ROS 2 integration using Docker. Follow the steps below to get your environment up and running.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Setup](#running-the-setup)
  - [1. Run the Load Map Script](#1-run-the-load-map-script)
  - [2. Build the Docker Image](#2-build-the-docker-image)
  - [3. Run the Docker Container Interactively](#3-run-the-docker-container-interactively)
  - [4. Configure and Launch the CARLA Client Inside the Container](#4-configure-and-launch-the-carla-client-inside-the-container)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Python 3.8+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [ROS 2](https://docs.ros.org/en/foxy/Installation.html) (Ensure compatibility with your setup)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/CanoniCarLTD/Canonicar.git
   cd Canonicar
   ```



## Running the Setup

### 1. Build the Docker Image

Build the Docker image using Docker Compose. This step will set up the necessary environment for the CARLA client.
If you are using a simulator as carla server which means the simulator is not running on your local machine, you have to activate the `carla_server` at the docker compose.

```bash
docker compose up --build
```

### 2. Run the Docker Container Interactively

Once the Docker image is built, run the container from a second terminal in interactive mode to access the shell.

```bash
docker exec -it carla_client /bin/bash
```

### 3. Configure and Launch the CARLA Client Inside the Container

Inside the Docker container, perform the following steps:

#### a. Source ROS 2 Environment

Initialize the ROS 2 environment by sourcing.

```bash
source install/setup.bash
```

#### b. Run the CARLA Pipeline

Launch the CARLA pipeline using ROS 2.

```bash
ros2 launch ros_bridge_launch system_launch.py host:=your_host_ip_addres
```
**Replace `your_host_ip_address` with the actual IP address of your host machine.**

## Troubleshooting

- **Cannot Connect to CARLA Host:**
  - Verify that the CARLA server is running and accessible from the Docker container.

- **Docker Compose Issues:**
  - Make sure Docker and Docker Compose are properly installed and running.
  - Check for any errors during the `docker compose up --build` process and resolve them accordingly.

- **ROS 2 Errors:**
  - Ensure that ROS 2 is correctly sourced and all dependencies are installed.
  - Refer to the [ROS 2 Documentation](https://docs.ros.org/en/foxy/index.html) for detailed troubleshooting steps.

- **Adding nodes:**
  - In case you want to add new node, you have to create new pacgkge with the following ROS commands:
  ```bash
  ros2 pkg create --build-type ament_python --license Apache-2.0 <package_name>
  ```
  - if you are making changes at the nodes, and the containers are up and running, you have to do `colcon build` inside the container itself before launching the nodes.

## Watching live training metrics progress (STILL NEEDS TESTING)

```bash
tensorboard --logdir tensorboard_logs
```

## License

This project is licensed under the [MIT License](LICENSE).
