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

### 1. Run the Load Map Script

Execute the `runLoadMap.py` script to load the map. Make sure to adjust the host IP address in the script before running.

```bash
cd track_generator
python runLoadMap.py
```

**Note:** Open `runLoadMap.py` in your preferred text editor and set the `CARLA_HOST` variable to your host machine's IP address.

### 2. Build the Docker Image

Build the Docker image using Docker Compose. This step will set up the necessary environment for the CARLA client.

```bash
docker compose up --build
```

### 3. Run the Docker Container Interactively

Once the Docker image is built, run the container in interactive mode to access the shell.

```bash
docker exec -it carla_client /bin/bash
```

### 4. Configure and Launch the CARLA Client Inside the Container

Inside the Docker container, perform the following steps:

#### a. Source ROS 2 Environment

Initialize the ROS 2 environment by sourcing the setup script.

```bash
source setup_ros.sh
```

#### b. Export CARLA Host IP

Set the `CARLA_HOST` environment variable to the IP address of your host machine.

```bash
export CARLA_HOST=your_host_ip_address
```

**Replace `your_host_ip_address` with the actual IP address of your host machine.**

#### c. Run the CARLA Client Node

Launch the CARLA client node using ROS 2.

```bash
ros2 run client_node spawn_vehicle_node
```

## Troubleshooting

- **Cannot Connect to CARLA Host:**
  - Ensure that the `CARLA_HOST` environment variable is correctly set to your host machine's IP address.
  - Verify that the CARLA server is running and accessible from the Docker container.

- **Docker Compose Issues:**
  - Make sure Docker and Docker Compose are properly installed and running.
  - Check for any errors during the `docker compose up --build` process and resolve them accordingly.

- **ROS 2 Errors:**
  - Ensure that ROS 2 is correctly sourced and all dependencies are installed.
  - Refer to the [ROS 2 Documentation](https://docs.ros.org/en/foxy/index.html) for detailed troubleshooting steps.


## License

This project is licensed under the [MIT License](LICENSE).
