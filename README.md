# Master Project 2 - Autonomous Robot Navigation with Human Interaction

This project implements an autonomous robot navigation system that can navigate in environments with dynamic human obstacles. The system uses computer vision for human detection, dynamic path planning, and predictive modeling for human movement patterns.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)


## Overview

This ROS2-based project simulates a differential drive robot named "Serena" equipped with:
- Dual Intel D435 cameras for stereo vision
- LiDAR sensor for mapping and navigation
- Custom navigation algorithms with human-aware path planning
- Real-time human detection and tracking using YOLO

## Features

- **Autonomous Navigation**: Nav2-based navigation with custom path planning
- **Human Detection**: YOLO-based computer vision for detecting and tracking humans
- **Dynamic Mapping**: Real-time map updates based on sensor data
- **Human Simulation**: Realistic human behavior simulation in Gazebo
- **Predictive Modeling**: Flow grid system for predicting human movement
- **Path Planning**: A* algorithm with dynamic cost updates
- **Database Logging**: SQLite databases for data persistence

## System Requirements

- **OS**: Ubuntu 22.04 LTS (recommended)
- **ROS2**: Humble Hawksbill or newer
- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU recommended for YOLO inference

## Installation

### 1. ROS2 Installation

Install ROS2 Humble following the [official guide](https://docs.ros.org/en/humble/Installation.html):

```bash
# Add ROS2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS2
sudo apt update
sudo apt install ros-humble-desktop-full
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Dependencies Installation

Install required system packages:

```bash
# ROS2 packages
sudo apt install \
    ros-humble-nav2-bringup \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-tools \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-message-filters

# Python packages
pip3 install \
    ultralytics \
    opencv-python \
    numpy \
    scipy \
    pillow \
    matplotlib \
    pyyaml

# Additional system packages
sudo apt install \
    sqlite3 \
    python3-sqlite3 \
    python3-opencv \
    python3-pil
```

### 3. Workspace Setup

Create and build the ROS2 workspace:

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone the project (replace with actual repository URL)
git clone <repository-url> master_project2

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select master_project2
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4. YOLO Model Setup

Download the YOLO weights file:

```bash
# Create config directory for YOLO
mkdir -p ~/ros2_ws/src/master_project2/config/yolo

# Download pre-trained YOLO weights (you may need to train your own model)
# Place the weights file as: ~/ros2_ws/src/master_project2/config/yolo/best.pt
```

## Configuration

### 1. Map Configuration

Ensure you have map files in the `maps/` directory:
- `map.yaml` and `map.pgm` - Original map
- Additional map variants will be generated automatically

### 2. Waypoint Configuration

Configure waypoints in `config/robot_waypoint_graph.json`:

```json
{
  "metadata": {
    "name": "robot_waypoints",
    "description": "Waypoint graph for robot navigation"
  },
  "nodes": [
    {
      "id": 0,
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "cost": 1.0
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "distance": 1.0
    }
  ]
}
```

### 3. Human Simulation Configuration

Configure human paths in `config/paths.json` and human waypoints in `config/humans_waypoint_graph.json`.

## Usage

### Basic Simulation

1. **Launch the main simulation**:
```bash
ros2 launch master_project2 main.launch.py
```

2. **Launch additional systems** (in a new terminal):
```bash
ros2 launch master_project2 master_launch.py
```

3. **Start a mission** (in a new terminal):
```bash
ros2 launch master_project2 start_mission.launch.py target_waypoints:="1,2,3"
```

### Advanced Usage

#### Manual Navigation

Send navigation goals manually:
```bash
ros2 service call /navigate_to_waypoint master_project2/srv/NavigateToWaypoint "{waypoint_id: '5', start_waypoint_id: ''}"
```

#### Monitor System Status

Check various system topics:
```bash
# Human detections
ros2 topic echo /raw_human_detections

# Path execution status
ros2 topic echo /path_execution_status

# Waypoint costs
ros2 topic echo /waypoint_costs
```

#### Database Management

Reset databases if needed:
```bash
ros2 service call /reset_database std_srvs/srv/Trigger
```

### Launch Parameters

Common launch parameters:

- `gui:=true/false` - Run Gazebo with/without GUI
- `use_sim_time:=true/false` - Use simulation time
- `target_waypoints:="1,2,3"` - Specify waypoint sequence for missions
- `human_count:=100` - Number of humans to spawn
- `human_speed:=0.5` - Speed of simulated humans
- `random_seed:=666` - Seed for reproducible simulations

## Project Structure

```
master_project2/
├── config/                    # Configuration files
│   ├── nav2_params.yaml      # Navigation parameters
│   ├── robot_waypoint_graph.json
│   ├── humans_waypoint_graph.json
│   └── yolo/best.pt          # YOLO weights
├── description/              # Robot URDF/Xacro files
│   └── serena.urdf.xacro     # Main robot description
├── launch/                   # Launch files
│   ├── main.launch.py        # Main simulation
│   ├── master_launch.py      # Additional systems
│   └── start_mission.launch.py # Mission execution
├── maps/                     # Map files
├── scripts/                  # Python scripts
│   ├── Astar_pathplanner.py  # A* path planning
│   ├── database_handler.py   # Data management
│   ├── flowgrid.py          # Human flow prediction
│   ├── map_cleaner.py       # Map processing
│   ├── spawn_humans.py      # Human simulation
│   ├── yolo_visualizer.py   # Human detection
│   └── waypoints.py         # Waypoint management
├── srv/                     # Custom service definitions
└── worlds/                  # Gazebo world files
```

