# Master Project 2 - Autonomous Robot Navigation with Human Interaction

This project implements an autonomous robot navigation system that can operate with single or multiple robots in environments with dynamic human obstacles. The system uses computer vision for human detection, dynamic path planning, predictive modeling for human movement patterns, and sophisticated multi-robot coordination.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Single Robot Mode](#single-robot-mode)
  - [Multi-Robot Mode](#multi-robot-mode)
- [Multi-Robot System](#multi-robot-system)
- [Project Structure](#project-structure)


## Overview

This ROS2-based project simulates differential drive robots named "Serena" equipped with:
- Dual Intel D435 cameras for stereo vision
- LiDAR sensor for mapping and navigation
- Custom navigation algorithms with human-aware path planning
- Real-time human detection and tracking using YOLO
- Multi-robot coordination and collision avoidance

### Supported Configurations
- **Single Robot**: Traditional autonomous navigation with human interaction
- **Multi-Robot**: Coordinated navigation for 1-5 robots with shared perception and collision avoidance

## Features

### Core Navigation Features
- **Autonomous Navigation**: Nav2-based navigation with custom path planning
- **Human Detection**: YOLO-based computer vision for detecting and tracking humans
- **Dynamic Mapping**: Real-time map updates based on sensor data
- **Human Simulation**: Realistic human behavior simulation in Gazebo
- **Predictive Modeling**: Flow grid system for predicting human movement
- **Path Planning**: A* algorithm with dynamic cost updates
- **Database Logging**: SQLite databases for data persistence

### Multi-Robot Features
- **Coordinated Navigation**: Up to 5 robots with collision avoidance
- **Shared Perception**: Aggregated human detection across all robots
- **Priority-Based Coordination**: Intelligent conflict resolution
- **Deadlock Detection**: Automatic deadlock detection and resolution
- **Safety Zones**: Dynamic safety zones around each robot
- **Backup Maneuvers**: Automatic backup when robots get stuck
- **Global Flow Grid**: Combined human movement prediction from all robots

## System Requirements

- **OS**: Ubuntu 22.04 LTS (recommended)
- **ROS2**: Humble Hawksbill or newer
- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended (32GB for multi-robot with 5 robots)
- **GPU**: NVIDIA GPU recommended for YOLO inference

## Installation

### 1. Dependencies Installation

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
#Download intelRealSense
https://github.com/IntelRealSense/realsense-ros
```

### 2. YOLO Model Setup

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

### 3. Multi-Robot Configuration

Configure multi-robot parameters in `config/multi_robot_nav2_params.yaml` for each robot namespace.

### 4. Human Simulation Configuration

Configure human paths in `config/paths.json` and human waypoints in `config/humans_waypoint_graph.json`.

## Usage

### Single Robot Mode

#### Basic Single Robot Simulation

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

### Multi-Robot Mode

#### Basic Multi-Robot Simulation

1. **Launch multi-robot simulation** (3 robots by default):
```bash
ros2 launch master_project2 multibot_launch_sim.launch.py num_robots:=3 gui:=true
```

2. **Launch the complete multi-robot system**:
```bash
ros2 launch master_project2 multibot_master_launch.launch.py num_robots:=3
```

#### Coordinated Path Planning

Launch coordinated path planning with collision avoidance:
```bash
ros2 launch master_project2 coordinated_path_planning.launch.py \
    num_robots:=3 \
    safety_distance:=1.5 \
    wait_time:=5.0
```

#### Multi-Robot Human Detection

Start human detection across all robots:
```bash
ros2 launch master_project2 multibot_yolo_visualizer_launch.py num_robots:=3
```

Launch flow grid aggregation:
```bash
ros2 launch master_project2 multibot_flowgrid_launch.py num_robots:=3
```

#### Individual Robot Navigation

Send navigation commands to specific robots:
```bash
# Navigate robot_0 to waypoint 5
ros2 service call /robot_0/navigate_to_waypoint master_project2/srv/NavigateToWaypoint "{waypoint_id: '5', start_waypoint_id: ''}"

# Navigate robot_1 through multiple waypoints
ros2 service call /robot_1/navigate_to_waypoint master_project2/srv/NavigateToWaypoint "{waypoint_id: '1,3,7,9', start_waypoint_id: ''}"
```

## Multi-Robot System

### Architecture

The multi-robot system consists of several key components:

#### 1. Robot Coordination (`multibot_coordinator.py`)
- **Priority-based navigation**: Robots have assigned priorities to resolve conflicts
- **Safety zones**: Dynamic safety zones around each robot (configurable radius)
- **Deadlock detection**: Automatic detection of stuck robots and resolution strategies
- **Backup maneuvers**: Robots can reverse to resolve deadlocks
- **Real-time monitoring**: Continuous monitoring of robot positions and status

#### 2. Shared Perception
- **Human Detection Aggregator**: Combines detections from all robot cameras
- **Flow Grid**: Unified prediction grid based on observations from all robots
- **Duplicate Filtering**: Removes redundant detections between robots

#### 3. Distributed Path Planning
- **Individual Planners**: Each robot runs its own path planner
- **Shared Cost Updates**: Waypoint costs updated based on global observations
- **Coordination Messages**: Robots communicate their intentions and status

### Configuration Parameters

#### Coordination Parameters
```bash
# Safety distance between robots (meters)
safety_distance:=1.0

# Time robots wait when in conflict (seconds)
wait_time:=10.0

# Distance at which robots can resume after waiting
resume_distance:=1.5

# Timeout for deadlock detection (seconds)
deadlock_timeout:=60.0

# Backup distance for deadlock resolution (meters)
backup_distance:=3.0
```

#### Detection Parameters
```bash
# Threshold for duplicate human detection filtering (meters)
duplicate_threshold:=0.5

# Update interval for human detection aggregation (seconds)
republish_interval:=0.1
```

### Supported Robot Counts

The system supports 1-5 robots with predefined starting positions:
- **robot_0**: (1.0, 0.0)
- **robot_1**: (3.0, 3.0)
- **robot_2**: (4.0, 0.0)
- **robot_3**: (-2.0, 0.0)
- **robot_4**: (0.0, -2.0)

### Multi-Robot Testing

Test individual robots:
```bash
# Test robot_0 navigation to waypoints 1,2,3
ros2 run master_project2 multibot_test.py robot_0 1,2,3

# Test robot_1 navigation to waypoint 5
ros2 run master_project2 multibot_test.py robot_1 5
```

### Advanced Usage

#### Custom Number of Robots
```bash
ros2 launch master_project2 multibot_master_launch.launch.py num_robots:=5
```

#### Headless Mode (No GUI)
```bash
ros2 launch master_project2 multibot_launch_sim.launch.py num_robots:=3 gui:=false
```

#### Monitor System Status
```bash
# Check coordination status
ros2 topic echo /coordinator_visualization

# Monitor individual robot status
ros2 topic echo /robot_0/coordination_status
ros2 topic echo /robot_1/path_execution_status

# View aggregated human detections
ros2 topic echo /raw_human_detections

# Monitor flow grid updates
ros2 topic echo /flow_grid_update_complete
```

## Project Structure

```
master_project2/
├── config/                           # Configuration files
│   ├── nav2_params.yaml             # Single robot navigation parameters
│   ├── multi_robot_nav2_params.yaml # Multi-robot navigation parameters
│   ├── robot_waypoint_graph.json    # Waypoint graph for robots
│   ├── humans_waypoint_graph.json   # Waypoint graph for human simulation
│   └── yolo/best.pt                 # YOLO weights
├── description/                      # Robot URDF/Xacro files
│   ├── serena.urdf.xacro            # Single robot description
│   └── multibot_serena.urdf.xacro   # Multi-robot compatible description
├── launch/                          # Launch files
│   ├── main.launch.py               # Single robot simulation
│   ├── master_launch.py             # Single robot systems
│   ├── start_mission.launch.py      # Single robot missions
│   ├── multibot_launch_sim.launch.py        # Multi-robot simulation
│   ├── multibot_master_launch.py            # Multi-robot master launcher
│   ├── multibot_navigation.launch.py        # Multi-robot navigation
│   ├── improved_multi_robot_localization.launch.py # Multi-robot AMCL
│   ├── coordinated_path_planning.launch.py  # Coordinated planning
│   ├── multibot_flowgrid_launch.py          # Multi-robot flow grid
│   └── multibot_yolo_visualizer_launch.py   # Multi-robot detection
├── maps/                            # Map files
├── scripts/                         # Python scripts
│   ├── Astar_pathplanner.py         # Single robot A* planner
│   ├── multibot_coordinator.py      # Multi-robot coordinator
│   ├── multibot_path_planner.py     # Multi-robot path planner
│   ├── multibot_flowgrid.py         # Multi-robot flow grid
│   ├── multibot_human_detection_aggregator.py # Detection aggregator
│   ├── multibot_yolo_visualizer.py  # Multi-robot YOLO detection
│   ├── multibot_waypoint_updater.py # Multi-robot waypoint cost updates
│   ├── multibot_test.py             # Multi-robot testing
│   ├── database_handler.py          # Data management
│   ├── map_cleaner.py               # Map processing
│   ├── spawn_humans.py              # Human simulation
│   └── waypoints.py                 # Waypoint management
├── srv/                             # Custom service definitions
└── worlds/                          # Gazebo world files
```


