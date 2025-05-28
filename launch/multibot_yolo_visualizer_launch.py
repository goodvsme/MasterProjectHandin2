#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    use_sim_time = LaunchConfiguration('use_sim_time')
    num_robots = LaunchConfiguration('num_robots')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
        
    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to launch YOLO visualizer for')
    
    def launch_yolo_nodes(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        
        actions = []
        actions.append(LogInfo(msg=f"Launching multibot YOLO visualizers for {num_robots_value} robots..."))
        
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            
            actions.append(LogInfo(msg=f"Launching YOLO visualizer for {robot_name}"))
            
            actions.append(Node(
                package='master_project2',
                executable='multibot_yolo_visualizer.py',  
                namespace=robot_name,
                name='yolo_visualizer',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'robot_namespace': robot_name,
                    'use_map_frame': True,
                    'human_radius': 0.3,
                    'visualize': True,
                    'duplicate_threshold': 0.5,
                    'distance_threshold': 1.0,
                    'min_movement': 0.1,
                    'multi_robot': True  
                }],
                remappings=[
                    ('/tf', '/tf'),
                    ('/tf_static', '/tf_static'),
                ]
            ))
        
        actions.append(LogInfo(msg="Verifying human detection topics..."))
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            actions.append(LogInfo(msg=f"Expected topic: /{robot_name}/human_detections"))
        
        return actions
    
    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_num_robots_cmd)
    
    ld.add_action(LogInfo(msg="Initializing multibot YOLO visualizer system..."))
    
    ld.add_action(OpaqueFunction(function=launch_yolo_nodes))
    
    return ld