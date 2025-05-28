#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    ###################
    # Launch Arguments
    ###################
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation clock'
    )
    
    ##################################
    # Group 1: Map cleaner
    ##################################
    
    map_cleaner_node = TimerAction(
        period=1.0,  
        actions=[
            LogInfo(msg="Starting map cleaner..."),
            Node(
                package='master_project2',
                executable='map_cleaner.py',
                name='map_cleaner',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                }],
                output='screen'
            )
        ]
    )
    
    ##################################
    # Group 2: FlowGrid
    ##################################
    
    flowgrid_node = TimerAction(
        period=2.0,  
        actions=[
            LogInfo(msg="Starting FlowGrid..."),
            Node(
                package='master_project2',
                executable='flowgrid.py',
                name='flow_grid',
                output='screen',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'num_directions': 8,
                    'use_fixed_resolution': True,
                    'fixed_resolution': 1.0,
                    'prediction_concentration': 5.0,
                    'max_propagation_distance': 12,
                    'distance_decay_factor': 0.95,
                    'observation_decay_rate': 0.95
                }]
            )
        ]
    )
    
    ##################################
    # Group 3: Waypoints
    ##################################
    
    waypoint_node = TimerAction(
        period=3.0,  
        actions=[
            LogInfo(msg="Starting waypoint graph publisher..."),
            Node(
                package='master_project2',
                executable='waypoints.py',
                name='waypoint_graph_publisher',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'base_update_frequency': 0.1
                }],
                output='screen'
            )
        ]
    )
    
    waypoint_updater_node = TimerAction(
        period=4.0,  
        actions=[
            LogInfo(msg="Starting waypoint cost updater..."),
            Node(
                package='master_project2',
                executable='waypoint_updater.py',
                name='waypoint_cost_updater',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'min_cost': 1.0,
                    'max_cost': 10.0,
                    'obstacle_cost': 1000.0,
                    'base_update_frequency': 0.1
                }],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        
        # Group 1: Map Cleaner
        map_cleaner_node,
        
        # Group 2: FlowGrid
        #flowgrid_node,
        
        # Group 3: Waypoints
        waypoint_node,
        waypoint_updater_node,
    ])