#!/usr/bin/env python3
#main.launch.py
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
    GroupAction,
    LogInfo
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    world_file = os.path.join(pkg_share, 'worlds', 'bilka_odense.world')
    rviz_config = os.path.join(pkg_share, 'config', 'main.rviz')
    map_file = os.path.join(pkg_share, 'maps', 'map.yaml')
    params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    
    ###################
    # Launch Arguments
    ###################
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation clock'
    )

    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='false',  
        description='Run Gazebo with GUI'
    )

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', 
        default_value='warn',  
        description='Log level for all nodes'
    )
    
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=map_file,
        description='Full path to map yaml file'
    )
    
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=params_file,
        description='Full path to the ROS2 parameters file to use'
    )
    
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', 
        default_value='true',
        description='Automatically startup the nav2 stack'
    )
    
    declare_box1_x = DeclareLaunchArgument(
        'box1_x',
        default_value='-17.0',
        description='X position of the first box'
    )
    
    declare_box1_y = DeclareLaunchArgument(
        'box1_y',
        default_value='-67.0',
        description='Y position of the first box'
    )
    
    declare_box2_x = DeclareLaunchArgument(
        'box2_x',
        default_value='-17.0',
        description='X position of the second box'
    )
    
    declare_box2_y = DeclareLaunchArgument(
        'box2_y',
        default_value='-95.0',
        description='Y position of the second box'
    )
    
    #####################
    # Group 1: Simulation
    #####################
    
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share, 'launch', 'launch_sim.launch.py')
        ]),
        launch_arguments={
            'world': world_file,
            'gui': LaunchConfiguration('gui'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )
    
    rviz_cmd = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config, '--ros-args', '--log-level', 'WARN'],
                output='screen'
            )
        ]
    )
    
    box_spawner_node = TimerAction(
        period=4.0,  
        actions=[
            LogInfo(msg="Spawning obstacle boxes..."),
            Node(
                package='master_project2',
                executable='spawn_boxes.py',
                name='box_spawner',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'box1_x': LaunchConfiguration('box1_x'),
                    'box1_y': LaunchConfiguration('box1_y'),
                    'box1_z': 0.75,  
                    'box2_x': LaunchConfiguration('box2_x'),
                    'box2_y': LaunchConfiguration('box2_y'),
                    'box2_z': 0.75,  
                }],
                output='screen'
            )
        ]
    )
    
    ###########################
    # Group 2: Map & Localization
    ###########################
    
    localization_launch = TimerAction(
        period=5.0,  
        actions=[
            LogInfo(msg="Starting localization..."),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(pkg_share, 'launch', 'localization_launch.py')
                ]),
                launch_arguments={
                    'map': LaunchConfiguration('map'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'autostart': LaunchConfiguration('autostart'),
                    'params_file': LaunchConfiguration('params_file'),
                    'log_level': LaunchConfiguration('log_level')
                }.items()
            )
        ]
    )
    
    ##########################
    # Group 3: Navigation Stack
    ##########################
    
    navigation_launch = TimerAction(
        period=10.0,  
        actions=[
            LogInfo(msg="Starting navigation..."),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(pkg_share, 'launch', 'navigation_launch.py')
                ]),
                launch_arguments={
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'autostart': LaunchConfiguration('autostart'),
                    'params_file': LaunchConfiguration('params_file'),
                    'log_level': LaunchConfiguration('log_level')
                }.items()
            )
        ]
    )   

    
    return LaunchDescription([
        declare_use_sim_time,
        declare_gui,
        declare_map_yaml_cmd,
        declare_params_file_cmd,
        declare_autostart_cmd,
        declare_log_level_cmd, 
        declare_box1_x,
        declare_box1_y,
        declare_box2_x,
        declare_box2_y,

        
        # Group 1: Simulation
        sim_launch,
        rviz_cmd,
        box_spawner_node, 
        
        # Group 2: Map & Localization
        localization_launch,
        
        # Group 3: Navigation Stack
        navigation_launch,
    ])