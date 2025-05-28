from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    ExecuteProcess, 
    DeclareLaunchArgument,
    TimerAction,
    LogInfo
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    rviz_config = os.path.join(pkg_share, 'config', 'multibot_rviz_config.rviz')

    declare_headless = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo in headless mode'
    )
    
    declare_num_robots = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to spawn (up to 5)'
    )
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_map = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_share, 'maps', 'map.yaml'),
        description='Full path to map yaml file to load'
    )
    
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'multi_robot_nav2_params.yaml'),
        description='Full path to multi-robot navigation parameters'
    )

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_share, 'launch', 'multibot_launch_sim.launch.py')
        ]),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'num_robots': LaunchConfiguration('num_robots'),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    rviz_cmd = TimerAction(
        period=5.0,  
        actions=[
            LogInfo(msg="Starting RViz..."),
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],  
                output='screen'
            )
        ]
    )
    
    localization_launch = TimerAction(
        period=10.0,  
        actions=[
            LogInfo(msg="Starting localization stack..."),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(pkg_share, 'launch', 'improved_multi_robot_localization.launch.py')
                ]),
                launch_arguments={
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'num_robots': LaunchConfiguration('num_robots'),
                    'map': LaunchConfiguration('map'),
                    'params_file': LaunchConfiguration('params_file')
                }.items()
            )
        ]
    )
    
    navigation_launch = TimerAction(
        period=20.0,  
        actions=[
            LogInfo(msg="Starting navigation stack..."),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(pkg_share, 'launch', 'multibot_navigation.launch.py')
                ]),
                launch_arguments={
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'num_robots': LaunchConfiguration('num_robots'),
                    'map': LaunchConfiguration('map'),
                    'params_file': LaunchConfiguration('params_file')
                }.items()
            )
        ]
    )

    
    verify_system = TimerAction(
        period=30.0,  
        actions=[
            LogInfo(msg="Verifying multi-robot system..."),
            ExecuteProcess(
                cmd=['ros2', 'topic', 'list'],
                output='screen'
            ),
            ExecuteProcess(
                cmd=['ros2', 'node', 'list'],
                output='screen'
            )
        ]
    )

    
    return LaunchDescription([
        declare_headless,
        declare_num_robots,
        declare_use_sim_time,
        declare_map,
        declare_params_file,
        LogInfo(msg="Starting multi-robot system..."),
        sim_launch,
        rviz_cmd,
        localization_launch,
        navigation_launch,
        verify_system
    ])