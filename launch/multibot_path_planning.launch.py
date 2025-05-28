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
        description='Number of robots to launch')
    
    def launch_path_planner_nodes(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        
        actions = []
        actions.append(LogInfo(msg=f"Launching waypoint graph publisher for all robots..."))
        
        actions.append(Node(
            package='master_project2',
            executable='waypoints.py',
            name='waypoint_graph_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'base_update_frequency': 1.0
            }]
        ))
        
        actions.append(LogInfo(msg=f"Launching path planners for {num_robots_value} robots..."))
        
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            
            actions.append(LogInfo(msg=f"Launching path planner for {robot_name}"))
            
            actions.append(Node(
                package='master_project2',
                executable='multibot_path_planner.py',
                namespace=robot_name,
                name='path_planner',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'base_update_frequency': 1.0
                }]
            ))
        
        return actions
    
    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_num_robots_cmd)
    
    ld.add_action(LogInfo(msg="Initializing multi-robot path planning system..."))
    
    ld.add_action(OpaqueFunction(function=launch_path_planner_nodes))
    
    return ld