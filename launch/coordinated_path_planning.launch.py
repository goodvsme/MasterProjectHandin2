import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    use_sim_time = LaunchConfiguration('use_sim_time')
    num_robots = LaunchConfiguration('num_robots')
    safety_distance = LaunchConfiguration('safety_distance')
    wait_time = LaunchConfiguration('wait_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
        
    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to launch')
    
    declare_safety_distance_cmd = DeclareLaunchArgument(
        'safety_distance',
        default_value='1.0',
        description='Safety distance around each robot in meters')
        
    declare_wait_time_cmd = DeclareLaunchArgument(
        'wait_time',
        default_value='10.0',
        description='Time a robot waits when in another robot\'s safety zone')
    
    def launch_coordinator_and_planners(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        safety_distance_value = float(context.launch_configurations['safety_distance'])
        wait_time_value = float(context.launch_configurations['wait_time'])
        
        actions = []
        
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'multibot_path_planning.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'num_robots': num_robots
            }.items()
        ))
        
        actions.append(LogInfo(msg=f"Launching enhanced multibot coordinator for {num_robots_value} robots..."))
        actions.append(Node(
            package='master_project2',
            executable='multibot_coordinator.py',  
            name='multibot_coordinator',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'num_robots': num_robots_value,
                'update_rate': 5.0,  
                'safety_distance': safety_distance_value,  
                'wait_time': wait_time_value, 
                'resume_distance': safety_distance_value * 1.5  
            }]
        ))
        
        return actions
    
    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_num_robots_cmd)
    ld.add_action(declare_safety_distance_cmd)  
    ld.add_action(declare_wait_time_cmd)  
    
    
    ld.add_action(LogInfo(msg="Initializing multi-robot path planning system with enhanced coordination..."))
    
    ld.add_action(OpaqueFunction(function=launch_coordinator_and_planners))
    
    return ld