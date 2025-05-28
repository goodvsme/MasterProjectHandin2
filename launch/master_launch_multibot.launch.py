from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    ExecuteProcess, 
    DeclareLaunchArgument,
    TimerAction,
    OpaqueFunction,
    SetEnvironmentVariable
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    world_file = os.path.join(pkg_share, 'worlds', 'bilka_odense.world')
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
    
    gazebo_models_path = os.path.join(pkg_share, 'models')
    
    set_gazebo_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_models_path)
    
    gazebo_gui_condition = UnlessCondition(LaunchConfiguration('headless'))
    
    gzserver = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so', 
             '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen'
    )
    
    gzclient = ExecuteProcess(
        condition=gazebo_gui_condition,
        cmd=['gzclient'],
        output='screen'
    )
    
    rviz_cmd = TimerAction(
        period=2.0,  
        actions=[
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen'
            )
        ]
    )
    
    robot_positions = [
        {'name': 'robot_0', 'x': 0.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        {'name': 'robot_1', 'x': 2.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        {'name': 'robot_2', 'x': 0.0, 'y': 2.0, 'z': 0.01, 'yaw': 0.0},
        {'name': 'robot_3', 'x': -2.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        {'name': 'robot_4', 'x': 0.0, 'y': -2.0, 'z': 0.01, 'yaw': 0.0},
    ]
    
    def spawn_robots(context):
        num_robots = int(context.launch_configurations['num_robots'])
        robots_to_spawn = robot_positions[:min(num_robots, len(robot_positions))]
        
        actions = []
        
        xacro_file = os.path.join(pkg_share, 'description', 'multibot_serena.urdf.xacro')
        
        for i, robot in enumerate(robots_to_spawn):
            robot_name = robot['name']
            
            robot_description_config = xacro.process_file(xacro_file)
            
            rsp_node = Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace=robot_name,
                output='screen',
                parameters=[{
                    'robot_description': robot_description_config.toxml(),
                    'use_sim_time': True,
                    'frame_prefix': f'{robot_name}/' 
                }],
                remappings=[
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static')
                ]
            )
            
            spawn_entity = Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                output='screen',
                arguments=[
                    '-topic', f'/{robot_name}/robot_description',
                    '-entity', robot_name,
                    '-robot_namespace', robot_name,
                    '-x', str(robot['x']),
                    '-y', str(robot['y']),
                    '-z', str(robot['z']),
                    '-Y', str(robot['yaw'])
                ]
            )
            
            spawn_timer = TimerAction(
                period=i * 2.0,  
                actions=[spawn_entity]
            )
            
            actions.extend([rsp_node, spawn_timer])
        
        return actions
    
    ld = LaunchDescription([
        declare_headless,
        declare_num_robots,
        set_gazebo_model_path,
        gzserver,  
        gzclient,  
        rviz_cmd,
        OpaqueFunction(function=spawn_robots)
    ])
    
    return ld