import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, SetEnvironmentVariable, 
    TimerAction, ExecuteProcess, LogInfo, OpaqueFunction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    bringup_dir = get_package_share_directory('master_project2')

    use_sim_time = LaunchConfiguration('use_sim_time')
    num_robots = LaunchConfiguration('num_robots')
    map_yaml_file = LaunchConfiguration('map')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    log_level = LaunchConfiguration('log_level')
    
    robot_positions = {
        'robot_0': {'x': 1.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        'robot_1': {'x': 3.0, 'y': 3.0, 'z': 0.01, 'yaw': 0.0},
        'robot_2': {'x': 4.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        'robot_3': {'x': -2.0, 'y': 0.0, 'z': 0.01, 'yaw': 0.0},
        'robot_4': {'x': 0.0, 'y': -2.0, 'z': 0.01, 'yaw': 0.0},
    }
    
    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to localize')
    
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(bringup_dir, 'maps', 'map.yaml'),
        description='Full path to map yaml file to load')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'config', 'multi_robot_nav2_params.yaml'),
        description='Full path to the ROS2 parameters file')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', 
        default_value='true',
        description='Automatically startup the localization stack')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', 
        default_value='info',
        description='Log level')
        
    ld = LaunchDescription()

    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))

    ld.add_action(declare_num_robots_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_log_level_cmd)
    
    def launch_localization_for_robots(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        
        actions = []
        actions.append(LogInfo(msg=f"Setting up localization for {num_robots_value} robots..."))
        
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            robot_pos = robot_positions.get(robot_name, {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0})
            
            remappings = [
                ('/tf', '/tf'),
                ('/tf_static', '/tf_static'),
                ('map', '/map'),  
                ('scan', f'/{robot_name}/scan')
            ]
            
            robot_params = RewrittenYaml(
                source_file=params_file,
                root_key=robot_name,  
                param_rewrites={
                    'use_sim_time': use_sim_time,
                    'global_frame': 'map',
                    'robot_base_frame': f'{robot_name}/base_footprint',
                    'odom_frame_id': f'{robot_name}/odom',
                    'base_frame_id': f'{robot_name}/base_footprint'
                },
                convert_types=True
            )
            
            actions.append(Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'temp_map_to_{robot_name}_odom',
                arguments=[
                    str(robot_pos['x']), 
                    str(robot_pos['y']), 
                    str(robot_pos['z']),
                    '0', '0', str(robot_pos['yaw']), 
                    'map', f'{robot_name}/odom'
                ]
            ))
            
            actions.append(LogInfo(msg=f"Starting AMCL for {robot_name}"))
            actions.append(Node(
                package='nav2_amcl',
                executable='amcl',
                name='amcl',
                namespace=robot_name,
                output='screen',
                parameters=[
                    robot_params,
                    {
                        'publish_tf': True,
                        'tf_broadcast': True,
                        'transform_tolerance': 2.0,
                        
                        'odom_frame_id': f'{robot_name}/odom',
                        'base_frame_id': f'{robot_name}/base_footprint',
                        'global_frame_id': 'map',
                        
                        'initial_pose_x': robot_pos['x'],
                        'initial_pose_y': robot_pos['y'],
                        'initial_pose_z': robot_pos['z'],
                        'initial_pose_a': robot_pos['yaw'],
                        
                        'scan_topic': 'scan',
                        
                        'max_particles': 2000,
                        'min_particles': 500,
                        
                        'recovery_alpha_fast': 0.1,
                        'recovery_alpha_slow': 0.05
                    }
                ],
                remappings=remappings
            ))
            
            actions.append(Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_localization',
                namespace=robot_name,
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'autostart': autostart,
                    'node_names': ['amcl'],
                    'bond_timeout': 60.0,            
                    'bond_respawn_max_duration': 120.0 
                }],
                remappings=remappings
            ))
            
            actions.append(TimerAction(
                period=15.0 + (i * 2.0),  
                actions=[
                    LogInfo(msg=f"Setting initial pose for {robot_name}..."),
                    ExecuteProcess(
                        cmd=[
                            'ros2', 'topic', 'pub', '--once', f'/{robot_name}/initialpose',
                            'geometry_msgs/msg/PoseWithCovarianceStamped',
                            f"""{{header: {{frame_id: "map"}},
                                pose: {{pose: {{position: {{x: {robot_pos['x']}, y: {robot_pos['y']}, z: {robot_pos['z']}}},
                                            orientation: {{z: 0.0, w: 1.0}}}},
                                    covariance: [0.25, 0, 0, 0, 0, 0,
                                                0, 0.25, 0, 0, 0, 0,
                                                0, 0, 0.25, 0, 0, 0,
                                                0, 0, 0, 0.0685, 0, 0,
                                                0, 0, 0, 0, 0.0685, 0,
                                                0, 0, 0, 0, 0, 0.0685]}}}}"""
                        ],
                        output='screen'
                    )
                ]
            ))
        
        actions.append(TimerAction(
            period=30.0,
            actions=[
                LogInfo(msg="Verifying localization for all robots..."),
                ExecuteProcess(
                    cmd=['ros2', 'node', 'list', '|', 'grep', 'amcl'],
                    output='screen',
                    shell=True
                ),
                ExecuteProcess(
                    cmd=['ros2', 'topic', 'list', '|', 'grep', 'amcl_pose'],
                    output='screen',
                    shell=True
                )
            ]
        ))
        
        return actions
        
    ld.add_action(OpaqueFunction(function=launch_localization_for_robots))
    
    return ld