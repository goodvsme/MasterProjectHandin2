from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    ExecuteProcess, 
    DeclareLaunchArgument,
    TimerAction,
    OpaqueFunction,
    SetEnvironmentVariable,
    LogInfo
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    world_file = os.path.join(pkg_share, 'worlds', 'bilka_odense.world')
    rviz_config = os.path.join(pkg_share, 'config', 'multibot_rviz_config.rviz')
    map_yaml_file = os.path.join(pkg_share, 'maps', 'map.yaml')
    
    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Set to "false" to run Gazebo in headless mode'
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
    
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=map_yaml_file,
        description='Full path to map yaml file to load'
    )
    
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'multi_robot_nav2_params.yaml'),
        description='Full path to the ROS2 parameters file'
    )
    
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', 
        default_value='true',
        description='Automatically startup the nav2 stack'
    )
    
    gazebo_models_path = os.path.join(pkg_share, 'models')
    
    set_gazebo_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_models_path)
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'gui': LaunchConfiguration('gui'),  
        }.items()
    )
    
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, 'launch', 'multibot_rsp.launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'num_robots': LaunchConfiguration('num_robots')
        }.items()
    )
    
    rviz_cmd = TimerAction(
        period=3.0, 
        actions=[
            LogInfo(msg="Starting RViz..."),
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen'
            )
        ]
    )
    
    world_to_map_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_map',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map']
    )
    
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'yaml_filename': LaunchConfiguration('map'),
            'frame_id': 'map',
            'topic_name': 'map',
            'use_map_server_node': True,
            'use_transient_local': True,
            'publish_frequency': 2.0
        }],
        remappings=[
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static')
        ]
    )
    
    map_server_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_server',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': LaunchConfiguration('autostart'),
            'node_names': ['map_server'],
            'bond_timeout': 30.0,
            'bond_respawn_max_duration': 60.0
        }],
        remappings=[
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static')
        ]
    )
    
    spawn_positions = [
        (1.0, 0.0, 0.01, 0.0),    # robot_0
        (3.0, 3.0, 0.01, 0.0),    # robot_1
        (4.0, 0.0, 0.01, 0.0),    # robot_2
        (-2.0, 0.0, 0.01, 0.0),   # robot_3
        (0.0, -2.0, 0.01, 0.0),   # robot_4
    ]
    
    def spawn_robots(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        num_robots_to_spawn = min(num_robots_value, len(spawn_positions))
        
        actions = []
        
        actions.append(LogInfo(msg="Setting up map server..."))
        actions.append(world_to_map_transform)
        actions.append(map_server)
        actions.append(map_server_lifecycle_manager)
        
        actions.append(LogInfo(msg=f"Spawning {num_robots_value} robots in Gazebo..."))
        
        for i in range(num_robots_to_spawn):
            robot_name = f'robot_{i}'
            x, y, z, yaw = spawn_positions[i]
            
            actions.append(LogInfo(msg=f"Preparing to spawn {robot_name} at position ({x}, {y}, {z})"))
            
            spawn_entity = Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                output='screen',
                arguments=[
                    '-topic', f'/{robot_name}/robot_description',
                    '-entity', robot_name,
                    '-robot_namespace', robot_name,
                    '-x', str(x),
                    '-y', str(y),
                    '-z', str(z),
                    '-Y', str(yaw)
                ]
            )
            
            spawn_timer = TimerAction(
                period=i * 3.0,  
                actions=[spawn_entity]
            )
            
            actions.append(spawn_timer)
        
        actions.append(TimerAction(
            period=10.0,  
            actions=[
                LogInfo(msg="Verifying robot topics..."),
                ExecuteProcess(
                    cmd=['ros2', 'topic', 'list'],
                    output='screen'
                )
            ]
        ))
        
        return actions
    
    ld = LaunchDescription([
        declare_gui,  
        declare_num_robots,
        declare_use_sim_time,
        declare_map_yaml_cmd,
        declare_params_file_cmd,
        declare_autostart_cmd,
        set_gazebo_model_path,
        
        gazebo,          
        rsp,             
        rviz_cmd,        
        
        OpaqueFunction(function=spawn_robots)
    ])
    
    return ld