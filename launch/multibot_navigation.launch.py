import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                           TimerAction, ExecuteProcess, LogInfo, OpaqueFunction)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    bringup_dir = get_package_share_directory('master_project2')
    
    bt_dir = os.path.join(bringup_dir, 'behavior_trees')
    os.makedirs(bt_dir, exist_ok=True)
    
    minimal_bt_xml_path = os.path.join(bt_dir, 'minimal_navigate.xml')
    
    minimal_bt_xml_content = '''<?xml version="1.0"?>
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      <FollowPath path="{path}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
'''
    
    with open(minimal_bt_xml_path, 'w') as f:
        f.write(minimal_bt_xml_content)

    use_sim_time = LaunchConfiguration('use_sim_time')
    num_robots = LaunchConfiguration('num_robots')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to navigate')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'config', 'multi_robot_nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', 
        default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', 
        default_value='True',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', 
        default_value='info',
        description='log level')

    ld = LaunchDescription()

    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'))
    ld.add_action(SetEnvironmentVariable('RCUTILS_LOGGING_BACKTRACE_ON_ERROR', '1'))

    ld.add_action(declare_num_robots_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)

    ld.add_action(LogInfo(msg=f"Setting up navigation using minimal behavior tree: {minimal_bt_xml_path}"))
    
    def launch_navigation_for_robots(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        
        actions = []
        actions.append(LogInfo(msg=f"Setting up navigation for {num_robots_value} robots..."))
        
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            
            lifecycle_nodes = [
                'controller_server',
                'planner_server',
                'bt_navigator',
                'velocity_smoother'
            ]
            
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
                    'autostart': autostart,
                    'global_frame': 'map',
                    'robot_base_frame': f'{robot_name}/base_footprint',
                    'odom_frame_id': f'{robot_name}/odom',
                    'base_frame_id': f'{robot_name}/base_footprint'
                },
                convert_types=True
            )
            
            actions.append(LogInfo(msg=f"Starting controller server for {robot_name}"))
            actions.append(Node(
                package='nav2_controller',
                executable='controller_server',
                namespace=robot_name,
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[
                    robot_params,
                    {
                        'controller_frequency': 5.0,
                        'global_frame': 'map',
                        'robot_base_frame': f'{robot_name}/base_footprint',
                        'use_sim_time': use_sim_time,
                        
                        'local_costmap': {
                            'global_frame': 'map',
                            'robot_base_frame': f'{robot_name}/base_footprint',
                            'rolling_window': True,
                            'width': 3.0,
                            'height': 3.0,
                            'update_frequency': 5.0,
                            'publish_frequency': 2.0,
                            'transform_tolerance': 1.0,
                            'plugins': ["static_layer", "obstacle_layer", "inflation_layer"],
                            'static_layer': {
                                'map_subscribe_transient_local': True,
                                'map_topic': "/map"
                            },
                            'obstacle_layer': {
                                'observation_sources': 'scan',
                                'scan': {
                                    'topic': f'/{robot_name}/scan',
                                    'max_obstacle_height': 2.0,
                                    'clearing': True,
                                    'marking': True,
                                    'data_type': "LaserScan"
                                }
                            },
                            'inflation_layer': {
                                'cost_scaling_factor': 3.0,
                                'inflation_radius': 0.55
                            }
                        }
                    }
                ],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]
            ))
            
            actions.append(LogInfo(msg=f"Starting planner server for {robot_name}"))
            actions.append(Node(
                package='nav2_planner',
                executable='planner_server',
                name='planner_server',
                namespace=robot_name,
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[
                    robot_params,
                    {
                        'expected_planner_frequency': 5.0,
                        'global_frame': 'map',
                        'robot_base_frame': f'{robot_name}/base_footprint',
                        'use_sim_time': use_sim_time,
                        
                        'global_costmap': {
                            'global_frame': 'map',
                            'robot_base_frame': f'{robot_name}/base_footprint',
                            'rolling_window': False,
                            'update_frequency': 1.0,
                            'publish_frequency': 1.0,
                            'transform_tolerance': 1.0,
                            'plugins': ["static_layer", "obstacle_layer", "inflation_layer"],
                            'static_layer': {
                                'map_subscribe_transient_local': True,
                                'map_topic': "/map"
                            },
                            'obstacle_layer': {
                                'observation_sources': 'scan',
                                'scan': {
                                    'topic': f'/{robot_name}/scan',
                                    'max_obstacle_height': 2.0,
                                    'clearing': True,
                                    'marking': True,
                                    'data_type': "LaserScan"
                                }
                            },
                            'inflation_layer': {
                                'cost_scaling_factor': 3.0,
                                'inflation_radius': 0.55
                            }
                        }
                    }
                ],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings
            ))
            
            actions.append(LogInfo(msg=f"Starting BT navigator for {robot_name}"))
            actions.append(Node(
                package='nav2_bt_navigator',
                executable='bt_navigator',
                name='bt_navigator',
                namespace=robot_name,
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[
                    robot_params,
                    {
                        'global_frame': 'map',
                        'robot_base_frame': f'{robot_name}/base_footprint',
                        'odom_topic': f'/{robot_name}/odom',
                        'transform_tolerance': 1.0,
                        'bt_loop_duration': 30,
                        'default_server_timeout': 90,
                        'default_nav_to_pose_bt_xml': minimal_bt_xml_path,
                        'default_nav_through_poses_bt_xml': minimal_bt_xml_path,
                    }
                ],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings
            ))
            
            actions.append(LogInfo(msg=f"Starting velocity smoother for {robot_name}"))
            actions.append(Node(
                package='nav2_velocity_smoother',
                executable='velocity_smoother',
                name='velocity_smoother',
                namespace=robot_name,
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[
                    robot_params,
                    {
                        'smoothing_frequency': 10.0,
                        'scale_velocities': False,
                        'feedback': "OPEN_LOOP",
                        'max_velocity': [0.4, 0.0, 0.4],
                        'min_velocity': [-0.4, 0.0, -0.4],
                        'max_accel': [1.0, 0.0, 1.5],
                        'max_decel': [-1.0, 0.0, -1.5],
                        'odom_topic': f'/{robot_name}/odom',
                        'odom_duration': 0.5,
                        'deadband_velocity': [0.0, 0.0, 0.0],
                        'velocity_timeout': 1.0
                    }
                ],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=remappings + [
                    ('cmd_vel', 'cmd_vel_nav'), 
                    ('cmd_vel_smoothed', 'cmd_vel')
                ]
            ))
            
            actions.append(LogInfo(msg=f"Starting lifecycle manager for {robot_name}"))
            actions.append(Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                namespace=robot_name,
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': lifecycle_nodes},
                    {'bond_timeout': 60.0},
                    {'bond_respawn_max_duration': 120.0}
                ],
                arguments=['--ros-args', '--log-level', log_level],
            ))
            
            actions.append(TimerAction(
                period=30.0 + (i * 1.0),  
                actions=[
                    LogInfo(msg=f"Setting up periodic costmap clearing for {robot_name}..."),
                    ExecuteProcess(
                        cmd=[
                            'bash', '-c', 
                            f'while sleep 30; do ' +
                            f'ros2 service call /{robot_name}/clear_global_costmap nav2_msgs/srv/ClearEntireCostmap "{{}}"; ' +
                            f'ros2 service call /{robot_name}/clear_local_costmap nav2_msgs/srv/ClearEntireCostmap "{{}}"; ' +
                            f'done'
                        ],
                        output='screen'
                    )
                ]
            ))
        
        actions.append(TimerAction(
            period=45.0,
            actions=[
                LogInfo(msg="Verifying navigation for all robots..."),
                ExecuteProcess(
                    cmd=['ros2', 'node', 'list', '|', 'grep', 'nav'],
                    output='screen',
                    shell=True
                ),
                ExecuteProcess(
                    cmd=['ros2', 'topic', 'list', '|', 'grep', 'costmap'],
                    output='screen',
                    shell=True
                )
            ]
        ))
        
        return actions
        
    ld.add_action(OpaqueFunction(function=launch_navigation_for_robots))
    
    return ld