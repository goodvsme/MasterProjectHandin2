#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
    LogInfo,
    OpaqueFunction
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    
    declare_target_waypoints = DeclareLaunchArgument(
        'target_waypoints',
        default_value='',  
        description='Comma-separated list of target waypoint IDs to visit in sequence'
    )
    
    declare_human_count = DeclareLaunchArgument(
        'human_count',
        default_value='130',
        description='Number of humans to spawn'
    )
    
    declare_human_speed = DeclareLaunchArgument(
        'human_speed',
        default_value='0.5',
        description='Speed of spawned humans'
    )
    
    declare_random_seed = DeclareLaunchArgument(
        'random_seed',
        default_value='666',  
        description='Random seed for reproducible human paths'
    )
    
    declare_stop_distance = DeclareLaunchArgument(
        'stop_distance',
        default_value='1.5',
        description='Distance at which humans will stop for the robot (meters)'
    )
    
    declare_stop_percentage = DeclareLaunchArgument(
        'stop_percentage',
        default_value='1.0',
        description='Percentage of humans that will stop for the robot (0.0-1.0)'
    )
    
    declare_map_updater = DeclareLaunchArgument(
        'start_map_updater',
        default_value='true',
        description='Whether to start the map updater node'
    )

    declare_yolo_enabled = DeclareLaunchArgument(
        'start_yolo_visualizer',
        default_value='true',
        description='Whether to start the YOLO visualizer node'
    )

    declare_use_map_frame = DeclareLaunchArgument(
        'use_map_frame', 
        default_value='true', 
        description='Use map frame for human detections'
    )
    
    declare_visualize = DeclareLaunchArgument(
        'visualize', 
        default_value='true', 
        description='Visualize human detections'
    )
    
    declare_duplicate_threshold = DeclareLaunchArgument(
        'duplicate_threshold', 
        default_value='0.5', 
        description='Distance threshold for duplicate detections'
    )
    
    declare_human_radius = DeclareLaunchArgument(
        'human_radius', 
        default_value='0.3', 
        description='Radius of a human for detection purposes'
    )
        
    ###################
    # Wait for waypoints
    ###################
    
    waypoint_wait = ExecuteProcess(
        cmd=[
            'bash', '-c',
            'echo "Waiting for waypoint graph data from main.launch.py..."; ' +
            'until ros2 topic list | grep -q "/waypoint_graph"; do ' +
            'echo "Waiting for waypoint graph to be published..."; ' +
            'sleep 4; ' +
            'done; ' +
            'echo "Waypoint graph data detected!"'
        ],
        name='waypoint_wait',
        output='screen',
        shell=True
    )
    
    ##################################
    # Group 1: Database Handler
    ##################################
    
    database_handler_node = Node(
        package='master_project2',
        executable='database_handler.py',
        name='database_handler',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'database_path': os.path.join(pkg_share, 'db/humans.db'),  
            'deduplication_window': 5.0,
            'radius_threshold': 0.5,
            'publish_interval': 1.0,
            'map_publish_interval': 3,
            'temp_map_yaml': os.path.join(pkg_share, 'maps/temp_map.yaml'),
            'updated_map_yaml': os.path.join(pkg_share, 'maps/updated_map.yaml'),
            'cleaned_map_yaml': os.path.join(pkg_share, 'maps/cleaned_map.yaml'),
            'recent_detection_window': 20.0,
            'republish_interval': 0.1
        }],
        output='screen'
    )
    
    ##################################
    # Group 2: Human Spawner
    ##################################
    
    def launch_human_spawner(context):
        human_count = LaunchConfiguration('human_count').perform(context)
        human_speed = LaunchConfiguration('human_speed').perform(context)
        stop_distance = LaunchConfiguration('stop_distance').perform(context)
        stop_percentage = LaunchConfiguration('stop_percentage').perform(context)
        random_seed = LaunchConfiguration('random_seed').perform(context)
        
        human_spawner_log = LogInfo(msg="Starting human spawner in the background...")
        
        human_spawner = ExecuteProcess(
            cmd=[
                'python3', os.path.join(pkg_share, 'scripts', 'spawn_humans.py'),
                '--ros-args',
                '-p', 'use_sim_time:=true',
                '-p', f'num_humans:={human_count}', 
                '-p', f'human_speed:={human_speed}',
                '-p', f'stop_distance:={stop_distance}',
                '-p', f'stop_percentage:={stop_percentage}',
                '-p', f'random_seed:={random_seed}'
            ],
            name='human_spawner',
            output='screen',
        )
        
        return [human_spawner_log, human_spawner]
    
    human_spawner_launch = OpaqueFunction(function=launch_human_spawner)
    
    ##################################
    # Group 3: Map Updater
    ##################################
    
    def launch_map_updater(context):
        use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
        start_map_updater = LaunchConfiguration('start_map_updater').perform(context)
        
        if start_map_updater.lower() != 'true':
            return []
        
        map_updater_log = LogInfo(msg="Starting split map updater architecture (collector + processor)...")
        
        package_name = 'master_project2'
        pkg_share = get_package_share_directory(package_name)
        mapping_db_path = os.path.join(pkg_share, 'db/mapping.db')
        humans_db_path = os.path.join(pkg_share, 'db/humans.db')
        
        scan_collector = Node(
            package='master_project2',
            executable='scan_collector.py',
            name='scan_collector',
            parameters=[{
                'use_sim_time': use_sim_time == 'true',
                'database_path': mapping_db_path,
                'max_scans': 1000,
                'max_poses': 5000,
                'max_data_age': 300.0,
                'cleanup_interval': 60.0,
                'enable_cmd_vel_tracking': True,
                'compression_enabled': True,
                'buffer_size': 10,
                'use_reliable_qos': True
            }],
            output='screen',
        )
        
        map_processor = Node(
            package='master_project2',
            executable='map_processor.py',
            name='map_processor',
            parameters=[{
                'use_sim_time': use_sim_time == 'true',
                'mapping_database_path': mapping_db_path,
                'humans_database_path': humans_db_path,
                'cell_size': 0.1,
                'd_threshold': 0.15,
                'confidence_threshold': 5,  
                'batch_size': 500,
                'processing_rate': 0.5,
                'verification_threshold': 3,  
                'max_detection_range': 5.0,
                'human_radius': 1.0,
                'laser_has_roll_180': False,
                'laser_x_offset': 0.3059,   # Original: 0.3059  Previous: 0.5059
                'laser_y_offset': 0.0,      # Original: 0.0     Previous: 0.1
                'max_scan_pose_time_diff': 0.05
            }],
            output='screen',
        )
        
        return [map_updater_log, scan_collector, map_processor]
    
    map_updater_launch = OpaqueFunction(function=launch_map_updater)

    ##################################
    # Group 4: YOLO Visualizer
    ##################################
    
    def launch_yolo_visualizer(context):
        start_yolo_visualizer = LaunchConfiguration('start_yolo_visualizer').perform(context)
        use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
        
        if start_yolo_visualizer.lower() != 'true':
            return []
        
        yolo_visualizer_log = LogInfo(msg="Starting YOLO visualizer...")
        
        yolo_visualizer = Node(
            package='master_project2',
            executable='yolo_visualizer.py',
            name='human_detector',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'use_map_frame': LaunchConfiguration('use_map_frame'),
                'human_radius': LaunchConfiguration('human_radius'),
                'visualize': LaunchConfiguration('visualize'),
                'duplicate_threshold': LaunchConfiguration('duplicate_threshold')
            }],
            output='screen'
        )
        
        return [yolo_visualizer_log, yolo_visualizer]
    
    yolo_visualizer_launch = OpaqueFunction(function=launch_yolo_visualizer)
    
    ##################################
    # Group 5: A* Path Planner
    ##################################
    
    def launch_path_planner(context):
        target_waypoints_str = LaunchConfiguration('target_waypoints').perform(context)
        
        mission_monitor = Node(
            package='master_project2',
            executable='mission_monitor.py',
            name='mission_monitor',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'log_directory': 'logs',
                'stop_velocity_threshold': 0.01,
                'stop_time_threshold': 2.0,
                'random_seed': LaunchConfiguration('random_seed'),
            }],
            output='screen'
        )
        
        path_planner_node = Node(
            package='master_project2',
            executable='Astar_pathplanner.py',
            name='astar_path_planner',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            output='screen'
        )
        
        if target_waypoints_str:
            log_msg = f"Starting mission with waypoints: {target_waypoints_str}"
            mission_log = LogInfo(msg=log_msg)
            
            test_script = TimerAction(
                period=15.0,  
                actions=[
                    LogInfo(msg=f"Starting test script with waypoints: {target_waypoints_str}"),
                    Node(
                        package='master_project2',
                        executable='test.py',
                        name='path_planner_tester',
                        parameters=[{
                            'use_sim_time': LaunchConfiguration('use_sim_time'),
                        }],
                        arguments=[target_waypoints_str],
                        output='screen'
                    )
                ]
            )
            return [mission_log, mission_monitor, path_planner_node, test_script]
        else:
            return [
                LogInfo(msg="Starting path planner with no specific waypoints"),
                mission_monitor,
                path_planner_node,
            ]
    
    path_planner_launch = OpaqueFunction(function=launch_path_planner)
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_target_waypoints,
        declare_human_count,
        declare_human_speed,
        declare_random_seed,         
        declare_stop_distance,
        declare_stop_percentage,
        declare_map_updater,
        
        # YOLO arguments
        declare_yolo_enabled,
        declare_use_map_frame,
        declare_visualize,
        declare_duplicate_threshold,
        declare_human_radius,
        
        waypoint_wait,
        
        # Group 1: Database Handler
        database_handler_node,
        
        # Group 2: Human Spawner
        #human_spawner_launch,
        
        # Group 3: Map Updater
        #map_updater_launch,

        # Group 4: YOLO Visualizer
        #yolo_visualizer_launch,
        
        # Group 5: A* Path Planner
        #path_planner_launch
    ])