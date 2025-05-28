from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
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
        description='Use simulation (Gazebo) clock if true'
    )
        
    declare_num_robots_cmd = DeclareLaunchArgument(
        'num_robots',
        default_value='3',
        description='Number of robots to aggregate detections from'
    )
    
    log_info = LogInfo(
        msg="Starting the multibot human detection aggregator and flow grid..."
    )
    
    human_detection_aggregator = Node(
        package='master_project2',
        executable='multibot_human_detection_aggregator.py',
        name='multibot_human_detection_aggregator',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'num_robots': num_robots,
            'detection_topic_pattern': '/robot_{}/human_detections',
            'output_topic': '/raw_human_detections',
            'duplicate_threshold': 0.5,
            'republish_interval': 0.1
        }]
    )
    
    flow_grid = Node(
        package='master_project2',
        executable='multibot_flowgrid.py',
        name='multibot_flow_grid',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'num_directions': 8,
            'use_fixed_resolution': True,
            'fixed_resolution': 0.1,
            'prediction_threshold': 0.01,
            'prediction_concentration': 1.0,
            'max_propagation_distance': 10,
            'distance_decay_factor': 0.95,
            'prediction_scale_factor': 10000,
            'observation_decay_rate': 0.95,
            'observation_base_duration': 0.1,
            'save_path': '/tmp',
            'base_update_frequency': 0.2,
            'rapid_update_interval': 0.1
        }]
    )
    
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_num_robots_cmd,
        log_info,
        human_detection_aggregator,
        flow_grid
    ])