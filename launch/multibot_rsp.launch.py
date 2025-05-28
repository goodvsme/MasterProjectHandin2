import os
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

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
    
    def launch_rsp_nodes(context):
        num_robots_value = int(context.launch_configurations['num_robots'])
        
        actions = []
        
        for i in range(num_robots_value):
            robot_name = f'robot_{i}'
            
            actions.append(LogInfo(msg=f"Creating robot state publisher for {robot_name}"))
            
            xacro_file = os.path.join(pkg_share, 'description', 'multibot_serena.urdf.xacro')
            
            robot_description_config = xacro.process_file(
                xacro_file, 
                mappings={'robot_namespace': robot_name}
            ).toxml()
            
            actions.append(Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace=robot_name,
                name='robot_state_publisher',
                output='screen',
                parameters=[{
                    'robot_description': robot_description_config,
                    'use_sim_time': use_sim_time,
                    'frame_prefix': f'{robot_name}/',
                    'publish_frequency': 15.0
                }],
                remappings=[
                    ('/tf', '/tf'),
                    ('/tf_static', '/tf_static')
                ]
            ))
        
        return actions
    
    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_num_robots_cmd)
    
    ld.add_action(LogInfo(msg="Initializing robot state publishers..."))
    
    ld.add_action(OpaqueFunction(function=launch_rsp_nodes))
    
    return ld