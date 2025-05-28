from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Set to "false" to run Gazebo in headless mode'
    )

    package_name = 'master_project2'
    package_share_directory = get_package_share_directory(package_name)

    gazebo_models_path = os.path.join(package_share_directory, 'models')

    set_gazebo_model_path = SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_models_path)

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(package_share_directory, 'launch', 'rsp.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items()
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )

    return LaunchDescription([
        declare_gui,
        set_gazebo_model_path,
        rsp,
        gazebo,
        spawn_entity,
    ])