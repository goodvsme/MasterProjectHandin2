#!/usr/bin/env python3

import os
import rclpy
import subprocess
import datetime
import shutil
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
    LogInfo,
    OpaqueFunction
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, FindExecutable

def get_user_input(prompt):
    """Get user input with a prompt."""
    try:
        return input(prompt)
    except Exception as e:
        print(f"Error getting user input: {e}")
        return None

def generate_launch_description():
    package_name = 'master_project2'
    pkg_share = get_package_share_directory(package_name)
    
    world_file = os.path.join(pkg_share, 'worlds', 'bilka_odense.world')
    db_dir = os.path.join(pkg_share, 'db')
    logs_dir = os.path.join(pkg_share, 'logs')
    maps_dir = os.path.join(pkg_share, 'maps')
    db_file = os.path.join(db_dir, 'mapping.db')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation clock'
    )
    
    declare_seed = DeclareLaunchArgument(
        'seed',
        default_value='42',
        description='Seed value to include in backup directory name'
    )
    
    print("\n===== SIMULATION RESET =====")
    print("1) Minor Reset: Deletes humans and resets robot position")
    print("2) Complete Reset: Saves all data to a backup folder, resets database, and resets simulation")
    print("===========================")
    reset_type = get_user_input("Enter reset type (1 or 2): ")
    
    if reset_type != "2":
        reset_type = "1"
        print("Performing Minor Reset...")
    else:
        print("Performing Complete Reset with backup...")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    seed = LaunchConfiguration('seed')
    
    def launch_reset_sequence(context):
        actions = []

        if reset_type == "2":  
            seed_value = seed.perform(context)
            backup_dir = os.path.join(pkg_share, f"backups/reset_{timestamp}_seed{seed_value}")
            os.makedirs(backup_dir, exist_ok=True)
            
            actions.append(LogInfo(msg=f"Starting complete reset with backup to {backup_dir}"))
            
            if os.path.exists(db_file):
                backup_cmd = ExecuteProcess(
                    cmd=['cp', db_file, os.path.join(backup_dir, 'db_mapping.db')],
                    output='screen'
                )
                actions.append(backup_cmd)
            
            backup_logs_cmd = ExecuteProcess(
                cmd=['bash', '-c', f'for f in {logs_dir}/*; do cp "$f" "{backup_dir}/log_$(basename "$f")" 2>/dev/null || true; done'],
                output='screen'
            )
            actions.append(backup_logs_cmd)
            
            backup_maps_cmd = ExecuteProcess(
                cmd=['bash', '-c', f'for f in {maps_dir}/*; do cp "$f" "{backup_dir}/map_$(basename "$f")" 2>/dev/null || true; done'],
                output='screen'
            )
            actions.append(backup_maps_cmd)
            
            actions.append(TimerAction(
                period=2.0,
                actions=[LogInfo(msg="Backup completed, proceeding with reset...")]
            ))
        
        
        reset_gazebo = ExecuteProcess(
            cmd=[
                'ros2', 'service', 'call', '/reset_simulation', 'std_srvs/srv/Empty', '{}'
            ],
            output='screen'
        )
        actions.append(reset_gazebo)
        
        reset_robot = ExecuteProcess(
            cmd=[
                'ros2', 'service', 'call', '/gazebo/set_entity_state', 'gazebo_msgs/srv/SetEntityState',
                '{ state: { name: "my_bot", pose: { position: { x: 0.0, y: 0.0, z: 0.1 }, ' +
                'orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } }, ' +
                'twist: { linear: { x: 0.0, y: 0.0, z: 0.0 }, angular: { x: 0.0, y: 0.0, z: 0.0 } }, ' +
                'reference_frame: "world" } }'
            ],
            output='screen'
        )
        actions.append(reset_robot)
        
        reset_amcl = ExecuteProcess(
            cmd=[
                'ros2', 'topic', 'pub', '--once', '/initialpose',
                'geometry_msgs/PoseWithCovarianceStamped',
                '{ header: { frame_id: "map" }, pose: { pose: { position: { x: 0.0, y: 0.0, z: 0.0 }, ' +
                'orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 } }, ' +
                'covariance: [0.25, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, ' +
                '0, 0, 0, 0.0685, 0, 0, 0, 0, 0, 0, 0.0685, 0, 0, 0, 0, 0, 0, 0.0685] } }'
            ],
            output='screen'
        )
        actions.append(reset_amcl)
        
        kill_human_spawner = ExecuteProcess(
            cmd=['pkill', '-f', 'spawn_humans.py'],
            output='screen',
            on_exit=[LogInfo(msg="Attempted to kill spawn_humans.py processes")]
        )
        actions.append(kill_human_spawner)
        
        delete_humans = ExecuteProcess(
            cmd=[
                'bash', '-c',
                'for i in $(seq 0 1000); do ' +
                '  ros2 service call /gazebo/delete_entity gazebo_msgs/srv/DeleteEntity ' +
                '  "{name: \\"human_path_$i\\"}" > /dev/null 2>&1; ' +
                'done'
            ],
            output='screen',
            on_exit=[LogInfo(msg="Attempted to delete all spawned humans")]
        )
        actions.append(delete_humans)
        
        if reset_type == "2":
            clear_database = ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call', '/database_handler/reset_database', 'std_srvs/srv/Trigger', '{}'
                ],
                output='screen',
                on_exit=[LogInfo(msg="Reset database")]
            )
            actions.append(clear_database)
        
        if reset_type == "2":
            actions.append(LogInfo(msg="Complete reset finished. Robot position set to origin (0,0,0)."))
        else:
            actions.append(LogInfo(msg="Minor reset finished. Removed humans and reset robot position to origin (0,0,0)."))
        
        return actions
    
    ld = LaunchDescription()
    
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_seed)
    
    ld.add_action(OpaqueFunction(function=launch_reset_sequence))
    
    return ld