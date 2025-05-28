#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
import argparse
import time
import sys

class SimpleHumanSpawner(Node):
    def __init__(self, x=2.0, y=1.0, z=0.0, name="test_human"):
        super().__init__('simple_human_spawner')
        
        self.x_position = x
        self.y_position = y
        self.z_position = z
        self.human_name = name
        
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        self.get_logger().info("Waiting for Gazebo spawn service...")
        if not self.spawn_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("Spawn service not available! Is Gazebo running?")
            raise RuntimeError("Spawn service not available")
        
        self.get_logger().info("Spawn service available!")
        
        success = self.spawn_human(
            self.x_position, 
            self.y_position, 
            self.z_position, 
            self.human_name
        )
        
        if success:
            self.get_logger().info(f"Human '{self.human_name}' spawned successfully at ({x}, {y}, {z})")
            self.get_logger().info("Press Ctrl+C to remove the human and exit")
        else:
            self.get_logger().error("Failed to spawn human")
            sys.exit(1)
    
    def spawn_human(self, x, y, z, name):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = f""" 
        <sdf version='1.7'>
        <model name='{req.name}'>
            <static>1</static>
            <link name='link'>
            <collision name='collision'>
                <geometry>
                <cylinder>
                    <radius>0.3</radius>
                    <length>1.8</length>
                </cylinder>
                </geometry>
            </collision>
            <visual name='visual'>
                <geometry>
                <cylinder>
                    <radius>0.3</radius>
                    <length>1.8</length>
                </cylinder>
                </geometry>
                <material>
                <ambient>0.8 0.1 0.1 1</ambient>
                <diffuse>0.8 0.1 0.1 1</diffuse>
                <specular>0.5 0.5 0.5 1</specular>
                </material>
            </visual>
            </link>
        </model>
        </sdf> """
        
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.initial_pose.orientation.w = 1.0
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result() is not None


def main():
    parser = argparse.ArgumentParser(description='Spawn a human in Gazebo for YOLO testing')
    parser.add_argument('-x', '--x-position', type=float, default=3.0, help='X position (default: 2.0)')
    parser.add_argument('-y', '--y-position', type=float, default=0.0, help='Y position (default: 1.0)')
    parser.add_argument('-z', '--z-position', type=float, default=0.0, help='Z position (default: 0.0)')
    parser.add_argument('-n', '--name', type=str, default='test_human', help='Human name (default: test_human)')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        node = SimpleHumanSpawner(
            x=args.x_position,
            y=args.y_position,
            z=args.z_position,
            name=args.name
        )
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down human spawner...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()