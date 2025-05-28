#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

class BoxSpawner(Node):
    def __init__(self):
        super().__init__('box_spawner')
        
        # boxes parameters 
        self.declare_parameter('box1_x', -17.0) 
        self.declare_parameter('box1_y', -67.0) 
        self.declare_parameter('box1_z', 0.75)  

        self.declare_parameter('box2_x', -17.0) 
        self.declare_parameter('box2_y', -95.0) 
        self.declare_parameter('box2_z', 0.75)  

        self.declare_parameter('box3_x', -11.0)  
        self.declare_parameter('box3_y', -34.0)  
        self.declare_parameter('box3_z', 1.0)
        self.declare_parameter('box3_width', 1.0)
        self.declare_parameter('box3_length', 2.0)
        self.declare_parameter('box3_height', 2.0)

        self.declare_parameter('box4_x', -11.0) 
        self.declare_parameter('box4_y', -41.0) 
        self.declare_parameter('box4_z', 1.0)    
        self.declare_parameter('box4_width', 1.0)
        self.declare_parameter('box4_length', 2.0)
        self.declare_parameter('box4_height', 2.0)

        self.declare_parameter('box5_x', -12.0)  
        self.declare_parameter('box5_y', -48.8)  
        self.declare_parameter('box5_z', 1.0)   
        self.declare_parameter('box5_width', 1.5)
        self.declare_parameter('box5_length', 0.5)
        self.declare_parameter('box5_height', 2.0)

        self.declare_parameter('box6_x', -16.0) 
        self.declare_parameter('box6_y', -48.8)  
        self.declare_parameter('box6_z', 1.0)  
        self.declare_parameter('box6_width', 1.5)
        self.declare_parameter('box6_length', 0.5)
        self.declare_parameter('box6_height', 2.0)

        self.declare_parameter('box7_x', -17.0)  
        self.declare_parameter('box7_y', -81.0)  
        self.declare_parameter('box7_z', 0.5)
        self.declare_parameter('box7_width', 1.0)
        self.declare_parameter('box7_length', 2.0)
        self.declare_parameter('box7_height', 1.0)

        self.declare_parameter('box8_x', -17.0) 
        self.declare_parameter('box8_y', -113.0) 
        self.declare_parameter('box8_z', 0.5)  
        self.declare_parameter('box8_width', 1.0)
        self.declare_parameter('box8_length', 2.0)
        self.declare_parameter('box8_height', 1.0)
        
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for spawn service...')
        
        self.spawn_boxes()
        
    def spawn_boxes(self):
        """Spawn all boxes at the specified locations"""
        box1_x = self.get_parameter('box1_x').value
        box1_y = self.get_parameter('box1_y').value
        box1_z = self.get_parameter('box1_z').value
        box2_x = self.get_parameter('box2_x').value
        box2_y = self.get_parameter('box2_y').value
        box2_z = self.get_parameter('box2_z').value
        
        self.get_logger().info(f'Spawning box 1 at position ({box1_x}, {box1_y}, {box1_z})')
        self.spawn_box('box1', box1_x, box1_y, box1_z, 1.5, 1.5, 1.5)
        
        self.get_logger().info(f'Spawning box 2 at position ({box2_x}, {box2_y}, {box2_z})')
        self.spawn_box('box2', box2_x, box2_y, box2_z, 1.5, 1.5, 1.5)
        
        box3_x = self.get_parameter('box3_x').value
        box3_y = self.get_parameter('box3_y').value
        box3_z = self.get_parameter('box3_z').value
        box3_width = self.get_parameter('box3_width').value
        box3_length = self.get_parameter('box3_length').value
        box3_height = self.get_parameter('box3_height').value
        
        box4_x = self.get_parameter('box4_x').value
        box4_y = self.get_parameter('box4_y').value
        box4_z = self.get_parameter('box4_z').value
        box4_width = self.get_parameter('box4_width').value
        box4_length = self.get_parameter('box4_length').value
        box4_height = self.get_parameter('box4_height').value
        
        box5_x = self.get_parameter('box5_x').value
        box5_y = self.get_parameter('box5_y').value
        box5_z = self.get_parameter('box5_z').value
        box5_width = self.get_parameter('box5_width').value
        box5_length = self.get_parameter('box5_length').value
        box5_height = self.get_parameter('box5_height').value
        
        box6_x = self.get_parameter('box6_x').value
        box6_y = self.get_parameter('box6_y').value
        box6_z = self.get_parameter('box6_z').value
        box6_width = self.get_parameter('box6_width').value
        box6_length = self.get_parameter('box6_length').value
        box6_height = self.get_parameter('box6_height').value
        
        box7_x = self.get_parameter('box7_x').value
        box7_y = self.get_parameter('box7_y').value
        box7_z = self.get_parameter('box7_z').value
        box7_width = self.get_parameter('box7_width').value
        box7_length = self.get_parameter('box7_length').value
        box7_height = self.get_parameter('box7_height').value
        
        box8_x = self.get_parameter('box8_x').value
        box8_y = self.get_parameter('box8_y').value
        box8_z = self.get_parameter('box8_z').value
        box8_width = self.get_parameter('box8_width').value
        box8_length = self.get_parameter('box8_length').value
        box8_height = self.get_parameter('box8_height').value
        
        self.get_logger().info(f'Spawning box 3 at position ({box3_x}, {box3_y}, {box3_z}) with dimensions ({box3_width}, {box3_length}, {box3_height})')
        self.spawn_box('box3', box3_x, box3_y, box3_z, box3_width, box3_length, box3_height)
        
        self.get_logger().info(f'Spawning box 4 at position ({box4_x}, {box4_y}, {box4_z}) with dimensions ({box4_width}, {box4_length}, {box4_height})')
        self.spawn_box('box4', box4_x, box4_y, box4_z, box4_width, box4_length, box4_height)
        
        self.get_logger().info(f'Spawning box 5 at position ({box5_x}, {box5_y}, {box5_z}) with dimensions ({box5_width}, {box5_length}, {box5_height})')
        self.spawn_box('box5', box5_x, box5_y, box5_z, box5_width, box5_length, box5_height)
        
        self.get_logger().info(f'Spawning box 6 at position ({box6_x}, {box6_y}, {box6_z}) with dimensions ({box6_width}, {box6_length}, {box6_height})')
        self.spawn_box('box6', box6_x, box6_y, box6_z, box6_width, box6_length, box6_height)
        
        self.get_logger().info(f'Spawning box 7 at position ({box7_x}, {box7_y}, {box7_z}) with dimensions ({box7_width}, {box7_length}, {box7_height})')
        self.spawn_box('box7', box7_x, box7_y, box7_z, box7_width, box7_length, box7_height)
        
        self.get_logger().info(f'Spawning box 8 at position ({box8_x}, {box8_y}, {box8_z}) with dimensions ({box8_width}, {box8_length}, {box8_height})')
        self.spawn_box('box8', box8_x, box8_y, box8_z, box8_width, box8_length, box8_height)
        
        self.get_logger().info('All boxes spawned successfully')
        
    def spawn_box(self, name, x, y, z, width, length, height):
        """Spawn a single box at the given position with specified dimensions"""
        req = SpawnEntity.Request()
        req.name = name
        req.xml = f"""
        <sdf version='1.7'>
          <model name='{name}'>
            <static>true</static>
            <link name='link'>
              <collision name='collision'>
                <geometry>
                  <box>
                    <size>{width} {length} {height}</size>
                  </box>
                </geometry>
              </collision>
              <visual name='visual'>
                <geometry>
                  <box>
                    <size>{width} {length} {height}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>0.3 0.3 0.3 1</ambient>
                  <diffuse>0.5 0.5 0.5 1</diffuse>
                  <specular>0.1 0.1 0.1 1</specular>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.initial_pose.orientation.w = 1.0
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Successfully spawned {name}')
        else:
            self.get_logger().error(f'Failed to spawn {name}')

def main(args=None):
    rclpy.init(args=args)
    node = BoxSpawner()
    
    try:
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()