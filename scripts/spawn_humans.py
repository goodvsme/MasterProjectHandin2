#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import PoseWithCovarianceStamped
import json
import os
import time
import threading
import math
import random
from ament_index_python.packages import get_package_share_directory

class HumanSpawnerAndMover(Node):
    def __init__(self):
        super().__init__('human_spawner_and_mover')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        self.declare_parameter('start_delay', 20.0)
        self.declare_parameter('num_humans', 130)
        self.declare_parameter('human_speed', 0.5)
        self.declare_parameter('stop_distance', 1.5)
        self.declare_parameter('stop_percentage', 0.3)
        self.declare_parameter('random_seed', 666)
        
        self.start_delay = self.get_parameter('start_delay').value
        self.num_humans = self.get_parameter('num_humans').value
        self.human_speed = self.get_parameter('human_speed').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.stop_percentage = self.get_parameter('stop_percentage').value
        self.random_seed = self.get_parameter('random_seed').value
        
        self.robot_position = None
        self.robot_orientation = 0.0
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.robot_pose_callback,
            10
        )
        
        random_seed = int(self.random_seed)
        random.seed(random_seed)
        self.get_logger().info(f"Using random seed: {random_seed} for reproducible path selection")
        
        self.get_logger().info("Waiting for Gazebo spawn service...")
        self.spawn_client.wait_for_service()
        self.get_logger().info("Spawn service available!")
        
        self.get_logger().info("Waiting for Gazebo set_entity_state service...")
        self.set_state_client.wait_for_service()
        self.get_logger().info("Set entity state service available!")
        
        self.z = 0.0
        
        package_name = 'master_project2'
        package_share_dir = get_package_share_directory(package_name)

        self.waypoints_path = os.path.join(package_share_dir, 'config', 'humans_waypoint_graph.json')
        self.paths_path = os.path.join(package_share_dir, 'config', 'paths.json')
        
        self.humans = {}
        
        self._node_cache = {}
        
        self.update_rate = 20.0
        self.update_interval = 1.0 / self.update_rate
        
        self.waypoints = self.load_waypoints()
        self.paths = self.load_paths()
        
        self.stopping = False
        
        self.get_logger().info(f"Human Spawner initialized with {self.stop_percentage*100:.0f}% of humans stopping when within {self.stop_distance}m of robot")
        
    def robot_pose_callback(self, msg):
        self.robot_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        
        orientation = msg.pose.pose.orientation
        _, _, yaw = self.euler_from_quaternion(
            orientation.x, 
            orientation.y, 
            orientation.z, 
            orientation.w
        )
        self.robot_orientation = yaw
    
    def euler_from_quaternion(self, x, y, z, w):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def load_waypoints(self):
        try:
            self.get_logger().info(f"Loading waypoints from: {self.waypoints_path}")
            with open(self.waypoints_path, 'r') as f:
                data = json.load(f)
                node_count = len(data['nodes']) if 'nodes' in data else 0
                self.get_logger().info(f"Loaded {node_count} waypoints")
                
                if 'nodes' in data:
                    for node in data['nodes']:
                        self._node_cache[node['id']] = node
                    self.get_logger().info(f"Built node cache with {len(self._node_cache)} entries")
                
                return data
        except FileNotFoundError:
            self.get_logger().error(f"Waypoints file not found at {self.waypoints_path}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse waypoints file: {e}")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
        return None
    
    def load_paths(self):
        try:
            self.get_logger().info(f"Loading paths from: {self.paths_path}")
            with open(self.paths_path, 'r') as f:
                data = json.load(f)
                self.get_logger().info(f"Loaded {len(data)} paths")
                return data
        except FileNotFoundError:
            self.get_logger().error(f"Paths file not found at {self.paths_path}")
            return None
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse paths file: {e}")
            return None
        except Exception as e:
            self.get_logger().error(f"Failed to load paths: {e}")
            return None
    
    def get_node_by_id(self, node_id):
        if node_id in self._node_cache:
            return self._node_cache[node_id]
            
        if self.waypoints and 'nodes' in self.waypoints:
            for node in self.waypoints['nodes']:
                if node['id'] == node_id:
                    self._node_cache[node_id] = node
                    return node
        
        self.get_logger().warning(f"Node ID {node_id} not found in waypoints")
        return None
    
    def spawn_human(self, x, y, z, name=None):
        if name is None:
            name = f"human_{x}_{y}"
        
        req = SpawnEntity.Request()
        req.name = name
        req.xml = f"""
        <sdf version='1.7'>
          <model name='{req.name}'>
            <static>0</static>
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
            <!-- Add Gazebo ROS state plugin -->
            <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
            <robot_namespace>/</robot_namespace>
            <update_rate>1</update_rate>
            </plugin>
          </model>
        </sdf>
        """
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            return True
        else:
            self.get_logger().error(f"Failed to spawn human at ({x}, {y}, {z})")
            return False
    
    def set_entity_state(self, entity_name, x, y, z, vx=0.0, vy=0.0, vz=0.0):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = entity_name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation.w = 1.0
        
        req.state.twist.linear.x = vx
        req.state.twist.linear.y = vy
        req.state.twist.linear.z = vz
        
        req.state.reference_frame = 'world'
        
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
            if self._log_counter % 20 == 0:
                pass
        else:
            self._log_counter = 0
        
        future = self.set_state_client.call_async(req)
        return True
    
    def spawn_at_path_starts(self):
        if not self.paths or not self.waypoints:
            self.get_logger().error("Cannot spawn humans - missing path or waypoint data")
            return
        
        num_paths_to_select = min(len(self.paths), int(self.num_humans))
        
        if num_paths_to_select < 1:
            self.get_logger().error("No paths to select. Check if paths.json contains valid paths.")
            return
        
        self.get_logger().info(f"Randomly selecting {num_paths_to_select} paths from {len(self.paths)} available paths")
        selected_paths = random.sample(self.paths, num_paths_to_select)
        
        self.humans_spawned = []
        
        for i, path in enumerate(selected_paths):
            if 'path' not in path:
                self.get_logger().error(f"Path {i} does not have a 'path' field")
                continue
                
            start_node = None
            path_node_ids = path['path']
            
            if 'start' in path:
                start_node_id = path['start']
                start_node = self.get_node_by_id(start_node_id)
                if start_node:
                    pass
            
            if not start_node and path_node_ids:
                first_node_id = path_node_ids[0]
                start_node = self.get_node_by_id(first_node_id)
            
            if not start_node:
                self.get_logger().error(f"Could not find a valid start node for path {i}")
                continue
                
            x = start_node['x']
            y = start_node['y']
            z = start_node.get('z', self.z)
            name = f"human_path_{i}"
            
            if self.spawn_human(x, y, z, name):
                self.humans[name] = {
                    'path_nodes': path_node_ids,
                    'path_index': i,
                    'current_pos': (x, y, z),
                    'current_waypoint_index': 0,
                    'is_stopped': False,
                    'is_avoiding': False,
                    'stop_start_time': None,
                    'will_stop': random.random() < self.stop_percentage
                }
                self.humans_spawned.append(name)
            else:
                self.get_logger().error(f"Failed to spawn human for path {i}")
    
    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def interpolate_position(self, pos1, pos2, fraction):
        x = pos1[0] + (pos2[0] - pos1[0]) * fraction
        y = pos1[1] + (pos2[1] - pos1[1]) * fraction
        z = pos1[2] + (pos2[2] - pos1[2]) * fraction
        return (x, y, z)
    
    def calculate_orientation(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        yaw = math.atan2(dy, dx)
        
        return yaw
    
    def move_human_smoothly(self, human_name):
        if human_name not in self.humans:
            self.get_logger().error(f"Human {human_name} not found in humans dictionary")
            return
        
        human_data = self.humans[human_name]
        current_pos = human_data['current_pos']
        
        if self.robot_position is not None:
            dist_to_robot = self.calculate_distance(current_pos, self.robot_position)
            current_time = time.time()
            
            if dist_to_robot < self.stop_distance and human_data['will_stop']:
                if not human_data['is_stopped']:
                    human_data['is_stopped'] = True
                    human_data['stop_start_time'] = current_time
                    self.set_entity_state(
                        human_name, 
                        current_pos[0], 
                        current_pos[1], 
                        current_pos[2], 
                        0.0, 0.0, 0.0
                    )
                    return
                
                elif current_time - human_data.get('stop_start_time', 0) >= 4.0:
                    
                    dx = current_pos[0] - self.robot_position[0]
                    dy = current_pos[1] - self.robot_position[1]
                    
                    robot_facing = self.robot_orientation
                    
                    if robot_facing is not None:
                        sin_theta = math.sin(robot_facing)
                        cos_theta = math.cos(robot_facing)
                        
                        perp1_x, perp1_y = -sin_theta, cos_theta
                        perp2_x, perp2_y = sin_theta, -cos_theta
                        
                        dot1 = perp1_x*dx + perp1_y*dy
                        dot2 = perp2_x*dx + perp2_y*dy
                        
                        if dot1 > dot2:
                            dx, dy = perp1_x, perp1_y
                        else:
                            dx, dy = perp2_x, perp2_y
                    else:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist > 0:
                            dx /= dist
                            dy /= dist
                    
                    target_x = current_pos[0] + dx
                    target_y = current_pos[1] + dy
                    target_z = current_pos[2]
                    
                    step_size = 1.2 * self.human_speed * self.update_interval
                    
                    new_x = current_pos[0] + dx * step_size
                    new_y = current_pos[1] + dy * step_size
                    new_z = current_pos[2]
                    
                    vx = dx * 1.2 * self.human_speed
                    vy = dy * 1.2 * self.human_speed
                    vz = 0.0
                    
                    self.set_entity_state(human_name, new_x, new_y, new_z, vx, vy, vz)
                    
                    human_data['current_pos'] = (new_x, new_y, new_z)
                    human_data['is_avoiding'] = True
                    return
                
                return
                
            elif human_data['is_stopped'] and dist_to_robot >= self.stop_distance:
                human_data['is_stopped'] = False
                human_data['is_avoiding'] = False
                if 'stop_start_time' in human_data:
                    del human_data['stop_start_time']
        
        if human_data.get('is_avoiding', False):
            if self.robot_position is not None:
                dist_to_robot = self.calculate_distance(current_pos, self.robot_position)
                if dist_to_robot >= 1.5:
                    human_data['is_avoiding'] = False
                    self.get_logger().debug(f"{human_name} has moved far enough away from the robot, resuming path")
            return
        
        if human_data['is_stopped']:
            return
        
        path_nodes = human_data['path_nodes']
        current_waypoint_index = human_data['current_waypoint_index']
        
        if current_waypoint_index >= len(path_nodes) - 1:
            return
        
        current_node_id = path_nodes[current_waypoint_index]
        next_node_id = path_nodes[current_waypoint_index + 1]
        
        current_node = self.get_node_by_id(current_node_id)
        next_node = self.get_node_by_id(next_node_id)
        
        if not current_node or not next_node:
            self.get_logger().error(f"Could not find nodes for {human_name} path segment")
            return
        
        current_waypoint_pos = (current_node['x'], current_node['y'], current_node.get('z', self.z))
        next_waypoint_pos = (next_node['x'], next_node['y'], next_node.get('z', self.z))
        
        distance_to_next = self.calculate_distance(current_pos, next_waypoint_pos)
        
        if distance_to_next < 0.1:
            human_data['current_waypoint_index'] += 1
            return
        
        dx = next_waypoint_pos[0] - current_pos[0]
        dy = next_waypoint_pos[1] - current_pos[1]
        dz = next_waypoint_pos[2] - current_pos[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance > 0:
            dx /= distance
            dy /= distance
            dz /= distance
        
        step_size = self.human_speed * self.update_interval
        
        if step_size > distance:
            step_size = distance
        
        new_x = current_pos[0] + dx * step_size
        new_y = current_pos[1] + dy * step_size
        new_z = current_pos[2] + dz * step_size
        
        vx = dx * self.human_speed
        vy = dy * self.human_speed
        vz = dz * self.human_speed
        
        self.set_entity_state(human_name, new_x, new_y, new_z, vx, vy, vz)
        
        human_data['current_pos'] = (new_x, new_y, new_z)
    
    def follow_paths_smoothly(self):
        self.get_logger().info(f"Waiting {self.start_delay} seconds before starting movement...")
        time.sleep(self.start_delay)
        
        self.get_logger().info(f"Starting movement for {len(self.humans_spawned)} humans at speed {self.human_speed} m/s")
        
        stopping_humans = sum(1 for name in self.humans_spawned if self.humans[name]['will_stop'])
        self.get_logger().info(f"{stopping_humans} humans ({stopping_humans/len(self.humans_spawned)*100:.1f}%) will stop for the robot")
        
        while not self.stopping:
            start_time = time.time()
            
            for human_name in self.humans_spawned:
                self.move_human_smoothly(human_name)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.update_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def follow_all_paths(self):
        thread = threading.Thread(target=self.follow_paths_smoothly)
        thread.daemon = True
        thread.start()
        return thread


def main(args=None):
    rclpy.init(args=args)
    
    node = HumanSpawnerAndMover()
    
    node.spawn_at_path_starts()
    
    time.sleep(1.0)
    
    follow_thread = node.follow_all_paths()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly by user")
    except Exception as e:
        node.get_logger().error(f"Error while running node: {e}")
    finally:
        node.stopping = True
        if follow_thread.is_alive():
            follow_thread.join(timeout=2.0)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()