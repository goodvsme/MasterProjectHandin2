#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Bool, String
from master_project2.srv import UpdateWaypointCost, NavigateToWaypoint
import threading
import time
import math
from collections import defaultdict

class EnhancedMultibotCoordinator(Node):
    def __init__(self):
        super().__init__('enhanced_multibot_coordinator')
        
        self.declare_parameter('num_robots', 3)
        self.declare_parameter('update_rate', 5.0)
        self.declare_parameter('safety_distance', 1.0)
        self.declare_parameter('wait_time', 3.0)
        self.declare_parameter('resume_distance', 1.5)
        self.declare_parameter('deadlock_timeout', 60.0)
        self.declare_parameter('backup_distance', 3.0)
        self.declare_parameter('blocked_velocity_threshold', 0.1)
        
        self.num_robots = self.get_parameter('num_robots').value
        self.update_rate = self.get_parameter('update_rate').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.wait_time = self.get_parameter('wait_time').value
        self.resume_distance = self.get_parameter('resume_distance').value
        self.deadlock_timeout = self.get_parameter('deadlock_timeout').value
        self.backup_distance = self.get_parameter('backup_distance').value
        self.blocked_velocity_threshold = self.get_parameter('blocked_velocity_threshold').value
        
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.transient_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.robot_poses = {}
        self.robot_status = {}
        self.robot_wait_until = {}
        self.robot_priorities = {}
        
        self.robot_last_poses = {}
        self.robot_blocked_start = {}
        self.robot_velocities = {}
        self.active_deadlock_pairs = {}
        
        self.robot_backup_target = {}
        self.robot_backup_start_pose = {}
        
        self.waypoints = {}
        self.waypoint_costs = {}
        self.original_costs = {}
        
        self.robot_specific_costs = defaultdict(dict)
        
        self.callback_group = ReentrantCallbackGroup()
        
        self.waypoint_graph_sub = self.create_subscription(
            MarkerArray,
            '/waypoint_graph',
            self.waypoint_graph_callback,
            self.transient_qos,
            callback_group=self.callback_group
        )
        
        self.pose_subs = {}
        self.cmd_vel_pubs = {}
        self.wait_pubs = {}
        self.status_pubs = {}
        
        for i in range(self.num_robots):
            robot_ns = f'robot_{i}'
            
            self.robot_priorities[robot_ns] = i
            self.robot_status[robot_ns] = "moving"
            
            self.pose_subs[robot_ns] = self.create_subscription(
                PoseWithCovarianceStamped,
                f'/{robot_ns}/amcl_pose',
                lambda msg, ns=robot_ns: self.pose_callback(msg, ns),
                10,
                callback_group=self.callback_group
            )
            
            self.cmd_vel_pubs[robot_ns] = self.create_publisher(
                Twist,
                f'/{robot_ns}/cmd_vel',
                10
            )
            
            self.wait_pubs[robot_ns] = self.create_publisher(
                Bool,
                f'/{robot_ns}/wait_command',
                10
            )
            
            self.status_pubs[robot_ns] = self.create_publisher(
                String,
                f'/{robot_ns}/coordination_status',
                10
            )
        
        self.cost_update_clients = {}
        self.nav_clients = {}
        
        for i in range(self.num_robots):
            robot_ns = f'robot_{i}'
            
            self.cost_update_clients[robot_ns] = self.create_client(
                UpdateWaypointCost,
                f'/{robot_ns}/update_waypoint_cost',
                callback_group=self.callback_group
            )
            
            self.nav_clients[robot_ns] = self.create_client(
                NavigateToWaypoint,
                f'/{robot_ns}/navigate_to_waypoint',
                callback_group=self.callback_group
            )
        
        self.viz_pub = self.create_publisher(
            MarkerArray,
            '/coordinator_visualization',
            self.reliable_qos
        )
        
        self.coordination_timer = self.create_timer(
            1.0/self.update_rate,
            self.coordination_callback,
            callback_group=self.callback_group
        )
        
        self.lock = threading.RLock()
    
    def waypoint_graph_callback(self, msg):
        try:
            with self.lock:
                for marker in msg.markers:
                    if marker.ns == "waypoints" and marker.type == Marker.SPHERE:
                        waypoint_id = str(marker.id)
                        x = marker.pose.position.x
                        y = marker.pose.position.y
                        
                        self.waypoints[waypoint_id] = (x, y)
                        
                        cost = marker.color.r * 10.0
                        
                        if waypoint_id not in self.original_costs:
                            self.original_costs[waypoint_id] = max(0.1, cost)
                        self.waypoint_costs[waypoint_id] = max(0.1, cost)
                
        except Exception as e:
            pass
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def find_closest_waypoint(self, point):
        if not self.waypoints:
            return None
        
        min_dist = float('inf')
        closest_id = None
        
        for waypoint_id, (x, y) in self.waypoints.items():
            dist = self.calculate_distance(point, (x, y))
            if dist < min_dist:
                min_dist = dist
                closest_id = waypoint_id
        
        return closest_id
    
    def find_waypoints_in_radius(self, center, radius):
        result = []
        
        for waypoint_id, (x, y) in self.waypoints.items():
            dist = self.calculate_distance(center, (x, y))
            if dist <= radius:
                result.append(waypoint_id)
        
        return result
    
    def pose_callback(self, msg, robot_ns):
        try:
            with self.lock:
                self.robot_poses[robot_ns] = msg
                
                current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                current_time = time.time()
                
                if robot_ns in self.robot_last_poses:
                    last_pos, last_time = self.robot_last_poses[robot_ns]
                    
                    if current_time - last_time > 0:
                        distance = self.calculate_distance(current_pos, last_pos)
                        velocity = distance / (current_time - last_time)
                        self.robot_velocities[robot_ns] = velocity
                        
                        if (velocity < self.blocked_velocity_threshold and 
                            self.robot_status.get(robot_ns) == "moving"):
                            
                            if self.is_robot_blocked(robot_ns):
                                if robot_ns not in self.robot_blocked_start:
                                    self.robot_blocked_start[robot_ns] = current_time
                        else:
                            if robot_ns in self.robot_blocked_start:
                                del self.robot_blocked_start[robot_ns]
                
                self.robot_last_poses[robot_ns] = (current_pos, current_time)
                
        except Exception as e:
            pass
    
    def is_robot_blocked(self, robot_ns):
        if robot_ns not in self.robot_poses:
            return False
            
        robot_pos = (self.robot_poses[robot_ns].pose.pose.position.x, 
                     self.robot_poses[robot_ns].pose.pose.position.y)
        
        for other_robot, other_pose in self.robot_poses.items():
            if other_robot != robot_ns:
                other_pos = (other_pose.pose.pose.position.x, other_pose.pose.pose.position.y)
                dist = self.calculate_distance(robot_pos, other_pos)
                
                if (dist < self.safety_distance and 
                    self.robot_priorities[robot_ns] < self.robot_priorities[other_robot]):
                    return True
        
        return False
    
    def coordination_callback(self):
        try:
            with self.lock:
                if not self.waypoints or len(self.robot_poses) < 1:
                    return
                
                for robot_ns in self.robot_specific_costs:
                    self.robot_specific_costs[robot_ns] = {}
                
                self.check_and_handle_deadlocks()
                
                self.check_safety_violations()
                
                self.check_wait_timeouts()
                
                self.check_backup_completion()
                
                self.update_safety_zones()
                
                self.apply_cost_updates()
                
                self.publish_visualization()
        except Exception as e:
            pass
    
    def check_and_handle_deadlocks(self):
        current_time = time.time()
        
        for robot_ns, blocked_start_time in list(self.robot_blocked_start.items()):
            if current_time - blocked_start_time > self.deadlock_timeout:
                blocking_robot = self.find_blocking_robot(robot_ns)
                
                if blocking_robot:
                    if self.robot_priorities[robot_ns] > self.robot_priorities[blocking_robot]:
                        backup_robot = robot_ns
                    else:
                        backup_robot = blocking_robot
                    
                    if self.robot_status[backup_robot] != "backing_up":
                        self.initiate_backup(backup_robot)
                        
                        pair = tuple(sorted([robot_ns, blocking_robot]))
                        self.active_deadlock_pairs[pair] = current_time
    
    def find_blocking_robot(self, robot_ns):
        if robot_ns not in self.robot_poses:
            return None
            
        robot_pos = (self.robot_poses[robot_ns].pose.pose.position.x, 
                     self.robot_poses[robot_ns].pose.pose.position.y)
        
        closest_robot = None
        min_dist = float('inf')
        
        for other_robot, other_pose in self.robot_poses.items():
            if other_robot != robot_ns:
                other_pos = (other_pose.pose.pose.position.x, other_pose.pose.pose.position.y)
                dist = self.calculate_distance(robot_pos, other_pos)
                
                if dist < self.safety_distance and dist < min_dist:
                    min_dist = dist
                    closest_robot = other_robot
        
        return closest_robot
    
    def initiate_backup(self, robot_ns):
        if robot_ns not in self.robot_poses:
            return
            
        self.robot_status[robot_ns] = "backing_up"
        
        current_pose = self.robot_poses[robot_ns].pose.pose
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        
        orientation = current_pose.orientation
        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                         1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z))
        
        backup_x = current_x - self.backup_distance * math.cos(yaw)
        backup_y = current_y - self.backup_distance * math.sin(yaw)
        
        self.robot_backup_target[robot_ns] = (backup_x, backup_y)
        self.robot_backup_start_pose[robot_ns] = (current_x, current_y)
        
        self.send_backup_command(robot_ns)
        
        self.publish_status(robot_ns, f"Backing up {self.backup_distance}m to resolve deadlock")
    
    def send_backup_command(self, robot_ns):
        cmd = Twist()
        cmd.linear.x = -0.3
        cmd.angular.z = 0.0
        
        self.cmd_vel_pubs[robot_ns].publish(cmd)
    
    def check_backup_completion(self):
        robots_backing_up = [ns for ns, status in self.robot_status.items() if status == "backing_up"]
        
        for robot_ns in robots_backing_up:
            if robot_ns not in self.robot_poses or robot_ns not in self.robot_backup_target:
                continue
                
            current_pos = (self.robot_poses[robot_ns].pose.pose.position.x,
                          self.robot_poses[robot_ns].pose.pose.position.y)
            target_pos = self.robot_backup_target[robot_ns]
            
            distance_from_start = self.calculate_distance(current_pos, self.robot_backup_start_pose[robot_ns])
            
            if distance_from_start >= self.backup_distance * 0.9:
                self.complete_backup(robot_ns)
    
    def complete_backup(self, robot_ns):
        stop_cmd = Twist()
        self.cmd_vel_pubs[robot_ns].publish(stop_cmd)
        
        self.robot_status[robot_ns] = "moving"
        
        if robot_ns in self.robot_backup_target:
            del self.robot_backup_target[robot_ns]
        if robot_ns in self.robot_backup_start_pose:
            del self.robot_backup_start_pose[robot_ns]
        
        if robot_ns in self.robot_blocked_start:
            del self.robot_blocked_start[robot_ns]
        
        for pair in list(self.active_deadlock_pairs.keys()):
            if robot_ns in pair:
                del self.active_deadlock_pairs[pair]
        
        self.publish_status(robot_ns, "Backup complete, resuming navigation")
    
    def check_safety_violations(self):
        current_time = time.time()
        
        active_robots = {ns: pos for ns, pos in self.robot_poses.items() 
                        if self.robot_status.get(ns) != "backing_up"}
        
        robot_positions = {}
        for robot_ns, pose_msg in active_robots.items():
            robot_positions[robot_ns] = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        
        for robot1 in robot_positions:
            for robot2 in robot_positions:
                if robot1 >= robot2:
                    continue
                
                dist = self.calculate_distance(robot_positions[robot1], robot_positions[robot2])
                
                if dist < self.safety_distance:
                    
                    if self.robot_priorities[robot1] <= self.robot_priorities[robot2]:
                        wait_robot = robot2
                        continue_robot = robot1
                    else:
                        wait_robot = robot1
                        continue_robot = robot2
                    
                    if self.robot_status[wait_robot] == "moving":
                        self.send_wait_command(wait_robot, current_time + self.wait_time)
                        
                        self.publish_status(continue_robot, f"Continuing (higher priority than {wait_robot})")
                        self.publish_status(wait_robot, f"Waiting for {self.wait_time}s (lower priority than {continue_robot})")
    
    def send_wait_command(self, robot_ns, until_time):
        self.robot_status[robot_ns] = "waiting"
        self.robot_wait_until[robot_ns] = until_time
        
        stop_cmd = Twist()
        self.cmd_vel_pubs[robot_ns].publish(stop_cmd)
        
        wait_msg = Bool()
        wait_msg.data = True
        self.wait_pubs[robot_ns].publish(wait_msg)
    
    def send_resume_command(self, robot_ns):
        self.robot_status[robot_ns] = "moving"
        
        wait_msg = Bool()
        wait_msg.data = False
        self.wait_pubs[robot_ns].publish(wait_msg)
        
        self.publish_status(robot_ns, "Resuming after wait")
    
    def check_wait_timeouts(self):
        current_time = time.time()
        
        waiting_robots = [robot_ns for robot_ns, status in self.robot_status.items() if status == "waiting"]
        
        for robot_ns in waiting_robots:
            if robot_ns in self.robot_wait_until and current_time >= self.robot_wait_until[robot_ns]:
                if self.is_safe_to_resume(robot_ns):
                    self.send_resume_command(robot_ns)
                else:
                    self.robot_wait_until[robot_ns] = current_time + 1.0
    
    def is_safe_to_resume(self, robot_ns):
        if robot_ns not in self.robot_poses:
            return False
            
        robot_pos = (self.robot_poses[robot_ns].pose.pose.position.x, self.robot_poses[robot_ns].pose.pose.position.y)
        
        for other_robot, pose in self.robot_poses.items():
            if other_robot != robot_ns:
                other_pos = (pose.pose.pose.position.x, pose.pose.pose.position.y)
                dist = self.calculate_distance(robot_pos, other_pos)
                
                if dist < self.resume_distance:
                    return False
        
        return True
    
    def update_safety_zones(self):
        for robot_ns, pose_msg in self.robot_poses.items():
            if self.robot_status.get(robot_ns) in ["waiting", "backing_up"]:
                continue
                
            robot_pos = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
            
            safety_waypoints = self.find_waypoints_in_radius(robot_pos, self.safety_distance)
            
            for other_robot in self.robot_poses.keys():
                if other_robot != robot_ns:
                    self.increase_waypoint_costs(safety_waypoints, multiplier=5.0, robot_ns=other_robot)
    
    def increase_waypoint_costs(self, waypoint_ids, multiplier=5.0, robot_ns=None):
        if not robot_ns:
            return
            
        for waypoint_id in waypoint_ids:
            if waypoint_id in self.waypoints:
                base_cost = self.original_costs.get(waypoint_id, 1.0)
                new_cost = base_cost * multiplier
                self.robot_specific_costs[robot_ns][waypoint_id] = min(new_cost, 10.0)
    
    def apply_cost_updates(self):
        try:
            for robot_ns, client in self.cost_update_clients.items():
                if not client.service_is_ready():
                    continue
                
                request = UpdateWaypointCost.Request()
                request.node_id = -1
                
                max_id = 0
                for wid in self.waypoints.keys():
                    try:
                        max_id = max(max_id, int(wid))
                    except ValueError:
                        pass

                costs = [1.0] * (max_id + 1)

                for i in range(max_id + 1):
                    waypoint_id = str(i)
                    
                    if waypoint_id in self.robot_specific_costs[robot_ns]:
                        costs[i] = self.robot_specific_costs[robot_ns][waypoint_id]
                    elif waypoint_id in self.original_costs:
                        costs[i] = self.original_costs[waypoint_id]
                
                request.costs = costs
                
                future = client.call_async(request)
        except Exception as e:
            pass
    
    def publish_status(self, robot_ns, status_text):
        msg = String()
        msg.data = status_text
        self.status_pubs[robot_ns].publish(msg)
    
    def publish_visualization(self):
        try:
            markers = MarkerArray()
            
            for i, (robot_ns, pose_msg) in enumerate(self.robot_poses.items()):
                status_marker = Marker()
                status_marker.header.frame_id = "map"
                status_marker.header.stamp = self.get_clock().now().to_msg()
                status_marker.ns = "robot_status"
                status_marker.id = i
                status_marker.type = Marker.TEXT_VIEW_FACING
                status_marker.action = Marker.ADD
                
                status_marker.pose.position.x = pose_msg.pose.pose.position.x
                status_marker.pose.position.y = pose_msg.pose.pose.position.y
                status_marker.pose.position.z = 0.5
                
                status_marker.scale.z = 0.3
                
                status = self.robot_status.get(robot_ns)
                if status == "waiting":
                    status_marker.color.r = 1.0
                    status_marker.color.g = 0.0
                    status_marker.color.b = 0.0
                    status_marker.color.a = 1.0
                elif status == "backing_up":
                    status_marker.color.r = 1.0
                    status_marker.color.g = 1.0
                    status_marker.color.b = 0.0
                    status_marker.color.a = 1.0
                elif status == "blocked":
                    status_marker.color.r = 1.0
                    status_marker.color.g = 0.5
                    status_marker.color.b = 0.0
                    status_marker.color.a = 1.0
                else:
                    status_marker.color.r = 1.0
                    status_marker.color.g = 1.0
                    status_marker.color.b = 1.0
                    status_marker.color.a = 1.0
                
                status_text = f"{robot_ns}"
                if status == "waiting":
                    status_text += " [WAITING]"
                elif status == "backing_up":
                    status_text += " [BACKING UP]"
                elif robot_ns in self.robot_blocked_start:
                    blocked_duration = time.time() - self.robot_blocked_start[robot_ns]
                    status_text += f" [BLOCKED {blocked_duration:.0f}s]"
                
                status_marker.text = status_text
                
                markers.markers.append(status_marker)
                
                safety_marker = Marker()
                safety_marker.header.frame_id = "map"
                safety_marker.header.stamp = self.get_clock().now().to_msg()
                safety_marker.ns = "safety_zones"
                safety_marker.id = i
                safety_marker.type = Marker.CYLINDER
                safety_marker.action = Marker.ADD
                
                safety_marker.pose.position.x = pose_msg.pose.pose.position.x
                safety_marker.pose.position.y = pose_msg.pose.pose.position.y
                safety_marker.pose.position.z = 0.05
                
                safety_marker.scale.x = 2 * self.safety_distance
                safety_marker.scale.y = 2 * self.safety_distance
                safety_marker.scale.z = 0.1
                
                robot_id = int(robot_ns.split('_')[1])
                r, g, b = self.get_robot_color(robot_id)
                
                if status in ["waiting", "backing_up"]:
                    safety_marker.color.r = r
                    safety_marker.color.g = g
                    safety_marker.color.b = b
                    safety_marker.color.a = 0.15
                else:
                    safety_marker.color.r = r
                    safety_marker.color.g = g
                    safety_marker.color.b = b
                    safety_marker.color.a = 0.3
                
                markers.markers.append(safety_marker)
                
                if status == "backing_up" and robot_ns in self.robot_backup_target:
                    backup_marker = Marker()
                    backup_marker.header.frame_id = "map"
                    backup_marker.header.stamp = self.get_clock().now().to_msg()
                    backup_marker.ns = "backup_arrows"
                    backup_marker.id = i
                    backup_marker.type = Marker.ARROW
                    backup_marker.action = Marker.ADD
                    
                    backup_marker.points.append(Point(x=pose_msg.pose.pose.position.x, 
                                                    y=pose_msg.pose.pose.position.y, 
                                                    z=0.3))
                    backup_marker.points.append(Point(x=self.robot_backup_target[robot_ns][0], 
                                                    y=self.robot_backup_target[robot_ns][1], 
                                                    z=0.3))
                    
                    backup_marker.scale.x = 0.1
                    backup_marker.scale.y = 0.2
                    backup_marker.scale.z = 0.2
                    
                    backup_marker.color.r = 1.0
                    backup_marker.color.g = 1.0
                    backup_marker.color.b = 0.0
                    backup_marker.color.a = 0.8
                    
                    markers.markers.append(backup_marker)
            
            self.viz_pub.publish(markers)
        except Exception as e:
            pass
    
    def get_robot_color(self, robot_id):
        colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.5, 0.0),
            (0.5, 0.0, 1.0),
            (0.0, 0.5, 0.0),
        ]
        
        return colors[robot_id % len(colors)]
    
    def destroy_node(self):
        for robot_ns in self.robot_status:
            if self.robot_status[robot_ns] in ["waiting", "backing_up"]:
                self.send_resume_command(robot_ns)
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    coordinator = EnhancedMultibotCoordinator()
    
    executor = MultiThreadedExecutor()
    executor.add_node(coordinator)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()