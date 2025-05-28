#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from visualization_msgs.msg import MarkerArray
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import String, Float32MultiArray
from std_srvs.srv import Trigger
import heapq
import math
import tf2_ros
from tf2_ros import TransformException
from rclpy.duration import Duration
from master_project2.srv import NavigateToWaypoint

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('astar_path_planner')
        
        self.waypoints = {}  
        self.graph = {}      
        self.node_costs = {} 
        
        self.received_waypoints = False
        
        self.first_pose_request = True
        
        self.current_pose = None
        
        self.navigating = False
        self.current_path = []
        self.current_waypoint_index = 0
        self.path_orientations = {} 
        
        self.goal_waypoints = []
        self.final_goal_waypoint = None
        
        self.unknown_status_retries = 0
        self.max_unknown_status_retries = 2
        self.progress_history = [] 
        self.last_distance = None  
        self.distance_improved = False  
        self.waypoint_proximity_threshold = 0.4  
        self.last_log_time = None  
        self.strict_waypoint_following = True  
        self.aborted_retries = 0  
        
        self.last_nav_start_time = None
        self.max_nav_time = 120.0 
        
        self.declare_parameter('straight_line_threshold', 0.1)
        self.declare_parameter('turn_angle_threshold', 20.0)
        self.declare_parameter('slow_turn_threshold', 30.0)
        self.straight_line_threshold = self.get_parameter('straight_line_threshold').value
        self.turn_angle_threshold = self.get_parameter('turn_angle_threshold').value * (math.pi/180)
        self.slow_turn_threshold = self.get_parameter('slow_turn_threshold').value * (math.pi/180)
        
        self.path_segments = []
        
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_pose_callback,
            10
        )
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.waypoint_graph_sub = self.create_subscription(
            MarkerArray,
            'waypoint_graph',
            self.waypoint_graph_callback,
            10
        )
        
        self.costs_sub = self.create_subscription(
            Float32MultiArray,
            'waypoint_costs',
            self.waypoint_costs_callback,
            10
        )
        
        self.navigate_service = self.create_service(
            NavigateToWaypoint,
            'navigate_to_waypoint',
            self.navigate_to_waypoint_callback
        )
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.status_pub = self.create_publisher(String, 'path_execution_status', 10)
        
        self.timer = self.create_timer(1.0, self.navigation_status_check)
        
        self.debug_timer = self.create_timer(10.0, self.debug_navigation_status)
        
        self.get_logger().info('Enhanced A* Path Planner with improved continuous movement started')
    
    def waypoint_costs_callback(self, msg):
        self.get_logger().debug("Received waypoint costs data")
        
        for i, cost in enumerate(msg.data):
            if cost < 0:
                continue
            
            node_id = str(i)
            self.node_costs[node_id] = cost
        
        self.get_logger().debug(f"Updated costs for {len(self.node_costs)} waypoints")
        
    def waypoint_graph_callback(self, msg):
        if self.received_waypoints:
            return
            
        waypoints_markers = [m for m in msg.markers if m.ns == "waypoints"]
        edges_markers = [m for m in msg.markers if m.ns == "edges"]
        
        if not waypoints_markers:
            return
            
        self.waypoints = {}
        self.graph = {}
        
        for marker in waypoints_markers:
            node_id = str(marker.id)
            x = marker.pose.position.x
            y = marker.pose.position.y
            
            self.waypoints[node_id] = (x, y)
            self.graph[node_id] = []
            
            if node_id not in self.node_costs:
                self.node_costs[node_id] = 1.0
        
        processed_edges = set()
        for marker in edges_markers:
            if len(marker.points) >= 2:
                start_point = (marker.points[0].x, marker.points[0].y)
                end_point = (marker.points[1].x, marker.points[1].y)
                
                start_node = self.find_closest_waypoint(start_point)
                end_node = self.find_closest_waypoint(end_point)
                
                if start_node and end_node and start_node != end_node:
                    edge_id = tuple(sorted([start_node, end_node]))
                    
                    if edge_id not in processed_edges:
                        processed_edges.add(edge_id)
                        
                        if end_node not in self.graph[start_node]:
                            self.graph[start_node].append(end_node)
                        if start_node not in self.graph[end_node]:
                            self.graph[end_node].append(start_node)
        
        num_edges = sum(len(edges) for edges in self.graph.values()) // 2
        self.get_logger().info(f'Loaded waypoint graph: {len(self.waypoints)} waypoints, {num_edges} edges')
        
        self.received_waypoints = True
        
    def find_closest_waypoint(self, point):
        if not self.waypoints:
            return None
            
        closest = min(self.waypoints.items(), 
                     key=lambda item: self.calculate_distance(point, item[1]))
        return closest[0]
    
    def amcl_pose_callback(self, msg):
        self.current_pose = msg
        self.get_logger().debug(f"Updated robot pose from AMCL: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})")
    
    def get_robot_pose(self):
        if self.current_pose is not None:
            x = self.current_pose.pose.pose.position.x
            y = self.current_pose.pose.pose.position.y
            self.get_logger().debug(f"Using AMCL pose: ({x}, {y})")
            
            self.first_pose_request = False
            return (x, y)
            
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                now,
                timeout=Duration(seconds=1.0)
            )
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            
            self.get_logger().debug(f"Got robot pose from TF: ({x}, {y})")
            
            self.first_pose_request = False
            return (x, y)
            
        except TransformException as ex:
            if self.first_pose_request:
                self.get_logger().warning(
                    f"Could not get initial robot pose: {ex}. Using origin (0,0) as starting point."
                )
                self.first_pose_request = False
                return (0.0, 0.0)
            else:
                self.get_logger().warning(
                    f"Could not get robot pose: {ex}. Returning None."
                )
                return None
            
    def find_closest_waypoint_to_pose(self, pose):
        if not self.waypoints or not pose:
            return None
            
        closest = min(self.waypoints.items(), 
                     key=lambda item: self.calculate_distance(pose, item[1]))
        return closest[0]
    
    def navigate_to_waypoint_callback(self, request, response):
        waypoint_request = request.waypoint_id
        
        if self.navigating:
            response.success = False
            response.message = "Navigation already in progress. Please cancel before starting a new path."
            self.get_logger().error(response.message)
            return response
        
        wait_count = 0
        while not self.received_waypoints and wait_count < 5:
            self.get_logger().info("Waiting for waypoint graph data...")
            import time
            time.sleep(1.0)
            wait_count += 1
            
        if not self.received_waypoints:
            response.success = False
            response.message = "No waypoint graph data available. Please check the waypoint_graph topic."
            self.get_logger().error(response.message)
            return response
        
        if ',' in waypoint_request:
            self.goal_waypoints = [wp.strip() for wp in waypoint_request.split(',')]
            self.get_logger().info(f"Received multi-waypoint navigation request: {self.goal_waypoints}")
            target_waypoint = self.goal_waypoints[-1]
            self.final_goal_waypoint = target_waypoint
        else:
            target_waypoint = waypoint_request
            self.goal_waypoints = [target_waypoint]
            self.final_goal_waypoint = target_waypoint
        
        for waypoint in self.goal_waypoints:
            if waypoint not in self.waypoints:
                response.success = False
                response.message = f"Waypoint {waypoint} not found in the graph"
                self.get_logger().error(response.message)
                return response
        
        start_waypoint = request.start_waypoint_id
        
        if not start_waypoint or start_waypoint not in self.waypoints:
            max_attempts = 10
            attempts = 0
            robot_pose = None
            
            while robot_pose is None and attempts < max_attempts:
                self.get_logger().info(f"Waiting for valid robot pose (attempt {attempts+1}/{max_attempts})...")
                
                robot_pose = self.get_robot_pose()
                
                if robot_pose is None:
                    import time
                    time.sleep(1.0)
                    attempts += 1
            
            if robot_pose is None:
                response.success = False
                response.message = "Failed to get robot pose after multiple attempts"
                self.get_logger().error(response.message)
                return response
                
            start_waypoint = self.find_closest_waypoint_to_pose(robot_pose)
            self.get_logger().info(f"Using closest waypoint {start_waypoint} to position ({robot_pose[0]}, {robot_pose[1]})")

        self.get_logger().info(f"Planning full path from {start_waypoint} through waypoints {self.goal_waypoints}")
        
        combined_path = []
        current_start = start_waypoint
        
        for waypoint in self.goal_waypoints:
            self.get_logger().info(f"Planning path segment from {current_start} to {waypoint}")
            path_segment = self.plan_path_astar(current_start, waypoint)
            
            if not path_segment:
                response.success = False
                response.message = f"Could not find a path from {current_start} to {waypoint}"
                self.get_logger().error(response.message)
                return response
            
            if combined_path and path_segment:
                path_segment = path_segment[1:]
                
            combined_path.extend(path_segment)
            current_start = waypoint
        
        if not combined_path:
            response.success = False
            response.message = f"Failed to generate a valid combined path through all waypoints"
            self.get_logger().error(response.message)
            return response
        
        self.current_path = combined_path
        self.current_waypoint_index = 0
        
        self.analyze_path_segments(combined_path)
        
        self.calculate_path_orientations(combined_path)
        
        self.get_logger().info(f"Full path planned with {len(combined_path)} waypoints")
        
        path_str = ' -> '.join(self.current_path[:10])
        if len(self.current_path) > 10:
            path_str += f" ... (and {len(self.current_path)-10} more waypoints)"
        self.get_logger().info(f"Path: {path_str}")
        
        self.navigating = True
        
        self.navigate_to_next_waypoint()
        
        response.success = True
        response.message = f"Navigation to waypoint sequence {self.goal_waypoints} started"
        return response
    
    def analyze_path_segments(self, path):
        if len(path) < 3:
            self.path_segments = ["turn" for _ in path]
            return
                
        self.path_segments = []
        
        self.path_segments.append("turn")
        
        for i in range(1, len(path) - 1):
            prev_wp = self.waypoints[path[i-1]]
            curr_wp = self.waypoints[path[i]]
            next_wp = self.waypoints[path[i+1]]
            
            v1 = (curr_wp[0] - prev_wp[0], curr_wp[1] - prev_wp[1])
            v2 = (next_wp[0] - curr_wp[0], next_wp[1] - curr_wp[1])
            
            angle1 = math.atan2(v1[1], v1[0])
            angle2 = math.atan2(v2[1], v2[0])
            
            turn_angle = abs(self.normalize_angle(angle2 - angle1))
            
            if turn_angle > self.turn_angle_threshold:
                self.path_segments.append("turn")
                if turn_angle > self.turn_angle_threshold * 2:
                    self.get_logger().debug(f"Sharp turn detected at waypoint {path[i]} - angle: {turn_angle:.2f} rad")
            else:
                cross_product = v1[0]*v2[1] - v1[1]*v2[0]
                distance = abs(cross_product) / math.sqrt(v2[0]**2 + v2[1]**2)
                
                if distance < self.straight_line_threshold:
                    self.path_segments.append("straight")
                    
                    straight_count = 1
                    for j in range(i+1, min(i+5, len(path)-1)):
                        next_wp1 = self.waypoints[path[j]]
                        next_wp2 = self.waypoints[path[j+1]]
                        v_next = (next_wp2[0] - next_wp1[0], next_wp2[1] - next_wp1[1])
                        angle_next = math.atan2(v_next[1], v_next[0])
                        angle_diff = abs(self.normalize_angle(angle_next - angle1))
                        
                        if angle_diff < self.turn_angle_threshold:
                            straight_count += 1
                        else:
                            break
                            
                    if straight_count >= 3:
                        self.get_logger().debug(f"Long straight segment detected at waypoint {path[i]} extending for {straight_count} waypoints")
                else:
                    self.path_segments.append("turn")
        
        self.path_segments.append("turn")
        
        turn_count = self.path_segments.count("turn")
        straight_count = self.path_segments.count("straight")
        self.get_logger().info(f"Path analysis: {turn_count} turn points and {straight_count} straight segments")
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def calculate_path_orientations(self, path):
        if len(path) < 2:
            self.path_orientations = {}
            return
                
        self.path_orientations = {}
        
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            
            current_pos = self.waypoints[current_id]
            next_pos = self.waypoints[next_id]
            
            raw_yaw = self.calculate_heading(current_pos, next_pos)
            
            yaw = self.snap_to_45_degrees(raw_yaw)
            
            q = self.yaw_to_quaternion(yaw)
            self.path_orientations[current_id] = q
                
        last_id = path[-1]
        if len(path) > 1:
            prev_id = path[-2]
            prev_pos = self.waypoints[prev_id]
            last_pos = self.waypoints[last_id]
            
            raw_yaw = self.calculate_heading(prev_pos, last_pos)
            yaw = self.snap_to_45_degrees(raw_yaw)
            q = self.yaw_to_quaternion(yaw)
            
            self.path_orientations[last_id] = q
        else:
            q = Quaternion()
            q.w = 1.0
            self.path_orientations[last_id] = q
    
    def snap_to_45_degrees(self, angle):
        deg_angle = angle * 180.0 / math.pi
        
        rounded = round(deg_angle / 45.0) * 45.0
        
        if abs(deg_angle - rounded) <= 5.0:
            return rounded * math.pi / 180.0
        
        return angle
    
    def debug_navigation_status(self):
        if not self.navigating:
            return
                
        if self.current_waypoint_index < len(self.current_path):
            current_waypoint_id = self.current_path[self.current_waypoint_index]
            waypoint_type = "USER-REQUESTED" if current_waypoint_id in self.goal_waypoints else "intermediate"
            waypoint_idx = f"{self.current_waypoint_index+1}/{len(self.current_path)}"
            
            if current_waypoint_id in self.goal_waypoints:
                goal_idx = self.goal_waypoints.index(current_waypoint_id)
                goal_progress = f"{goal_idx+1}/{len(self.goal_waypoints)}"
                self.get_logger().info(
                    f"Current target: {waypoint_type} waypoint {current_waypoint_id} "
                    f"(goal {goal_progress}, waypoint {waypoint_idx})"
                )
            else:
                self.get_logger().info(
                    f"Current target: {waypoint_type} waypoint {current_waypoint_id} "
                    f"(waypoint {waypoint_idx})"
                )
                
            robot_pose = self.get_robot_pose()
            if robot_pose:
                waypoint_pos = self.waypoints[current_waypoint_id]
                distance = self.calculate_distance(robot_pose, waypoint_pos)
                self.get_logger().info(f"Distance to target: {distance:.2f} meters")
    
    def navigation_status_check(self):
        if not self.navigating:
            return
            
        if self.current_waypoint_index < len(self.current_path):
            import time
            current_time = time.time()
            
            if hasattr(self, 'last_nav_start_time') and self.last_nav_start_time is not None:
                elapsed_time = current_time - self.last_nav_start_time
                
                current_waypoint_id = self.current_path[self.current_waypoint_index]
                if elapsed_time > 120.0 and current_waypoint_id not in self.goal_waypoints:
                    self.get_logger().warn(
                        f"Navigation to waypoint {current_waypoint_id} has taken too long ({elapsed_time:.1f}s). "
                        f"Moving to next waypoint."
                    )
                    
                    self.current_waypoint_index += 1
                    self.unknown_status_retries = 0
                    self.last_nav_start_time = None
                    self.progress_history = []
                    self.distance_improved = False
                    
                    if self.current_waypoint_index < len(self.current_path):
                        self.navigate_to_next_waypoint()
                    else:
                        status_msg = String()
                        status_msg.data = "Path execution completed with some waypoints skipped due to timeout"
                        self.status_pub.publish(status_msg)
                        self.navigating = False
                    return
            
            robot_pose = self.get_robot_pose()
            if not robot_pose:
                return
                
            current_waypoint_id = self.current_path[self.current_waypoint_index]
            waypoint_pos = self.waypoints[current_waypoint_id]
            
            distance = self.calculate_distance(robot_pose, waypoint_pos)
            
            if not hasattr(self, 'progress_history'):
                self.progress_history = []
                
            if len(self.progress_history) >= 10:
                self.progress_history.pop(0)
            self.progress_history.append(distance)
            
            if len(self.progress_history) >= 5:
                oldest_distance = self.progress_history[0]
                newer_distances = self.progress_history[-3:]
                avg_recent = sum(newer_distances) / len(newer_distances)
                
                if oldest_distance - avg_recent > 0.5:
                    self.distance_improved = True
                    if self.unknown_status_retries > 0:
                        self.get_logger().info(
                            f"Making progress toward waypoint {current_waypoint_id} despite STATUS_UNKNOWN "
                            f"(distance reduced from {oldest_distance:.2f}m to {avg_recent:.2f}m)"
                        )
            
            proximity_threshold = self.waypoint_proximity_threshold
            if current_waypoint_id in self.goal_waypoints:
                proximity_threshold = 0.2
                
            if distance < proximity_threshold:
                self.get_logger().info(
                    f"Robot is close enough to waypoint {current_waypoint_id} (distance={distance:.2f}m). "
                    f"Considering it reached."
                )
                
                if current_waypoint_id in self.goal_waypoints:
                    idx = self.goal_waypoints.index(current_waypoint_id)
                    status_msg = String()
                    status_msg.data = f"Reached waypoint {current_waypoint_id} ({idx+1}/{len(self.goal_waypoints)})"
                    self.status_pub.publish(status_msg)
                    
                    self.get_logger().info(f"✓✓✓ REACHED USER-REQUESTED WAYPOINT {current_waypoint_id} ✓✓✓")
                
                if current_waypoint_id == self.final_goal_waypoint:
                    self.get_logger().info(f"Reached final goal waypoint {current_waypoint_id}!")
                    status_msg = String()
                    status_msg.data = "Path execution completed successfully"
                    self.status_pub.publish(status_msg)
                    self.navigating = False
                    return
                
                self.current_waypoint_index += 1
                self.unknown_status_retries = 0
                self.last_nav_start_time = None
                self.progress_history = []
                self.distance_improved = False
                
                if self.current_waypoint_index < len(self.current_path):
                    self.navigate_to_next_waypoint()
                else:
                    status_msg = String()
                    status_msg.data = "Path execution completed successfully (proximity detection)"
                    self.status_pub.publish(status_msg)
                    self.navigating = False
            
            self.last_distance = distance
            
            if self.last_log_time is None or current_time - self.last_log_time > 10.0:
                waypoint_type = "GOAL" if current_waypoint_id in self.goal_waypoints else "intermediate"
                self.get_logger().info(
                    f"Distance to {waypoint_type} waypoint {current_waypoint_id}: {distance:.2f} meters. "
                    f"Waypoint {self.current_waypoint_index + 1}/{len(self.current_path)}"
                )
                self.last_log_time = current_time
    
    def should_skip_waypoint(self, index):
        if index == 0 or index == len(self.current_path) - 1:
            return False
            
        current_waypoint_id = self.current_path[index]
        if current_waypoint_id in self.goal_waypoints:
            return False
            
        if self.path_segments[index] == "turn":
            return False
        
        next_index = index + 1
        if next_index >= len(self.current_path):
            return False
            
        next_waypoint_id = self.current_path[next_index]
        
        if next_waypoint_id in self.goal_waypoints or self.path_segments[next_index] == "turn":
            return False
            
        current_pos = self.waypoints[current_waypoint_id]
        next_pos = self.waypoints[next_waypoint_id]
        
        distance_to_next = self.calculate_distance(current_pos, next_pos)
        if distance_to_next > 4.0:
            return False
        
        return self.path_segments[index] == "straight"
    
    def plan_path_astar(self, start_waypoint, target_waypoint):
        if start_waypoint not in self.waypoints or target_waypoint not in self.waypoints:
            self.get_logger().error(f"Start or target waypoint not in graph")
            return []
            
        if start_waypoint == target_waypoint:
            return [start_waypoint]
            
        open_set = []
        closed_set = set()
        
        g_score = {}
        g_score[start_waypoint] = 0
        
        f_score = {}
        f_score[start_waypoint] = self.heuristic(start_waypoint, target_waypoint)
        
        in_open_set = {start_waypoint}
        
        heapq.heappush(open_set, (f_score[start_waypoint], 0, start_waypoint))
        
        came_from = {}
        
        nodes_explored = 0
        counter = 1
        
        max_nodes_to_explore = min(10000, len(self.waypoints) * 3)
        
        self.get_logger().info(f"Starting A* search from {start_waypoint} to {target_waypoint}")
        
        while open_set and nodes_explored < max_nodes_to_explore:
            _, _, current = heapq.heappop(open_set)
            in_open_set.remove(current)
            
            nodes_explored += 1
            
            if current == target_waypoint:
                path = self.reconstruct_path(came_from, current)
                self.get_logger().info(f"Path found with {len(path)} waypoints (explored {nodes_explored} nodes)")
                return path
            
            closed_set.add(current)
            
            for neighbor in self.graph[current]:
                if neighbor in closed_set:
                    continue
                
                distance = self.calculate_distance(
                    self.waypoints[current], 
                    self.waypoints[neighbor]
                )
                
                neighbor_cost = self.node_costs.get(neighbor, 1.0)
                tentative_g_score = g_score[current] + distance * neighbor_cost
                
                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                neighbor_f_score = tentative_g_score + self.heuristic(neighbor, target_waypoint)
                f_score[neighbor] = neighbor_f_score
                
                if neighbor not in in_open_set:
                    counter += 1
                    heapq.heappush(open_set, (neighbor_f_score, counter, neighbor))
                    in_open_set.add(neighbor)
        
        if nodes_explored >= max_nodes_to_explore:
            self.get_logger().warning(f"A* search stopped after exploring {nodes_explored} nodes - path might not be optimal")
        
        self.get_logger().error(f"No path found from {start_waypoint} to {target_waypoint}")
        return []
    
    def heuristic(self, waypoint1, waypoint2):
        return self.calculate_distance(
            self.waypoints[waypoint1],
            self.waypoints[waypoint2]
        )
    
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        
        return total_path[::-1]
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_heading(self, from_point, to_point):
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        return math.atan2(dy, dx)
    
    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    def get_robot_current_quaternion(self):
        if self.current_pose is not None:
            return self.current_pose.pose.pose.orientation
        else:
            q = Quaternion()
            q.w = 1.0
            return q
    
    def navigate_to_next_waypoint(self):
        if not self.navigating or not self.current_path:
            return
            
        if self.current_waypoint_index >= len(self.current_path):
            self.get_logger().info("Reached the end of the full path")
            status_msg = String()
            status_msg.data = "Path execution completed successfully"
            self.status_pub.publish(status_msg)
            self.navigating = False
            return
        
        if self.should_skip_waypoint(self.current_waypoint_index):
            self.get_logger().info(f"Skipping intermediate waypoint {self.current_path[self.current_waypoint_index]} for continuous movement")
            self.current_waypoint_index += 1
            self.navigate_to_next_waypoint()
            return
        
        waypoint_id = self.current_path[self.current_waypoint_index]
        x, y = self.waypoints[waypoint_id]
        
        is_goal_waypoint = waypoint_id in self.goal_waypoints
        is_final_goal = waypoint_id == self.final_goal_waypoint
        
        if is_final_goal:
            self.get_logger().info(f"Navigating to FINAL GOAL waypoint {waypoint_id} at ({x}, {y})")
        elif is_goal_waypoint:
            idx = self.goal_waypoints.index(waypoint_id)
            self.get_logger().info(f"Navigating to USER-SPECIFIED waypoint {waypoint_id} ({idx+1}/{len(self.goal_waypoints)}) at ({x}, {y})")
        else:
            if self.current_waypoint_index < len(self.path_segments):
                segment_type = self.path_segments[self.current_waypoint_index]
                self.get_logger().info(
                    f"Navigating to intermediate waypoint {waypoint_id} at ({x}, {y}) - {segment_type} segment"
                )
            else:
                self.get_logger().info(f"Navigating to intermediate waypoint {waypoint_id} at ({x}, {y})")
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        
        if waypoint_id in self.path_orientations:
            goal_pose.pose.orientation = self.path_orientations[waypoint_id]
            self.get_logger().debug(
                f"Using pre-calculated orientation for waypoint {waypoint_id}"
            )
        else:
            goal_pose.pose.orientation = self.get_robot_current_quaternion()
            self.get_logger().debug(
                f"Using current robot orientation for waypoint {waypoint_id}"
            )
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.nav_client.wait_for_server()
        
        status_msg = String()
        if is_final_goal:
            status_msg.data = f"Navigating to final goal waypoint {waypoint_id} ({self.current_waypoint_index+1}/{len(self.current_path)})"
        elif is_goal_waypoint:
            idx = self.goal_waypoints.index(waypoint_id)
            status_msg.data = f"Navigating to waypoint {waypoint_id} ({idx+1}/{len(self.goal_waypoints)})"
        else:
            status_msg.data = f"Navigating to intermediate waypoint {waypoint_id}"
        self.status_pub.publish(status_msg)
        
        import time
        self.last_nav_start_time = time.time()
        
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            self.handle_navigation_failure()
            return
            
        self.get_logger().info("Goal accepted, waiting for result...")
        
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        try:
            result_wrapper = future.result()
            if result_wrapper is None:
                self.get_logger().error("Got None result wrapper, cannot determine navigation status")
                
                robot_pose = self.get_robot_pose()
                if robot_pose and self.current_waypoint_index < len(self.current_path):
                    current_waypoint_id = self.current_path[self.current_waypoint_index]
                    waypoint_pos = self.waypoints[current_waypoint_id]
                    distance = self.calculate_distance(robot_pose, waypoint_pos)
                    
                    if distance < 3.0:
                        self.get_logger().info(f"Despite error, robot is close to waypoint {current_waypoint_id} (distance={distance:.2f}m)")
                        self.navigation_status_check()
                        return
                        
                self.handle_navigation_failure()
                return
                
            result = result_wrapper.result
            status = result_wrapper.status
            
            status_names = {
                1: "STATUS_EXECUTING",
                2: "STATUS_CANCELING", 
                3: "STATUS_CANCELED",
                4: "STATUS_SUCCEEDED",
                5: "STATUS_ABORTED",
                6: "STATUS_UNKNOWN"
            }
            status_name = status_names.get(status, f"UNKNOWN_STATUS_{status}")
            self.get_logger().info(f"Navigation result received with status: {status} ({status_name})")
            
            if status == 4:
                waypoint_id = self.current_path[self.current_waypoint_index]
                
                if waypoint_id in self.goal_waypoints:
                    idx = self.goal_waypoints.index(waypoint_id)
                    self.get_logger().info(f"Successfully reached goal waypoint {waypoint_id} ({idx+1}/{len(self.goal_waypoints)})")
                    
                    status_msg = String()
                    status_msg.data = f"Reached waypoint {waypoint_id} ({idx+1}/{len(self.goal_waypoints)})"
                    self.status_pub.publish(status_msg)
                    
                    if waypoint_id == self.final_goal_waypoint:
                        self.get_logger().info("FINAL GOAL REACHED! Mission completed!")
                        status_msg = String()
                        status_msg.data = "Path execution completed successfully"
                        self.status_pub.publish(status_msg)
                        self.navigating = False
                        return
                else:
                    self.get_logger().info(f"Successfully reached intermediate waypoint {waypoint_id}")
                
                self.current_waypoint_index += 1
                
                self.last_nav_start_time = None
                self.unknown_status_retries = 0
                self.progress_history = []
                self.distance_improved = False
                
                self.navigate_to_next_waypoint()
                
            elif status == 6:
                if hasattr(self, 'distance_improved') and self.distance_improved:
                    self.get_logger().info(
                        f"Received STATUS_UNKNOWN (6) but making progress toward waypoint. "
                        f"Continuing without resending navigation goal."
                    )
                    return
                    
                self.unknown_status_retries += 1
                
                current_waypoint_id = self.current_path[self.current_waypoint_index]
                is_goal_waypoint = current_waypoint_id in self.goal_waypoints
                
                if self.unknown_status_retries < self.max_unknown_status_retries or is_goal_waypoint:
                    self.get_logger().warn(
                        f"Received STATUS_UNKNOWN (6) - Navigation system may be initializing. "
                        f"Retrying current waypoint (attempt {self.unknown_status_retries})"
                    )
                    
                    import time
                    time.sleep(1.0 * min(self.unknown_status_retries, 3))
                    
                    self.last_nav_start_time = None
                    self.navigate_to_next_waypoint()
                else:
                    self.get_logger().warn(
                        f"Received STATUS_UNKNOWN (6) for {self.max_unknown_status_retries} times. "
                        f"Moving to next waypoint."
                    )
                    self.current_waypoint_index += 1
                    self.unknown_status_retries = 0
                    self.last_nav_start_time = None
                    self.progress_history = []
                    self.distance_improved = False
                    
                    if self.current_waypoint_index < len(self.current_path):
                        self.navigate_to_next_waypoint()
                    else:
                        status_msg = String()
                        status_msg.data = "Path execution completed with some waypoints skipped"
                        self.status_pub.publish(status_msg)
                        self.navigating = False
                
            elif status == 5:
                current_waypoint_id = self.current_path[self.current_waypoint_index]
                
                if current_waypoint_id in self.goal_waypoints:
                    robot_pose = self.get_robot_pose()
                    if robot_pose:
                        waypoint_pos = self.waypoints[current_waypoint_id]
                        distance = self.calculate_distance(robot_pose, waypoint_pos)
                        
                        if distance < 1.5:
                            self.get_logger().info(
                                f"Got ABORTED but close to goal waypoint {current_waypoint_id} (distance={distance:.2f}m). "
                                f"Counting as reached."
                            )
                            idx = self.goal_waypoints.index(current_waypoint_id)
                            status_msg = String()
                            status_msg.data = f"Reached waypoint {current_waypoint_id} ({idx+1}/{len(self.goal_waypoints)})"
                            self.status_pub.publish(status_msg)
                            
                            self.current_waypoint_index += 1
                            self.last_nav_start_time = None
                            self.unknown_status_retries = 0
                            
                            if self.current_waypoint_index < len(self.current_path):
                                self.navigate_to_next_waypoint()
                            else:
                                status_msg = String()
                                status_msg.data = "Path execution completed (close enough to final waypoint)"
                                self.status_pub.publish(status_msg)
                                self.navigating = False
                            return
                    
                    if not hasattr(self, 'aborted_retries'):
                        self.aborted_retries = 0
                        
                    self.aborted_retries += 1
                    if self.aborted_retries < 3:
                        self.get_logger().warn(
                            f"ABORTED for user-requested waypoint {current_waypoint_id}. "
                            f"Retrying (attempt {self.aborted_retries}/3)"
                        )
                        import time
                        time.sleep(2.0)
                        self.navigate_to_next_waypoint()
                        return
                
                self.get_logger().warn("Received ABORTED status, attempting to continue to next waypoint")
                self.current_waypoint_index += 1
                self.last_nav_start_time = None
                self.unknown_status_retries = 0
                if hasattr(self, 'aborted_retries'):
                    self.aborted_retries = 0
                self.progress_history = []
                self.distance_improved = False
                
                if self.current_waypoint_index < len(self.current_path):
                    self.navigate_to_next_waypoint()
                else:
                    status_msg = String()
                    status_msg.data = "Path execution completed with some waypoints skipped"
                    self.status_pub.publish(status_msg)
                    self.navigating = False
                    
            else:
                self.handle_navigation_failure()
        except Exception as e:
            self.get_logger().error(f"Exception in result callback: {e}")
            self.handle_navigation_failure()
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
    
    def handle_navigation_failure(self):
        current_waypoint_id = "unknown"
        if self.current_waypoint_index < len(self.current_path):
            current_waypoint_id = self.current_path[self.current_waypoint_index]
            
        status_msg = String()
        status_msg.data = f"Navigation failed at waypoint {current_waypoint_id}"
        self.status_pub.publish(status_msg)
        
        self.get_logger().warn("Considering whether to continue to next waypoint despite failure")
        
        if current_waypoint_id in self.goal_waypoints:
            self.get_logger().error(f"Failed to reach user-requested waypoint {current_waypoint_id}!")
            
            robot_pose = self.get_robot_pose()
            if robot_pose:
                waypoint_pos = self.waypoints[current_waypoint_id]
                distance = self.calculate_distance(robot_pose, waypoint_pos)
                
                if distance < 2.0:
                    self.get_logger().info(f"Despite failure, robot is close to requested waypoint {current_waypoint_id} (distance={distance:.2f}m)")
                    idx = self.goal_waypoints.index(current_waypoint_id)
                    status_msg = String()
                    status_msg.data = f"Reached waypoint {current_waypoint_id} ({idx+1}/{len(self.goal_waypoints)}) despite failure"
                    self.status_pub.publish(status_msg)
                    
                    self.current_waypoint_index += 1
                    self.last_nav_start_time = None
                    
                    if self.current_waypoint_index < len(self.current_path):
                        self.navigate_to_next_waypoint()
                        return
        
        if self.current_waypoint_index > len(self.current_path) / 2:
            self.get_logger().info("More than halfway through path, attempting to continue to next waypoint")
            self.current_waypoint_index += 1
            self.last_nav_start_time = None
            
            if self.current_waypoint_index < len(self.current_path):
                self.navigate_to_next_waypoint()
                return
        
        self.navigating = False
        self.get_logger().error("Navigation cancelled due to failure")

def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()