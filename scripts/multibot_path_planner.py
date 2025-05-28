#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header, Float32MultiArray
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from master_project2.srv import NavigateToWaypoint
import heapq
import math
import threading
import time
import queue
from rclpy.time import Time
from rclpy.duration import Duration
import numpy as np
import tf2_ros
from tf2_ros import TransformException

try:
    from scipy.spatial import KDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False

class MultibotPathPlanner(Node):
    def __init__(self):
        super().__init__('multibot_path_planner')
        
        self.declare_parameter('base_update_frequency', 1.0)
        self.declare_parameter('path_planning_timeout', 5.0)
        self.declare_parameter('path_simplification_tolerance', 0.1)
        self.declare_parameter('strict_graph_following', True)
        self.declare_parameter('use_fallback_pose', True)
        self.declare_parameter('enable_path_caching', True)
        self.declare_parameter('enable_kdtree', True)
        self.declare_parameter('verbose_amcl_logging', False)
        self.declare_parameter('navigation_timeout', 120.0)
        self.declare_parameter('stuck_robot_threshold', 30)
        
        self.base_update_frequency = self.get_parameter('base_update_frequency').value
        self.path_planning_timeout = self.get_parameter('path_planning_timeout').value
        self.path_simplification_tolerance = self.get_parameter('path_simplification_tolerance').value
        self.strict_graph_following = self.get_parameter('strict_graph_following').value
        self.use_fallback_pose = self.get_parameter('use_fallback_pose').value
        self.enable_path_caching = self.get_parameter('enable_path_caching').value
        self.enable_kdtree = self.get_parameter('enable_kdtree').value and KDTREE_AVAILABLE
        self.verbose_amcl_logging = self.get_parameter('verbose_amcl_logging').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value
        self.stuck_robot_threshold = self.get_parameter('stuck_robot_threshold').value
        
        self.robot_namespace = self.get_namespace().strip('/')
        
        self.status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.config_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.amcl_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.waypoint_graph_sub = self.create_subscription(
            MarkerArray,
            '/waypoint_graph',
            self.waypoint_graph_callback,
            self.config_qos
        )
        
        self.waypoint_costs_sub = self.create_subscription(
            Float32MultiArray,
            '/waypoint_costs',
            self.waypoint_costs_callback,
            self.config_qos
        )
        
        self.nav_service = self.create_service(
            NavigateToWaypoint,
            'navigate_to_waypoint',
            self.navigate_to_waypoint_callback
        )
        
        self.status_pub = self.create_publisher(
            String,
            'path_execution_status',
            self.status_qos
        )
        
        self.path_pub = self.create_publisher(
            Path,
            'planned_path',
            self.status_qos
        )
        
        absolute_amcl_topic = f'/{self.robot_namespace}/amcl_pose'
        
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            absolute_amcl_topic,
            self.amcl_pose_callback,
            self.amcl_qos
        )
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.first_pose_request = True
        self.last_known_good_pose = None
        
        self.waypoints = {}
        self.graph = {}
        self.waypoint_costs = {}
        
        self.graph_update_count = 0
        self.full_graph_processing_interval = 5
        
        if self.enable_path_caching:
            self.path_cache = {}
            self.path_cache_hits = 0
            self.path_cache_misses = 0
        
        if self.enable_kdtree:
            self.waypoint_kdtree = None
            self.kdtree_indices = []
            self.waypoint_kdtree_dirty = True
        
        self.active_task = None
        
        self.current_pose = None
        self.pose_received = False
        self.last_pose_time = time.time()
        
        self._shutdown_flag = threading.Event()
        self._active_navigation_threads = []
        self._thread_lock = threading.Lock()
        
        self.thread_pool_size = 4
        self.thread_pool = []
        self.task_queue = queue.Queue()
        for i in range(self.thread_pool_size):
            thread = threading.Thread(target=self._thread_worker, daemon=True)
            self.thread_pool.append(thread)
            thread.start()
            
        self.received_waypoints = False
        
        self.lock = threading.Lock()
        
        self.cleanup_timer = self.create_timer(10.0, self.cleanup_finished_threads)
        
        if self.verbose_amcl_logging:
            self.amcl_check_timer = self.create_timer(5.0, self.check_amcl_status)
        
        self.resource_cleanup_timer = self.create_timer(60.0, self.cleanup_resources)
        
        try:
            self.robot_id = int(self.robot_namespace.split('_')[1])
        except (IndexError, ValueError):
            self.robot_id = 0
        
        self.robot_default_positions = {
            0: (1.0, 0.0),
            1: (3.0, 3.0),
            2: (0.0, 2.0),
            3: (-2.0, 0.0),
            4: (0.0, -2.0),
        }
        
        self.log_level = 20
        self.log_count = 0
        self.max_logs_per_period = 100
        self.log_period_start = time.time()

        self.problematic_waypoints = set()
    
    def waypoint_costs_callback(self, msg):
        try:
            with self.lock:
                for idx, cost in enumerate(msg.data):
                    if cost >= 0:
                        self.waypoint_costs[str(idx)] = cost
        except Exception as e:
            self.adaptive_log(40, f"Error processing waypoint costs: {e}")
    
    def _thread_worker(self):
        while not self._shutdown_flag.is_set():
            try:
                task, args = self.task_queue.get(timeout=1.0)
                if task:
                    task(*args)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.adaptive_log(40, f"Error in thread worker: {e}")
    
    def adaptive_log(self, level, message):
        if level == 10 and "AMCL" in message and not self.verbose_amcl_logging:
            return
            
        current_time = time.time()
        if current_time - self.log_period_start > 60:
            self.log_count = 0
            self.log_period_start = current_time
            
        if level <= 30:
            if level == 30 and "No AMCL updates" in message and not self.verbose_amcl_logging:
                return
                
            if level == 30:
                self.get_logger().warn(message)
            elif level == 40:
                self.get_logger().error(message)
            elif level == 50:
                self.get_logger().fatal(message)
            return
            
        if self.log_count < self.max_logs_per_period:
            if level == 20:
                self.get_logger().info(message)
            elif level == 10:
                self.get_logger().debug(message)
            self.log_count += 1
        elif self.log_count == self.max_logs_per_period:
            self.get_logger().warn(f"Log rate limiting active - suppressing further logs for this period")
            self.log_count += 1
    
    def check_amcl_status(self):
        if not self.verbose_amcl_logging:
            return
            
        now = time.time()
        if self.current_pose is None:
            self.adaptive_log(30, f"No AMCL pose received yet after {now - self.last_pose_time:.1f} seconds")
            
            try:
                import subprocess
                result = subprocess.run(['ros2', 'topic', 'list'], 
                                        capture_output=True, 
                                        text=True, 
                                        timeout=2.0)
                
                absolute_topic = f"/{self.robot_namespace}/amcl_pose"
                relative_topic = f"{self.robot_namespace}/amcl_pose"
                
                if absolute_topic in result.stdout:
                    self.adaptive_log(20, f"AMCL topic {absolute_topic} exists, but no messages received yet")
                    
                    try:
                        echo_result = subprocess.run(
                            ['ros2', 'topic', 'echo', absolute_topic, '--once'], 
                            capture_output=True, 
                            text=True, 
                            timeout=0.5
                        )
                        if echo_result.stdout.strip():
                            self.adaptive_log(20, f"AMCL topic has data but not reaching callback: {echo_result.stdout[:100]}...")
                    except Exception as e:
                        self.adaptive_log(30, f"Error echoing topic: {e}")
                    
                    try:
                        info_result = subprocess.run(['ros2', 'topic', 'info', absolute_topic],
                                            capture_output=True,
                                            text=True,
                                            timeout=2.0)
                        self.adaptive_log(20, f"AMCL topic info: {info_result.stdout}")
                    except Exception as e:
                        self.adaptive_log(30, f"Error checking topic info: {e}")
                elif relative_topic in result.stdout:
                    self.adaptive_log(20, f"AMCL topic {relative_topic} exists, but no messages received yet")
                else:
                    self.adaptive_log(30, f"AMCL topic not found. Available topics:")
                    self.adaptive_log(30, result.stdout[:500] if len(result.stdout) > 500 else result.stdout)
            except Exception as e:
                self.adaptive_log(40, f"Error checking AMCL topic: {e}")
        else:
            time_since_last = now - self.last_pose_time
            if time_since_last > 5.0:
                self.adaptive_log(30, f"No AMCL updates for {time_since_last:.1f} seconds")
            else:
                self.adaptive_log(10, f"AMCL pose available: ({self.current_pose.pose.pose.position.x:.2f}, {self.current_pose.pose.pose.position.y:.2f})")
    
    def cleanup_resources(self):
        try:
            if self.enable_path_caching and hasattr(self, 'path_cache') and len(self.path_cache) > 50:
                cache_stats = f"hits: {self.path_cache_hits}, misses: {self.path_cache_misses}"
                self.adaptive_log(20, f"Clearing path cache (size: {len(self.path_cache)}, {cache_stats})")
                self.path_cache.clear()
                self.path_cache_hits = 0
                self.path_cache_misses = 0
                
            if self.enable_kdtree and self.waypoint_kdtree_dirty and len(self.waypoints) > 0:
                self.build_kdtree()
                
            if len(self.problematic_waypoints) > 20:
                self.adaptive_log(20, f"Clearing problematic waypoints list (size: {len(self.problematic_waypoints)})")
                self.problematic_waypoints = set(list(self.problematic_waypoints)[-5:])
                
            try:
                import gc
                gc.collect()
            except:
                pass
        except Exception as e:
            self.adaptive_log(40, f"Error in cleanup_resources: {e}")
    
    def _execute_once_and_destroy_timer(self, path, timer):
        try:
            self.destroy_timer(timer)
            self.execute_path_async(path)
        except Exception as e:
            self.adaptive_log(40, f"Error in one-time execution: {e}")
    
    def cleanup_finished_threads(self):
        try:
            with self._thread_lock:
                thread_statuses = []
                for i, thread in enumerate(self._active_navigation_threads):
                    is_alive = thread.is_alive()
                    thread_statuses.append(f"Thread {i}: {'alive' if is_alive else 'dead'}")
                
                if self._active_navigation_threads and self.verbose_amcl_logging:
                    self.adaptive_log(20, f"Thread status: {', '.join(thread_statuses)}")
                
                active_threads = []
                dead_threads = 0
                for thread in self._active_navigation_threads:
                    if thread.is_alive():
                        active_threads.append(thread)
                    else:
                        dead_threads += 1
                
                if dead_threads > 0:
                    self.adaptive_log(20, f"Cleaned up {dead_threads} finished threads, {len(active_threads)} still active")
                    
                self._active_navigation_threads = active_threads
        except Exception as e:
            self.adaptive_log(40, f"Error during thread cleanup: {e}")
    
    def destroy_node(self):
        try:
            self._shutdown_flag.set()
            
            if hasattr(self, 'active_task') and self.active_task:
                try:
                    self.active_task.cancel()
                except Exception as e:
                    self.adaptive_log(40, f"Error cancelling active task during shutdown: {e}")
            
            try:
                self.task_queue.join(timeout=1.0)
            except:
                pass
                
            time.sleep(0.2)
        except Exception as e:
            self.adaptive_log(40, f"Error during node shutdown: {e}")
        
        super().destroy_node()
    
    def amcl_pose_callback(self, msg):
        try:
            if not self.pose_received:
                self.get_logger().info(f"✅ RECEIVED FIRST AMCL POSE: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})")
                self.pose_received = True
                
            self.current_pose = msg
            self.last_pose_time = time.time()
            
            self.last_known_good_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            
            if self.verbose_amcl_logging:
                self.adaptive_log(10, f"Updated robot pose from AMCL: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")
        except Exception as e:
            self.adaptive_log(40, f"Error in AMCL pose callback: {e}")
    
    def build_kdtree(self):
        if not self.enable_kdtree or not KDTREE_AVAILABLE:
            return
            
        try:
            if len(self.waypoints) > 0:
                points = []
                self.kdtree_indices = []
                for waypoint_id, (x, y) in self.waypoints.items():
                    points.append([x, y])
                    self.kdtree_indices.append(waypoint_id)
                
                self.waypoint_kdtree = KDTree(points)
                self.waypoint_kdtree_dirty = False
                self.adaptive_log(20, f"KD-tree rebuilt with {len(points)} points")
        except Exception as e:
            self.adaptive_log(40, f"Error building KD-tree: {e}")
            self.waypoint_kdtree = None
    
    def waypoint_graph_callback(self, msg):
        try:
            self.graph_update_count += 1
            
            marker_count = len([m for m in msg.markers if m.ns == "waypoints"])
            if self.received_waypoints and marker_count == len(self.waypoints):
                if self.graph_update_count % 10 == 0:
                    self.adaptive_log(10, f"Received waypoint update with same count ({marker_count}), skipping full reprocessing")
                return
                
            if not self.received_waypoints or abs(marker_count - len(self.waypoints)) > 10 or self.graph_update_count % self.full_graph_processing_interval == 0:
                self.adaptive_log(20, f"Full graph processing (update #{self.graph_update_count}, {marker_count} markers)")
                
                with self.lock:
                    old_waypoint_count = len(self.waypoints)
                    
                    for marker in msg.markers:
                        if marker.ns == "waypoints" and marker.type == Marker.SPHERE:
                            waypoint_id = str(marker.id)
                            x = marker.pose.position.x
                            y = marker.pose.position.y
                            
                            self.waypoints[waypoint_id] = (x, y)
                            
                            if waypoint_id not in self.graph:
                                self.graph[waypoint_id] = []
                    
                    edge_pairs = set()
                    for marker in msg.markers:
                        if marker.ns == "edges" and marker.type == Marker.LINE_STRIP:
                            if len(marker.points) == 2:
                                start_point = (marker.points[0].x, marker.points[0].y)
                                end_point = (marker.points[1].x, marker.points[1].y)
                                
                                start_id = self.find_closest_waypoint(start_point)
                                end_id = self.find_closest_waypoint(end_point)
                                
                                if start_id and end_id:
                                    if end_id not in self.graph.get(start_id, []):
                                        self.graph.setdefault(start_id, []).append(end_id)
                                    if start_id not in self.graph.get(end_id, []):
                                        self.graph.setdefault(end_id, []).append(start_id)
                                    
                                    edge_pairs.add(tuple(sorted([start_id, end_id])))
                    
                    if old_waypoint_count != len(self.waypoints):
                        self.waypoint_kdtree_dirty = True
                    
                    if self.enable_path_caching and hasattr(self, 'path_cache') and len(self.path_cache) > 0:
                        if abs(old_waypoint_count - len(self.waypoints)) > 5:
                            self.adaptive_log(20, f"Waypoint count changed significantly ({old_waypoint_count} → {len(self.waypoints)}), clearing path cache")
                            self.path_cache.clear()
                    
                    if len(self.waypoints) > 0 and not self.received_waypoints:
                        self.received_waypoints = True
                        self.adaptive_log(20, f'Processed initial waypoint graph with {len(self.waypoints)} waypoints and {len(edge_pairs)} edges')
                        
                        isolated_waypoints = 0
                        for waypoint_id in self.waypoints:
                            if waypoint_id not in self.graph or not self.graph[waypoint_id]:
                                isolated_waypoints += 1
                        
                        if isolated_waypoints > 0:
                            self.adaptive_log(30, f'Found {isolated_waypoints} isolated waypoints with no connections')
                        
                        if self.waypoints:
                            total_connections = sum(len(neighbors) for neighbors in self.graph.values())
                            avg_connections = total_connections / len(self.waypoints)
                            self.adaptive_log(20, f'Average waypoint has {avg_connections:.2f} connections')
            else:
                self.adaptive_log(10, f"Skipping full graph processing (update #{self.graph_update_count})")
                
        except Exception as e:
            self.adaptive_log(40, f"Error in waypoint graph callback: {e}")
    
    def wait_for_amcl_pose(self, timeout=10.0):
        if self.current_pose is not None:
            return True
            
        self.adaptive_log(20, f"Waiting for AMCL pose with timeout of {timeout} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.current_pose is not None:
                self.adaptive_log(20, f"Received AMCL pose after {time.time() - start_time:.2f} seconds")
                return True
            time.sleep(0.5)
            
        self.adaptive_log(30, f"Timed out waiting for AMCL pose after {timeout} seconds")
        return False

    def get_robot_pose(self):
        try:
            if self.current_pose is not None:
                x = self.current_pose.pose.pose.position.x
                y = self.current_pose.pose.pose.position.y
                self.adaptive_log(10, f"Using AMCL pose: ({x}, {y})")
                self.first_pose_request = False
                return (x, y)
                
            try:
                now = rclpy.time.Time()
                map_frame = "map"
                robot_frame = f"{self.robot_namespace}/base_footprint"
                
                trans = self.tf_buffer.lookup_transform(
                    map_frame,
                    robot_frame,
                    now,
                    timeout=Duration(seconds=1.0)
                )
                
                x = trans.transform.translation.x
                y = trans.transform.translation.y
                self.adaptive_log(20, f"Using TF2 pose: ({x}, {y})")
                
                self.last_known_good_pose = (x, y)
                self.first_pose_request = False
                return (x, y)
            except TransformException as ex:
                self.adaptive_log(20, f"TF lookup failed: {ex}")
            
            if self.last_known_good_pose is not None:
                self.adaptive_log(30, f"Using last known good pose: {self.last_known_good_pose}")
                return self.last_known_good_pose
                
            if self.first_pose_request:
                self.adaptive_log(30, "First pose request, using default starting position")
                self.first_pose_request = False
                fallback_pos = self.robot_default_positions.get(self.robot_id, (0.0, 0.0))
                return fallback_pos
            
            if self.use_fallback_pose:
                fallback_pos = self.robot_default_positions.get(self.robot_id, (0.0, 0.0))
                self.adaptive_log(30, f"All fallbacks failed, using static position: {fallback_pos}")
                return fallback_pos
            else:
                self.adaptive_log(30, "No position available and fallback disabled")
                return None
        except Exception as e:
            self.adaptive_log(40, f"Error getting robot pose: {e}")
            if self.last_known_good_pose is not None:
                return self.last_known_good_pose
            if self.use_fallback_pose:
                fallback_pos = self.robot_default_positions.get(self.robot_id, (0.0, 0.0))
                self.adaptive_log(30, f"Error getting pose, using emergency fallback: {fallback_pos}")
                return fallback_pos
            return None

    def find_closest_waypoint(self, point):
        try:
            if self.enable_kdtree and KDTREE_AVAILABLE and len(self.waypoints) > 10:
                if self.waypoint_kdtree is None or self.waypoint_kdtree_dirty:
                    self.build_kdtree()
                
                if self.waypoint_kdtree is not None:
                    dist, idx = self.waypoint_kdtree.query([point[0], point[1]], k=1)
                    closest_id = self.kdtree_indices[idx]
                    
                    if dist > 3.0:
                        self.adaptive_log(30, f"Closest waypoint ({closest_id}) is {dist:.2f}m away - unusually distant")
                    
                    return closest_id
            
            min_dist = float('inf')
            closest_id = None
            
            for waypoint_id, (x, y) in self.waypoints.items():
                dist = self.calculate_distance(point, (x, y))
                if dist < min_dist:
                    min_dist = dist
                    closest_id = waypoint_id
            
            if min_dist > 3.0:
                self.adaptive_log(30, f"Closest waypoint ({closest_id}) is {min_dist:.2f}m away - unusually distant")
            
            return closest_id
        except Exception as e:
            self.adaptive_log(40, f"Error finding closest waypoint: {e}")
            return None

    def calculate_distance(self, point1, point2):
        try:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except Exception as e:
            self.adaptive_log(40, f"Error calculating distance: {e}")
            return float('inf')
    
    def verify_path_follows_graph(self, path):
        try:
            if len(path) < 2:
                return True
                
            for i in range(len(path) - 1):
                current = path[i]
                next_wp = path[i+1]
                
                if next_wp not in self.graph.get(current, []):
                    self.adaptive_log(40, f"Path does not follow graph: No edge between {current} and {next_wp}")
                    return False
            
            self.adaptive_log(20, "Path verification successful: All segments follow graph edges")
            return True
        except Exception as e:
            self.adaptive_log(40, f"Error verifying path: {e}")
            return False
    
    def simplify_path(self, path, tolerance=None):
        try:
            if tolerance is None:
                tolerance = self.path_simplification_tolerance
                
            if len(path) <= 2:
                return path
            
            if self.strict_graph_following and tolerance < 0.05:
                self.adaptive_log(20, "Strict graph following enabled with minimal tolerance - skipping simplification")
                return path
                
            result = [path[0]]
            
            for i in range(1, len(path)-1):
                prev_wp = self.waypoints[result[-1]]
                curr_wp = self.waypoints[path[i]]
                next_wp = self.waypoints[path[i+1]]
                
                line_dist = self.point_line_distance(curr_wp, prev_wp, next_wp)
                
                if line_dist > tolerance:
                    result.append(path[i])
                else:
                    if self.strict_graph_following:
                        prev_id = result[-1]
                        next_id = path[i+1]
                        
                        if next_id not in self.graph.get(prev_id, []):
                            result.append(path[i])
                            self.adaptive_log(10, f"Keeping waypoint {path[i]} to preserve graph connectivity")
            
            result.append(path[-1])
            
            if self.strict_graph_following:
                for i in range(len(result) - 1):
                    if result[i+1] not in self.graph.get(result[i], []):
                        self.adaptive_log(40, f"Simplified path would break graph connectivity at {result[i]} -> {result[i+1]}")
                        return path
            
            self.adaptive_log(20, f"Simplified path from {len(path)} to {len(result)} waypoints")
            return result
        except Exception as e:
            self.adaptive_log(40, f"Error simplifying path: {e}")
            return path

    def point_line_distance(self, point, line_start, line_end):
        try:
            p = np.array([point[0], point[1]])
            s = np.array([line_start[0], line_start[1]])
            e = np.array([line_end[0], line_end[1]])
            
            line_vec = e - s
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                return np.linalg.norm(p - s)
            
            line_unit_vec = line_vec / line_len
            
            point_vec = p - s
            
            proj_len = np.dot(point_vec, line_unit_vec)
            
            proj_point = s + line_unit_vec * proj_len
            
            return np.linalg.norm(p - proj_point)
        except Exception as e:
            self.adaptive_log(40, f"Error calculating point-line distance: {e}")
            return float('inf')
    
    def plan_path(self, start_waypoint_id, goal_waypoint_id):
        try:
            if start_waypoint_id not in self.waypoints or goal_waypoint_id not in self.waypoints:
                self.adaptive_log(40, f'Invalid waypoints: start={start_waypoint_id}, goal={goal_waypoint_id}')
                return None
            
            if self.enable_path_caching and hasattr(self, 'path_cache'):
                cache_key = f"{start_waypoint_id}_{goal_waypoint_id}"
                if cache_key in self.path_cache:
                    cached_path = self.path_cache[cache_key]
                    self.adaptive_log(20, f"Using cached path from {start_waypoint_id} to {goal_waypoint_id}")
                    self.path_cache_hits += 1
                    return cached_path.copy()
                self.path_cache_misses += 1
            
            start_pos = self.waypoints[start_waypoint_id]
            goal_pos = self.waypoints[goal_waypoint_id]
            direct_distance = self.calculate_distance(start_pos, goal_pos)
            self.adaptive_log(20, f"Planning path from waypoint {start_waypoint_id} to {goal_waypoint_id} (direct distance: {direct_distance:.2f}m)")
            
            path_result = [None]
            
            def planning_thread():
                try:
                    open_set = []
                    closed_set = set()
                    
                    g_score = {start_waypoint_id: 0}
                    f_score = {start_waypoint_id: self.heuristic(start_waypoint_id, goal_waypoint_id)}
                    
                    came_from = {}
                    
                    heapq.heappush(open_set, (f_score[start_waypoint_id], start_waypoint_id))
                    
                    nodes_expanded = 0
                    
                    while open_set and not self._shutdown_flag.is_set():
                        nodes_expanded += 1
                        if nodes_expanded > 10000:
                            self.adaptive_log(30, f"A* path planning reached node expansion limit")
                            break
                            
                        _, current_id = heapq.heappop(open_set)
                        
                        if current_id == goal_waypoint_id:
                            path = []
                            while current_id in came_from:
                                path.append(current_id)
                                current_id = came_from[current_id]
                            path.append(start_waypoint_id)
                            path.reverse()
                            path_result[0] = path
                            return
                        
                        closed_set.add(current_id)
                        
                        for neighbor_id in self.graph.get(current_id, []):
                            if neighbor_id in closed_set:
                                continue
                            
                            tentative_g_score = g_score[current_id] + self.edge_cost(current_id, neighbor_id)
                            
                            if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                                came_from[neighbor_id] = current_id
                                g_score[neighbor_id] = tentative_g_score
                                f_score[neighbor_id] = tentative_g_score + self.heuristic(neighbor_id, goal_waypoint_id)
                                
                                in_open_set = False
                                for idx, (_, wp_id) in enumerate(open_set):
                                    if wp_id == neighbor_id:
                                        in_open_set = True
                                        break
                                
                                if not in_open_set:
                                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
                    
                    self.adaptive_log(40, f'No path found from {start_waypoint_id} to {goal_waypoint_id}')
                    path_result[0] = None
                except Exception as e:
                    self.adaptive_log(40, f'Exception in path planning: {e}')
                    path_result[0] = None
            
            planning_thread = threading.Thread(target=planning_thread)
            planning_thread.daemon = True
            planning_thread.start()
            
            planning_thread.join(timeout=self.path_planning_timeout)
            
            if planning_thread.is_alive():
                self.adaptive_log(40, f'Path planning timeout after {self.path_planning_timeout} seconds')
                return None
            
            if path_result[0] and not self.verify_path_follows_graph(path_result[0]):
                self.adaptive_log(40, "Path validation failed: path does not follow graph edges")
                return None
                
            if path_result[0]:
                self.adaptive_log(20, f"Found path with {len(path_result[0])} waypoints")
                
                if self.enable_path_caching and hasattr(self, 'path_cache'):
                    cache_key = f"{start_waypoint_id}_{goal_waypoint_id}"
                    self.path_cache[cache_key] = path_result[0].copy()
                    
                    if len(self.path_cache) > 100:
                        self.adaptive_log(20, "Clearing path cache (limit reached)")
                        self.path_cache.clear()
                
            return path_result[0]
        except Exception as e:
            self.adaptive_log(40, f"Error in plan_path: {e}")
            return None

    def heuristic(self, waypoint_id, goal_id):
        try:
            if waypoint_id not in self.waypoints or goal_id not in self.waypoints:
                return float('inf')
            
            x1, y1 = self.waypoints[waypoint_id]
            x2, y2 = self.waypoints[goal_id]
            
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except Exception as e:
            self.adaptive_log(40, f"Error calculating heuristic: {e}")
            return float('inf')

    def edge_cost(self, waypoint1_id, waypoint2_id):
        try:
            if waypoint1_id not in self.waypoints or waypoint2_id not in self.waypoints:
                return float('inf')
            
            x1, y1 = self.waypoints[waypoint1_id]
            x2, y2 = self.waypoints[waypoint2_id]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            wp1_cost = self.waypoint_costs.get(waypoint1_id, 1.0)
            wp2_cost = self.waypoint_costs.get(waypoint2_id, 1.0)
            cost_factor = (wp1_cost + wp2_cost) / 2.0
            
            return distance * cost_factor
        except Exception as e:
            self.adaptive_log(40, f"Error calculating edge cost: {e}")
            return float('inf')
    
    def navigate_to_waypoint_callback(self, request, response):
        try:
            self.adaptive_log(20, f'Received navigation request: {request.waypoint_id}')
            
            if not self.received_waypoints:
                timeout = 5.0
                start_time = time.time()
                while not self.received_waypoints and time.time() - start_time < timeout:
                    self.adaptive_log(20, "Waiting for waypoint data...")
                    time.sleep(0.5)
                
                if not self.received_waypoints:
                    self.adaptive_log(40, "Timed out waiting for waypoint data")
                    response.success = False
                    response.message = "Timed out waiting for waypoint data"
                    return response
            
            if ',' in request.waypoint_id:
                waypoints = [wp.strip() for wp in request.waypoint_id.split(',')]
                self.adaptive_log(20, f'Planning path through multiple waypoints: {waypoints}')
                
                if request.start_waypoint_id:
                    start_waypoint_id = request.start_waypoint_id
                    self.adaptive_log(20, f'Using specified start waypoint: {start_waypoint_id}')
                else:
                    robot_pose = self.get_robot_pose()
                    if robot_pose:
                        start_waypoint_id = self.find_closest_waypoint(robot_pose)
                        self.adaptive_log(20, f'Using closest waypoint as start: {start_waypoint_id}')
                    else:
                        self.adaptive_log(40, 'Cannot determine robot pose for start waypoint')
                        response.success = False
                        response.message = 'Failed to determine start waypoint (robot pose unknown)'
                        return response
                
                for waypoint in waypoints + [start_waypoint_id]:
                    if waypoint not in self.waypoints:
                        self.adaptive_log(40, f'Waypoint {waypoint} not found in graph')
                        response.success = False
                        response.message = f'Waypoint {waypoint} not found in graph'
                        return response
                
                filtered_waypoints = []
                for wp in waypoints:
                    if wp in self.problematic_waypoints:
                        self.adaptive_log(30, f"Skipping known problematic waypoint {wp}")
                    else:
                        filtered_waypoints.append(wp)
                
                if not filtered_waypoints:
                    self.adaptive_log(40, "All requested waypoints are problematic")
                    response.success = False
                    response.message = "All requested waypoints are problematic and were filtered out"
                    return response
                
                try:
                    full_path = [start_waypoint_id]
                    for i in range(len(filtered_waypoints)):
                        segment_start = full_path[-1]
                        segment_goal = filtered_waypoints[i]
                        
                        self.adaptive_log(20, f'Planning path segment from {segment_start} to {segment_goal}')
                        segment_path = self.plan_path(segment_start, segment_goal)
                        if segment_path is None:
                            self.adaptive_log(40, f'Failed to plan path segment from {segment_start} to {segment_goal}')
                            response.success = False
                            response.message = f'Failed to plan path segment from {segment_start} to {segment_goal}'
                            return response
                        
                        full_path.extend(segment_path[1:])
                    
                    if not self.verify_path_follows_graph(full_path):
                        self.adaptive_log(40, "Full path does not follow graph edges")
                        response.success = False
                        response.message = "Failed to create a valid path that follows the graph"
                        return response
                    
                    simplified_path = self.simplify_path(full_path)
                    
                    if not self.verify_path_follows_graph(simplified_path):
                        self.adaptive_log(40, "Simplified path does not follow graph edges - using original path")
                        simplified_path = full_path
                    
                    response.success = True
                    response.message = f'Started navigation through {len(simplified_path)} waypoints'
                    
                    timer = self.create_timer(0.1, lambda: self._execute_once_and_destroy_timer(simplified_path, timer))
                    
                    return response
                    
                except Exception as e:
                    self.adaptive_log(40, f'Exception in path planning: {e}')
                    response.success = False
                    response.message = f'Error planning path: {str(e)}'
                    return response
            else:
                target_waypoint_id = request.waypoint_id
                
                if target_waypoint_id in self.problematic_waypoints:
                    self.adaptive_log(30, f"Target waypoint {target_waypoint_id} is known to be problematic")
                    response.success = False
                    response.message = f"Target waypoint {target_waypoint_id} is known to be problematic"
                    return response
                
                if target_waypoint_id not in self.waypoints:
                    self.adaptive_log(40, f'Waypoint {target_waypoint_id} not found in graph')
                    response.success = False
                    response.message = f'Waypoint {target_waypoint_id} not found in graph'
                    return response
                
                if request.start_waypoint_id:
                    start_waypoint_id = request.start_waypoint_id
                    if start_waypoint_id not in self.waypoints:
                        self.adaptive_log(40, f'Start waypoint {start_waypoint_id} not found in graph')
                        response.success = False
                        response.message = f'Start waypoint {start_waypoint_id} not found in graph'
                        return response
                else:
                    robot_pose = self.get_robot_pose()
                    if robot_pose:
                        start_waypoint_id = self.find_closest_waypoint(robot_pose)
                        self.adaptive_log(20, f'Using closest waypoint as start: {start_waypoint_id}')
                    else:
                        self.adaptive_log(40, 'Cannot determine robot pose for start waypoint')
                        response.success = False
                        response.message = 'Failed to determine start waypoint (robot pose unknown)'
                        return response
                
                try:
                    self.adaptive_log(20, f'Planning path from {start_waypoint_id} to {target_waypoint_id}')
                    path = self.plan_path(start_waypoint_id, target_waypoint_id)
                    if path is None:
                        self.adaptive_log(40, f'Failed to plan path from {start_waypoint_id} to {target_waypoint_id}')
                        response.success = False
                        response.message = f'Failed to plan path from {start_waypoint_id} to {target_waypoint_id}'
                        return response
                    
                    if not self.verify_path_follows_graph(path):
                        self.adaptive_log(40, "Path does not follow graph edges")
                        response.success = False
                        response.message = "Failed to create a valid path that follows the graph"
                        return response
                    
                    simplified_path = self.simplify_path(path)
                    
                    if not self.verify_path_follows_graph(simplified_path):
                        self.adaptive_log(40, "Simplified path does not follow graph edges - using original path")
                        simplified_path = path
                    
                    response.success = True
                    response.message = f'Started navigation through {len(simplified_path)} waypoints'
                    
                    timer = self.create_timer(0.1, lambda: self._execute_once_and_destroy_timer(simplified_path, timer))
                    
                    return response
                    
                except Exception as e:
                    self.adaptive_log(40, f'Exception in path planning: {e}')
                    response.success = False
                    response.message = f'Error planning path: {str(e)}'
                    return response
        except Exception as e:
            self.adaptive_log(40, f"Error in navigate_to_waypoint_callback: {e}")
            response.success = False
            response.message = f"Service error: {str(e)}"
            return response
    
    def execute_path_async(self, path):
        try:
            if not path or len(path) < 2:
                self.publish_status('Path too short or empty')
                return
            
            if not self.verify_path_follows_graph(path):
                self.adaptive_log(40, "Path does not follow graph edges - aborting execution")
                self.publish_status('Path does not follow graph edges - navigation aborted')
                return
            
            self.adaptive_log(20, f'Executing path asynchronously: {path}')
            
            self.visualize_path(path)
            
            with self._thread_lock:
                if self.active_task:
                    try:
                        self.active_task.cancel()
                        time.sleep(0.2)
                    except Exception as e:
                        self.adaptive_log(40, f"Error cancelling previous task: {e}")
            
            try:
                self.active_task = NavigationTask(
                    self, 
                    path, 
                    self._shutdown_flag, 
                    self.navigation_timeout,
                    self.stuck_robot_threshold,
                    self.problematic_waypoints
                )
                
                thread = threading.Thread(target=self.active_task.execute_path, daemon=True)
                thread.name = f"navigation-thread-{time.time()}"
                
                with self._thread_lock:
                    self._active_navigation_threads = [t for t in self._active_navigation_threads if t.is_alive()]
                    self._active_navigation_threads.append(thread)
                
                thread.start()
                
                self.publish_status(f'Started navigation through {len(path)} waypoints')
            except Exception as e:
                self.adaptive_log(40, f"Error creating navigation task: {e}")
                self.publish_status(f'Error creating navigation task: {str(e)}')
        except Exception as e:
            self.adaptive_log(40, f'Error executing path: {e}')
            self.publish_status(f'Error executing path: {str(e)}')

    def visualize_path(self, waypoint_path):
        try:
            if not waypoint_path:
                return
            
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            for waypoint_id in waypoint_path:
                if waypoint_id in self.waypoints:
                    x, y = self.waypoints[waypoint_id]
                    
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    
                    path_msg.poses.append(pose)
            
            self.path_pub.publish(path_msg)
            self.adaptive_log(20, f'Published path visualization with {len(path_msg.poses)} poses')
        except Exception as e:
            self.adaptive_log(40, f'Error visualizing path: {e}')

    def publish_status(self, status_msg):
        try:
            msg = String()
            msg.data = status_msg
            self.status_pub.publish(msg)
            self.adaptive_log(20, f'Status: {status_msg}')
        except Exception as e:
            self.adaptive_log(40, f'Error publishing status: {e}')


class NavigationTask:
    def __init__(self, planner_node, waypoint_path, shutdown_flag=None, navigation_timeout=1800.0, max_stuck_seconds=30, problematic_waypoints=None):
        self.planner = planner_node
        self.waypoint_path = waypoint_path
        self.current_waypoint_index = 0
        self.cancelled = False
        self.shutdown_flag = shutdown_flag if shutdown_flag else threading.Event()
        self.navigation_timeout = navigation_timeout
        self.max_stuck_seconds = max_stuck_seconds
        self.problematic_waypoints = problematic_waypoints if problematic_waypoints else set()
        
        self.nav_client = ActionClient(
            planner_node, 
            NavigateToPose, 
            'navigate_to_pose'
        )
    
    def cancel(self):
        try:
            self.cancelled = True
            self.planner.adaptive_log(20, "Navigation task cancelled")
        except Exception as e:
            try:
                self.planner.adaptive_log(40, f"Error cancelling navigation: {e}")
            except:
                pass
    
    def safely_monitor_goal(self, goal_handle, waypoint_id):
        try:
            result_future = goal_handle.get_result_async()
            
            start_time = time.time()
            last_position = self.planner.get_robot_pose()
            stuck_count = 0
            cancel_sent = False
            
            while rclpy.ok() and not self.cancelled and not self.shutdown_flag.is_set():
                try:
                    if result_future.done():
                        try:
                            result = result_future.result()
                            success = result and result.status == 4
                            
                            if success:
                                self.planner.publish_status(f'Successfully reached waypoint {waypoint_id}')
                            else:
                                status_code = result.status if result else "unknown"
                                self.planner.publish_status(f'Navigation to waypoint {waypoint_id} did not complete successfully, continuing to next waypoint')
                                
                            return success
                        except Exception as e:
                            self.planner.adaptive_log(40, f"Error processing result: {e}")
                            return False
                    
                    rclpy.spin_once(self.planner, timeout_sec=0.1)
                    
                    current_position = self.planner.get_robot_pose()
                    if not current_position:
                        continue
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.navigation_timeout:
                        self.planner.adaptive_log(30, f"Navigation timeout for waypoint {waypoint_id}")
                        self.safely_cancel_goal(goal_handle)
                        return False
                    
                    if last_position:
                        try:
                            distance = math.sqrt((current_position[0] - last_position[0])**2 + 
                                               (current_position[1] - last_position[1])**2)
                            
                            if distance < 0.05:
                                stuck_count += 1
                                
                                if stuck_count > self.max_stuck_seconds:
                                    self.planner.adaptive_log(30, f"Robot stuck at waypoint {waypoint_id} for over {stuck_count} seconds")
                                    self.planner.publish_status(f"Skipping stuck waypoint {waypoint_id} after {stuck_count} seconds")
                                    
                                    self.problematic_waypoints.add(waypoint_id)
                                    
                                    self.safely_cancel_goal(goal_handle)
                                    return False
                            else:
                                stuck_count = 0
                        except Exception as e:
                            self.planner.adaptive_log(40, f"Error checking stuck status: {e}")
                    
                    last_position = current_position
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.planner.adaptive_log(40, f"Error in monitoring loop: {e}")
                    time.sleep(0.5)
                    
            self.safely_cancel_goal(goal_handle)
            return False
            
        except Exception as e:
            self.planner.adaptive_log(40, f"Critical error in goal monitoring: {e}")
            try:
                self.safely_cancel_goal(goal_handle)
            except:
                pass
            return False
        
    def safely_cancel_goal(self, goal_handle):
        try:
            goal_handle.cancel_goal_async()
        except Exception as e:
            self.planner.adaptive_log(40, f"Error cancelling goal: {e}")
    
    def navigate_to_single_waypoint(self, waypoint_id, waypoint_idx):
        try:
            if waypoint_id not in self.planner.waypoints:
                self.planner.publish_status(f'Invalid waypoint {waypoint_id}')
                return False
                
            x, y = self.planner.waypoints[waypoint_id]
            self.planner.publish_status(f'Navigating to waypoint {waypoint_id} ({waypoint_idx+1}/{len(self.waypoint_path)})')
            
            server_ready = self.nav_client.wait_for_server(timeout_sec=2.0)
            if not server_ready:
                self.planner.adaptive_log(30, f"Navigation server not available for waypoint {waypoint_id}")
                return False
            
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.planner.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.position.z = 0.0
            goal_msg.pose.pose.orientation.w = 1.0
            
            try:
                send_goal_future = self.nav_client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self.planner, send_goal_future, timeout_sec=5.0)
                
                if not send_goal_future.done():
                    self.planner.adaptive_log(30, f"Goal sending timed out for waypoint {waypoint_id}")
                    return False
                    
                goal_handle = send_goal_future.result()
                if not goal_handle or not goal_handle.accepted:
                    self.planner.adaptive_log(30, f"Goal rejected for waypoint {waypoint_id}")
                    return False
                    
                return self.safely_monitor_goal(goal_handle, waypoint_id)
                
            except Exception as e:
                self.planner.adaptive_log(40, f"Error sending goal for waypoint {waypoint_id}: {e}")
                return False
                
        except Exception as e:
            self.planner.adaptive_log(40, f"Unexpected error handling waypoint {waypoint_id}: {e}")
            return False
    
    def execute_path(self):
        try:
            self.planner.adaptive_log(20, f"Starting execution of path with {len(self.waypoint_path)} waypoints")
            
            waypoint_idx = 0
            while waypoint_idx < len(self.waypoint_path) and not self.cancelled and not self.shutdown_flag.is_set():
                try:
                    waypoint_id = self.waypoint_path[waypoint_idx]
                    self.planner.adaptive_log(20, f"Processing waypoint {waypoint_id} ({waypoint_idx+1}/{len(self.waypoint_path)})")
                    
                    if waypoint_id in self.problematic_waypoints:
                        self.planner.adaptive_log(30, f"Skipping known problematic waypoint {waypoint_id}")
                        self.planner.publish_status(f"Skipping known problematic waypoint {waypoint_id}")
                        waypoint_idx += 1
                        continue
                    
                    success = self.navigate_to_single_waypoint(waypoint_id, waypoint_idx)
                    
                    waypoint_idx += 1
                    
                except Exception as e:
                    self.planner.adaptive_log(40, f"Error processing waypoint {self.waypoint_path[waypoint_idx]}: {e}")
                    waypoint_idx += 1
                    time.sleep(1.0)
            
            if self.cancelled or self.shutdown_flag.is_set():
                self.planner.publish_status('Navigation cancelled')
            else:
                self.planner.publish_status(f'Completed navigation through {len(self.waypoint_path)} waypoints')
                
        except Exception as e:
            try:
                self.planner.adaptive_log(40, f"Critical navigation error: {str(e)}")
                self.planner.publish_status(f"Navigation error: {str(e)}")
            except:
                pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MultibotPathPlanner()
        
        try:
            while rclpy.ok():
                try:
                    rclpy.spin_once(node, timeout_sec=0.1)
                except Exception as e:
                    node.get_logger().error(f"Error during spin_once: {e}")
                    time.sleep(0.5)
        except KeyboardInterrupt:
            node.adaptive_log(20, "Node stopped cleanly by user")
        finally:
            node.destroy_node()
    except Exception as e:
        print(f"Critical error during node initialization: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()