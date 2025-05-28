#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from master_project2.srv import NavigateToWaypoint
import sys
import time
import re

class MultibotPathPlannerTester(Node):
    def __init__(self):
        super().__init__('path_planner_tester')
        
        self.robot_ns, self.waypoints_list = self.parse_args()
        self.get_logger().info(f"Path Planner Tester started for robot: {self.robot_ns}")
        if self.waypoints_list:
            self.get_logger().info(f"Will navigate to waypoints: {self.waypoints_list}")
        else:
            self.get_logger().error("No waypoints specified - cannot navigate!")
        
        self.nav_client = self.create_client(
            NavigateToWaypoint, 
            f'/{self.robot_ns}/navigate_to_waypoint'
        )
        
        self.get_logger().info(f"Creating service client for: /{self.robot_ns}/navigate_to_waypoint")
        
        self.status_sub = self.create_subscription(
            String,
            f'/{self.robot_ns}/path_execution_status',
            self.status_callback,
            10
        )
        self.latest_status = ""
        
        self.mission_complete_pub = self.create_publisher(
            String, 
            f'/{self.robot_ns}/mission_complete', 
            10
        )
        
        self.waypoints = set()
        self.waypoint_sub = self.create_subscription(
            MarkerArray,
            'waypoint_graph',  
            self.waypoint_callback,
            10
        )
        
        self.waypoints_received = False
        
        self.navigation_in_progress = False
        
        self.service_call_in_progress = False
        
        self.current_waypoint_index = 0
        self.successful_waypoints = 0
        
        self.create_timer(1.0, self.check_service_and_start)
        self.service_ready = False
        self.mission_started = False
        
        self.create_timer(15.0, self.navigation_watchdog)
        self.last_status_time = time.time()
        self.navigation_timeout = 120.0  
        
        self.prev_statuses = set()

        self.connection_attempts = 0
        self.max_connection_attempts = 10
        
    def parse_args(self):
        robot_namespace = "robot_0"
        waypoint_list = []
        
        args = [arg for arg in sys.argv[1:] if not (arg.startswith('--') or ':=' in arg)]
        self.get_logger().info(f"Command line arguments: {args}")
        
        if len(args) >= 2:
            if args[0].startswith('robot_'):
                robot_namespace = args[0]
                
                if ',' in args[1]:
                    waypoint_list = args[1].split(',')
                else:
                    waypoint_list = [args[1]]
            else:
                self.get_logger().warn(f"First argument '{args[0]}' is not a valid robot namespace")
                waypoint_list = args
        elif len(args) == 1:
            if args[0].startswith('robot_'):
                robot_namespace = args[0]
                self.get_logger().warn("No waypoints specified")
            else:
                waypoint_list = [args[0]]
        else:
            self.get_logger().warn("No arguments provided, using defaults")
        
        return robot_namespace, waypoint_list
        
    def waypoint_callback(self, msg):
        waypoints_markers = [m for m in msg.markers if m.ns == "waypoints"]
        self.waypoints = {str(marker.id) for marker in waypoints_markers}
        
        if self.waypoints and not self.waypoints_received:
            self.waypoints_received = True
            self.get_logger().info(f'Received waypoint graph with {len(self.waypoints)} waypoints')
            try:
                sorted_waypoints = sorted([int(wp) for wp in self.waypoints])
                self.get_logger().info(f'Available waypoints: {sorted_waypoints[:10]}... (showing first 10 of {len(sorted_waypoints)})')
            except ValueError:
                self.get_logger().info(f'Available waypoints: {sorted(list(self.waypoints))}')
    
    def status_callback(self, msg):
        self.last_status_time = time.time()
        
        self.latest_status = msg.data
        
        status_key = msg.data
        if status_key not in self.prev_statuses:
            self.prev_statuses.add(status_key)
            self.get_logger().info(f"Path execution status for {self.robot_ns}: {msg.data}")
        
        if self.navigation_in_progress:
            if "completed" in self.latest_status.lower() and "successfully" in self.latest_status.lower():
                self.get_logger().info(f"Final goal reached! Mission completed successfully.")
                self.navigation_in_progress = False
                self.publish_mission_complete(success=True)
            elif "reached waypoint" in self.latest_status.lower():
                match = re.search(r'waypoint (\d+)', self.latest_status)
                if match:
                    waypoint = match.group(1)
                    self.get_logger().info(f"Successfully reached intermediate waypoint {waypoint}")
                    self.successful_waypoints += 1
            elif "failed" in self.latest_status.lower():
                self.get_logger().error(f"Navigation failed")
                self.navigation_in_progress = False
                self.publish_mission_complete(success=False)
    
    def navigation_watchdog(self):
        if not self.navigation_in_progress:
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_status_time
        
        if elapsed > self.navigation_timeout:
            self.get_logger().error(f"Navigation watchdog triggered - no status updates for {elapsed:.1f} seconds")
            self.navigation_in_progress = False
            self.publish_mission_complete(success=False)
    
    def check_service_and_start(self):
        if self.mission_started or not self.waypoints_received or self.service_call_in_progress:
            return
            
        if not self.service_ready:
            self.connection_attempts += 1
            
            if self.connection_attempts > self.max_connection_attempts:
                self.get_logger().warn(f"Exceeded maximum service connection attempts ({self.max_connection_attempts})")
                self.get_logger().warn("Will continue trying to connect but at a reduced rate")
                self.connection_attempts = 1  
                
            if self.nav_client.service_is_ready():
                self.get_logger().info(f"navigate_to_waypoint service for {self.robot_ns} is now available!")
                self.service_ready = True
                
                time.sleep(2.0)
                
                if self.waypoints_list:
                    self.get_logger().info(f"Starting navigation for {self.robot_ns} with waypoint sequence: {self.waypoints_list}")
                    self.mission_started = True
                    self.start_full_navigation(self.waypoints_list)
            else:
                if self.connection_attempts <= 3 or self.connection_attempts % 5 == 0:
                    self.get_logger().info(f"Waiting for navigate_to_waypoint service for {self.robot_ns} to become available (attempt {self.connection_attempts})...")
                    
                    if self.connection_attempts == 5:
                        try:
                            import subprocess
                            result = subprocess.run(['ros2', 'service', 'list'], 
                                                 capture_output=True, 
                                                 text=True,
                                                 shell=True)
                            self.get_logger().info(f"Available services: {result.stdout}")
                            
                            expected_service = f'/{self.robot_ns}/navigate_to_waypoint'
                            if expected_service in result.stdout:
                                self.get_logger().info(f"Service {expected_service} exists in service list but is not responding")
                            else:
                                self.get_logger().warn(f"Service {expected_service} NOT FOUND in service list")
                        except Exception as e:
                            self.get_logger().error(f"Error checking services: {e}")
    
    def publish_mission_complete(self, success=True):
        msg = String()
        if success:
            msg.data = f"Mission for {self.robot_ns} completed successfully"
        else:
            msg.data = f"Mission for {self.robot_ns} failed to complete"
            
        self.get_logger().info(f"Publishing mission completion: {msg.data}")
        
        self.get_logger().info(f"Successfully navigated to {self.successful_waypoints}/{len(self.waypoints_list)} waypoints")
        
        for _ in range(5):
            self.mission_complete_pub.publish(msg)
            time.sleep(0.5)
    
    def start_full_navigation(self, waypoint_list):
        self.service_call_in_progress = True
        
        if not waypoint_list:
            self.get_logger().warn("No valid waypoints to navigate to!")
            self.publish_mission_complete(success=False)
            self.service_call_in_progress = False
            return False
        
        valid_waypoints = []
        for wp in waypoint_list:
            if wp in self.waypoints:
                valid_waypoints.append(wp)
            else:
                self.get_logger().error(f"Target waypoint {wp} not found in the graph")
        
        if not valid_waypoints:
            self.get_logger().error("No valid waypoints found in the sequence")
            self.publish_mission_complete(success=False)
            self.service_call_in_progress = False
            return False
        
        if len(valid_waypoints) < len(waypoint_list):
            self.get_logger().warn(f"Only {len(valid_waypoints)} of {len(waypoint_list)} waypoints were valid")
        
        final_target = valid_waypoints[-1]
        
        request = NavigateToWaypoint.Request()
        
        request.waypoint_id = ','.join(valid_waypoints)
        request.start_waypoint_id = ""  
            
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                timeout = 10.0 + (attempt * 5.0)  
                self.get_logger().info(f"Waiting for navigate_to_waypoint service for {self.robot_ns} (attempt {attempt}/{max_attempts}, timeout {timeout}s)...")
                ready = self.nav_client.wait_for_service(timeout_sec=timeout)
                
                if not ready:
                    self.get_logger().error(f"Service not available after waiting {timeout} seconds (attempt {attempt}/{max_attempts})")
                    if attempt < max_attempts:
                        self.get_logger().info(f"Retrying in 5.0 seconds...")
                        time.sleep(5.0)
                        continue
                    else:
                        self.service_call_in_progress = False
                        return False
                
                self.get_logger().info(f"Sending navigation request for {self.robot_ns} for full path from start to waypoint {final_target} with intermediate waypoints")
                future = self.nav_client.call_async(request)
                
                rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
                
                if future.done():
                    result = future.result()
                    if result is not None:
                        self.get_logger().info(f"Service response for {self.robot_ns}: {result.message}")
                        if result.success:
                            self.navigation_in_progress = True
                            self.current_waypoint_index = 0
                            self.last_status_time = time.time()  
                            self.service_call_in_progress = False
                            return True
                        else:
                            self.get_logger().error(f"Service returned failure for {self.robot_ns}: {result.message}")
                            self.navigation_in_progress = True
                            self.last_status_time = time.time()
                            self.service_call_in_progress = False
                            return True
                    else:
                        self.get_logger().error(f'Service call for {self.robot_ns} returned None result')
                        if attempt < max_attempts:
                            self.get_logger().info(f"Retrying in 5.0 seconds...")
                            time.sleep(5.0)
                            continue
                        else:
                            self.service_call_in_progress = False
                            self.publish_mission_complete(success=False)
                            return False
                else:
                    self.get_logger().error(f'Service call for {self.robot_ns} timed out waiting for response')
                    if time.time() - self.last_status_time < 10.0:
                        self.get_logger().info(f"Receiving status updates for {self.robot_ns} despite service timeout - continuing navigation")
                        self.navigation_in_progress = True
                        self.service_call_in_progress = False
                        return True
                        
                    if attempt < max_attempts:
                        self.get_logger().info(f"Retrying in 5.0 seconds...")
                        time.sleep(5.0)
                        continue
                    else:
                        self.service_call_in_progress = False
                        self.publish_mission_complete(success=False)
                        return False
                    
            except Exception as e:
                self.get_logger().error(f'Error calling service for {self.robot_ns} (attempt {attempt}/{max_attempts}): {e}')
                if attempt < max_attempts:
                    self.get_logger().info(f"Retrying in 5.0 seconds...")
                    time.sleep(5.0)
                else:
                    self.service_call_in_progress = False
                    self.publish_mission_complete(success=False)
                    return False
        
        self.service_call_in_progress = False
        return False

def main(args=None):
    rclpy.init(args=args)
    
    tester = MultibotPathPlannerTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info("Node stopped cleanly by user")
    except Exception as e:
        tester.get_logger().error(f"Unhandled exception: {e}")
        try:
            tester.publish_mission_complete(success=False)
            time.sleep(2.0)
        except:
            pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))