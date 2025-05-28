#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from master_project2.srv import NavigateToWaypoint
import sys
import time
import random

class PathPlannerTester(Node):
    def __init__(self):
        super().__init__('path_planner_tester')
        
        self.waypoint_sequence = self.parse_command_line_args()
        self.get_logger().info(f"Path Planner Tester started. Will use waypoints provided via command line.")
        if self.waypoint_sequence:
            self.get_logger().info(f"Parsed waypoint IDs from argument: {self.waypoint_sequence}")
        
        self.nav_client = self.create_client(NavigateToWaypoint, 'navigate_to_waypoint')
        
        self.status_sub = self.create_subscription(
            String,
            'path_execution_status',
            self.status_callback,
            10
        )
        self.latest_status = ""
        
        self.mission_complete_pub = self.create_publisher(String, 'mission_complete', 10)
        
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
        
    def parse_command_line_args(self):
        waypoint_sequence = []
        
        filtered_args = []
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith('--') or ':=' in arg:
                if i + 1 < len(sys.argv) and not (sys.argv[i+1].startswith('--') or ':=' in sys.argv[i+1]):
                    i += 2
                else:
                    i += 1
            else:
                filtered_args.append(arg)
                i += 1
        
        self.get_logger().info(f"Filtered arguments (no ROS args): {filtered_args}")
        
        if filtered_args:
            arg = filtered_args[0]
            
            if ',' in arg:
                waypoint_ids = [wp.strip() for wp in arg.split(',')]
                waypoint_sequence.extend(waypoint_ids)
                self.get_logger().info(f"Parsed comma-separated waypoints: {waypoint_ids}")
            else:
                for arg in filtered_args:
                    waypoint_sequence.append(arg)
                self.get_logger().info(f"Parsed space-separated waypoints: {filtered_args}")
                    
        return waypoint_sequence
        
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
            self.get_logger().info(f"Path execution status: {msg.data}")
        
        if self.navigation_in_progress:
            if "completed" in self.latest_status.lower() and "successfully" in self.latest_status.lower():
                self.get_logger().info(f"Final goal reached! Mission completed successfully.")
                self.navigation_in_progress = False
                self.publish_mission_complete(success=True)
            elif "reached waypoint" in self.latest_status.lower():
                import re
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
            if self.nav_client.service_is_ready():
                self.get_logger().info("navigate_to_waypoint service is now available!")
                self.service_ready = True
                
                time.sleep(2.0)
                
                if self.waypoint_sequence:
                    self.get_logger().info(f"Starting navigation with full waypoint sequence: {self.waypoint_sequence}")
                    self.mission_started = True
                    self.start_full_navigation(self.waypoint_sequence)
            else:
                self.get_logger().debug("Waiting for navigate_to_waypoint service to become available...")
    
    def publish_mission_complete(self, success=True):
        msg = String()
        if success:
            msg.data = "Mission completed successfully"
        else:
            msg.data = "Mission failed to complete"
            
        self.get_logger().info(f"Publishing mission completion: {msg.data}")
        
        self.get_logger().info(f"Successfully navigated to {self.successful_waypoints}/{len(self.waypoint_sequence)} waypoints")
        
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
                self.get_logger().info(f"Waiting for navigate_to_waypoint service (attempt {attempt}/{max_attempts}, timeout {timeout}s)...")
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
                
                self.get_logger().info(f"Sending navigation request for full path from start to waypoint {final_target} with intermediate waypoints")
                future = self.nav_client.call_async(request)
                
                rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
                
                if future.done():
                    result = future.result()
                    if result is not None:
                        self.get_logger().info(f"Service response: {result.message}")
                        if result.success:
                            self.navigation_in_progress = True
                            self.current_waypoint_index = 0
                            self.last_status_time = time.time()
                            self.service_call_in_progress = False
                            return True
                        else:
                            self.get_logger().error(f"Service returned failure: {result.message}")
                            self.navigation_in_progress = True
                            self.last_status_time = time.time()
                            self.service_call_in_progress = False
                            return True
                    else:
                        self.get_logger().error('Service call returned None result')
                        if attempt < max_attempts:
                            self.get_logger().info(f"Retrying in 5.0 seconds...")
                            time.sleep(5.0)
                            continue
                        else:
                            self.service_call_in_progress = False
                            self.publish_mission_complete(success=False)
                            return False
                else:
                    self.get_logger().error('Service call timed out waiting for response')
                    if time.time() - self.last_status_time < 10.0:
                        self.get_logger().info("Receiving status updates despite service timeout - continuing navigation")
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
                self.get_logger().error(f'Error calling service (attempt {attempt}/{max_attempts}): {e}')
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
    
    tester = PathPlannerTester()
    
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