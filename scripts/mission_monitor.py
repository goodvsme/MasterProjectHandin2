#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Transform
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import math
import datetime
import json
import os
from ament_index_python.packages import get_package_share_directory
from rclpy.time import Time
from rclpy.duration import Duration

class MissionMonitor(Node):
    def __init__(self):
        super().__init__('mission_monitor')
        
        self.declare_parameter('log_directory', 'logs')
        self.declare_parameter('stop_velocity_threshold', 0.01)
        self.declare_parameter('stop_time_threshold', 2.0)
        self.declare_parameter('resume_velocity_threshold', 0.05)
        self.declare_parameter('path_tracking_mode', 'odometry')

        self.declare_parameter('random_seed', 100)
        self.random_seed = self.get_parameter('random_seed').value
        
        use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(f"Using simulation time: {use_sim_time}")
        
        self.log_directory = self.get_parameter('log_directory').value
        self.stop_velocity_threshold = self.get_parameter('stop_velocity_threshold').value
        self.stop_time_threshold = self.get_parameter('stop_time_threshold').value
        self.resume_velocity_threshold = self.get_parameter('resume_velocity_threshold').value
        self.path_tracking_mode = self.get_parameter('path_tracking_mode').value
        
        package_dir = get_package_share_directory('master_project2')
        self.log_path = os.path.join(package_dir, self.log_directory)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.mission_active = True
        self.mission_start_time = self.get_clock().now()
        self.mission_name = "default"
        self.path_length = 0.0
        self.stops_count = 0
        self.total_stop_duration = 0.0
        self.stopping_reasons = []
        self.stop_start_time = None
        self.is_stopped = False
        self.last_pose = None
        self.current_path = []
        self.events = []
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            '/path_execution_status',
            self.status_callback,
            10
        )
        
        self.timer = self.create_timer(0.5, self.check_status)
        
        self.get_logger().info('Mission monitor initialized. Mission active and timing started.')
    
    def amcl_callback(self, msg):
        if not self.mission_active:
            return
        
        current_pose = msg.pose.pose
        
        if self.last_pose is not None:
            if isinstance(self.last_pose, Transform):
                class TempPose:
                    def __init__(self, transform):
                        self.position = type('Position', (), {
                            'x': transform.translation.x,
                            'y': transform.translation.y,
                            'z': transform.translation.z
                        })
                last_pose_for_calc = TempPose(self.last_pose)
            else:
                last_pose_for_calc = self.last_pose
            
            distance = self.calculate_distance(last_pose_for_calc, current_pose)
            
            if distance > 0.01:
                self.path_length += distance
                
                path_point = (current_pose.position.x, current_pose.position.y)
                self.current_path.append(path_point)
        
        self.last_pose = current_pose
    
    def cmd_vel_callback(self, msg):
        if not self.mission_active:
            return
            
        linear_velocity = math.sqrt(
            msg.linear.x**2 + 
            msg.linear.y**2 + 
            msg.linear.z**2
        )
        
        self.check_stopped_state(linear_velocity)
        
        if self.is_stopped and linear_velocity > self.resume_velocity_threshold:
            if "obstacle" not in self.stopping_reasons:
                self.stopping_reasons.append("obstacle")
    
    def path_callback(self, msg):
        if not self.mission_active or not msg.poses:
            return
        
        if self.is_stopped and "replanning" not in self.stopping_reasons:
            self.stopping_reasons.append("replanning")
            
    def status_callback(self, msg):
        status_text = msg.data.lower()
        
        if "started" in status_text or "navigating" in status_text:
            self.get_logger().info(f"Navigation status update: {status_text}")
            self.log_checkpoint("navigation_started")
                
        elif "completed" in status_text or "finished" in status_text:
            if self.mission_active:
                self.end_mission("completed")
                
        elif "failed" in status_text or "error" in status_text:
            if self.mission_active:
                if self.is_stopped and "navigation failure" not in self.stopping_reasons:
                    self.stopping_reasons.append("navigation failure")
                self.end_mission("failed")
                
        elif "waypoint" in status_text:
            self.get_logger().info(f"Waypoint update: {status_text}")
            if self.mission_active:
                waypoint_num = self.extract_waypoint_number(status_text)
                if waypoint_num is not None:
                    self.log_checkpoint(f"reached_waypoint_{waypoint_num}")
    
    def check_stopped_state(self, linear_velocity):
        current_time = self.get_clock().now()
        
        if not self.is_stopped and linear_velocity < self.stop_velocity_threshold:
            if self.stop_start_time is None:
                self.stop_start_time = current_time
            elif (current_time - self.stop_start_time).nanoseconds / 1e9 > self.stop_time_threshold:
                self.is_stopped = True
                self.stops_count += 1
                self.get_logger().info(f"Robot stopped (stop #{self.stops_count})")
                self.stopping_reasons = ["unknown"]
        
        elif self.is_stopped and linear_velocity > self.resume_velocity_threshold:
            stop_duration = (current_time - self.stop_start_time).nanoseconds / 1e9
            self.total_stop_duration += stop_duration
            
            reason_str = ", ".join(self.stopping_reasons)
            self.get_logger().info(
                f"Robot resumed moving after {stop_duration:.2f} seconds. "
                f"Reason for stop: {reason_str}"
            )
            
            self.log_stop_event(stop_duration, self.stopping_reasons)
            
            self.is_stopped = False
            self.stop_start_time = None
            self.stopping_reasons = []
        
        elif self.is_stopped:
            self.current_stop_duration = (current_time - self.stop_start_time).nanoseconds / 1e9
    
    def check_status(self):
        if not self.mission_active:
            return
            
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.mission_start_time).nanoseconds / 1e9
        
        elapsed_int = int(elapsed_time)
        if elapsed_int % 30 == 0 and elapsed_int > 0:
            self.get_logger().info(
                f"Mission status: Duration={elapsed_time:.1f}s, "
                f"Path length={self.path_length:.2f}m, "
                f"Stops={self.stops_count}, "
                f"Stop time={self.total_stop_duration:.1f}s"
            )
    
    def start_mission(self, mission_name="mission"):
        self.mission_name = mission_name.replace(" ", "_")
        
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            self.last_pose = transform.transform
            self.get_logger().info(f"Got initial transform position: ({transform.transform.translation.x}, {transform.transform.translation.y})")
        except Exception as e:
            self.get_logger().warning(f"Could not get initial robot position: {e}")
            self.last_pose = None
        
        self.get_logger().info(f"Mission name set to: {self.mission_name}")
    
    def end_mission(self, status):
        if not self.mission_active:
            return
            
        current_time = self.get_clock().now()
        mission_duration = (current_time - self.mission_start_time).nanoseconds / 1e9

        self.get_logger().info(
            f"Mission {self.mission_name} {status}: "
            f"Duration={mission_duration:.1f}s, "
            f"Path length={self.path_length:.2f}m, "
            f"Stops={self.stops_count}, "
            f"Total stop time={self.total_stop_duration:.1f}s"
        )
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"mission_log_{timestamp}.json"
        log_filepath = os.path.join(self.log_path, log_filename)
        
        log_data = {
            "mission_name": self.mission_name,
            "random_seed": self.random_seed,
            "timestamp": timestamp,
            "duration_seconds": round(mission_duration, 2),
            "path_length_meters": round(self.path_length, 2),
            "status": status,
            "stops_count": self.stops_count,
            "total_stop_duration": round(self.total_stop_duration, 2),
            "path_points": self.current_path,
            "events": self.events
        }
                
        try:
            with open(log_filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.get_logger().info(f"Mission log saved to {log_filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to save mission log: {e}")
        
        self.mission_active = False
    
    def log_stop_event(self, duration, reasons):
        if self.mission_start_time:
            current_time = self.get_clock().now()
            timestamp = (current_time - self.mission_start_time).nanoseconds / 1e9
        else:
            timestamp = 0.0
            
        event = {
            "type": "stop",
            "timestamp": timestamp,
            "duration": round(duration, 2),
            "reasons": reasons
        }
        
        self.events.append(event)
    
    def log_checkpoint(self, checkpoint_name):
        if self.mission_start_time:
            current_time = self.get_clock().now()
            timestamp = (current_time - self.mission_start_time).nanoseconds / 1e9
        else:
            timestamp = 0.0
            
        event = {
            "type": "checkpoint",
            "name": checkpoint_name,
            "timestamp": timestamp,
            "path_length_so_far": round(self.path_length, 2)
        }
        
        self.events.append(event)
    
    def calculate_distance(self, pose1, pose2):
        return math.sqrt(
            (pose2.position.x - pose1.position.x) ** 2 +
            (pose2.position.y - pose1.position.y) ** 2 +
            (pose2.position.z - pose1.position.z) ** 2
        )
    
    def extract_waypoint_number(self, text):
        try:
            import re
            match = re.search(r'waypoint (\d+)', text)
            if match:
                return int(match.group(1))
        except:
            pass
        return None

def main(args=None):
    rclpy.init(args=args)
    node = MissionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.mission_active:
            node.end_mission("interrupted")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()