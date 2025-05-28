#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
import numpy as np
from scipy.spatial import KDTree

class MultibotHumanDetectionAggregator(Node):
    def __init__(self):
        super().__init__('multibot_human_detection_aggregator')
        
        # Parameters
        self.declare_parameter('num_robots', 3)
        self.num_robots = self.get_parameter('num_robots').value
        
        self.declare_parameter('detection_topic_pattern', '/robot_{}/human_detections')
        self.detection_topic_pattern = self.get_parameter('detection_topic_pattern').value
        
        self.declare_parameter('output_topic', '/raw_human_detections')
        self.output_topic = self.get_parameter('output_topic').value
        
        self.declare_parameter('duplicate_threshold', 0.5)  # Meters
        self.duplicate_threshold = self.get_parameter('duplicate_threshold').value
        
        self.declare_parameter('republish_interval', 0.1)  # Seconds
        self.republish_interval = self.get_parameter('republish_interval').value
        
        # QoS profile
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Create publisher
        self.publisher = self.create_publisher(
            PoseArray, 
            self.output_topic, 
            self.qos
        )
        
        # Create subscribers for each robot
        self.subscribers = []
        for i in range(self.num_robots):
            robot_topic = self.detection_topic_pattern.format(i)
            self.get_logger().info(f"Subscribing to {robot_topic}")
            
        
            subscriber = self.create_subscription(
                PoseArray,
                robot_topic,
                lambda msg, robot_id=i: self.detection_callback(msg, robot_id),
                self.qos
            )
            self.subscribers.append(subscriber)
        
        self.recent_detections = []  
        self.detection_timeout = 2.0  
        
        
        self.timer = self.create_timer(self.republish_interval, self.publish_aggregated_detections)
        
        self.get_logger().info(f"Multibot Human Detection Aggregator initialized with {self.num_robots} robots")
        self.get_logger().info(f"Publishing aggregated detections to {self.output_topic}")
        self.get_logger().info(f"Duplicate threshold: {self.duplicate_threshold} meters")
        self.get_logger().info(f"Republish interval: {self.republish_interval} seconds")
    
    def detection_callback(self, msg, robot_id):
        if len(msg.poses) > 0:
            self.get_logger().info(f"Received {len(msg.poses)} detections from robot_{robot_id}")
        
        current_time = self.get_clock().now().to_msg().sec
        

        for pose in msg.poses:
            self.recent_detections.append((pose, current_time, robot_id))
        
        self.clean_old_detections()
    
    def clean_old_detections(self):
        current_time = self.get_clock().now().to_msg().sec
        self.recent_detections = [
            (pose, timestamp, robot_id) for pose, timestamp, robot_id in self.recent_detections
            if current_time - timestamp < self.detection_timeout
        ]
    
    def remove_duplicates(self, poses):
        if not poses:
            return []
        
        points = [(pose.position.x, pose.position.y) for pose in poses]
        
        tree = KDTree(points)
        
        duplicate_indices = set()
        for i in range(len(points)):
            if i in duplicate_indices:
                continue
                
            indices = tree.query_ball_point(points[i], self.duplicate_threshold)
            
            indices = [idx for idx in indices if idx != i]
            
            duplicate_indices.update(indices)
        
        keep_indices = [i for i in range(len(poses)) if i not in duplicate_indices]
        
        return [poses[i] for i in keep_indices]
    
    def publish_aggregated_detections(self):
        self.clean_old_detections()
        
        all_poses = [pose for pose, _, _ in self.recent_detections]
        
        if not all_poses:
            return
        
        filtered_poses = self.remove_duplicates(all_poses)
        
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        pose_array.poses = filtered_poses
        
        self.publisher.publish(pose_array)
        
        if len(filtered_poses) > 0:
            self.get_logger().info(f"Published aggregated detections: {len(filtered_poses)} poses " +
                                 f"(removed {len(all_poses) - len(filtered_poses)} duplicates)")

def main(args=None):
    rclpy.init(args=args)
    node = MultibotHumanDetectionAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()