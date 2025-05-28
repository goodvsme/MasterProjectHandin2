#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
import math
import random
import time
import argparse
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q

def direction_to_yaw(direction):
    if direction == 'random':
        return random.uniform(0, 2 * math.pi)
    elif direction == 'right':
        return 0.0
    elif direction == 'up':
        return math.pi / 2
    elif direction == 'left':
        return math.pi
    elif direction == 'down':
        return 3 * math.pi / 2
    else:
        return random.uniform(0, 2 * math.pi)

class FlowGridTester(Node):
    def __init__(self, test_duration=60.0, publish_frequency=10.0, visualize=True):
        super().__init__('flow_grid_tester')
        
        self.test_duration = test_duration
        self.publish_frequency = publish_frequency
        self.visualize = visualize
        
        self.coordinates = [
            (-15.7, -64.0, 'up'),
            (-15.7, -84.0, 'down'),
            (46.3, -98.0, 'down'),
            (8.5, -75.0, 'down')
        ]
        
        self.num_humans = len(self.coordinates)
        
        self.latest_msg_qos = QoSProfile(
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
        
        self.human_publisher = self.create_publisher(
            PoseArray,
            '/human_detection',
            self.latest_msg_qos
        )
        
        if self.visualize:
            self.marker_publisher = self.create_publisher(
                MarkerArray,
                '/flow_grid_test_markers',
                self.latest_msg_qos
            )
        
        x_values = [coord[0] for coord in self.coordinates]
        y_values = [coord[1] for coord in self.coordinates]
        self.map_bounds = {
            'min_x': min(x_values) - 10.0,
            'max_x': max(x_values) + 10.0,
            'min_y': min(y_values) - 10.0,
            'max_y': max(y_values) + 10.0
        }
        
        self.initialize_humans()
        
        self.timer = self.create_timer(1.0/self.publish_frequency, self.publish_human_data)
        
        self.start_time = time.time()
        
        self.get_logger().info(f'FlowGridTester initialized with {self.num_humans} humans')
        self.get_logger().info(f'Test will run for {self.test_duration} seconds')
        self.get_logger().info(f'Publishing at {self.publish_frequency} Hz')
        coordinate_info = [(x, y, dir) for x, y, dir in self.coordinates]
        self.get_logger().info(f'Humans spawned at: {coordinate_info}')
    
    def initialize_humans(self):
        self.humans = []
        
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
        ]
        
        for i, (x, y, direction) in enumerate(self.coordinates):
            yaw = direction_to_yaw(direction)
            
            speed = 0.3
            
            self.humans.append({
                'id': i,
                'x': x,
                'y': y,
                'yaw': yaw,
                'target_x': x,
                'target_y': y,
                'speed': speed,
                'arrived': True,
                'color': colors[i % len(colors)]
            })
            
            self.get_logger().info(f"Human {i} at ({x}, {y}) with direction '{direction}' (yaw: {yaw:.2f} radians)")
        
        self.get_logger().info(f"Initialized {len(self.humans)} humans at exact coordinates with individual directions")

    
    def create_visualization_markers(self):
        marker_array = MarkerArray()
        
        for human in self.humans:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "test_humans"
            marker.id = human['id']
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            marker.pose.position.x = human['x']
            marker.pose.position.y = human['y']
            marker.pose.position.z = 0.2
            marker.pose.orientation = quaternion_from_euler(0.0, 0.0, human['yaw'])
            
            marker.scale.x = 0.5
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            marker.color = human['color']
            
            marker_array.markers.append(marker)
        
        return marker_array
    
    def publish_human_data(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > self.test_duration:
            self.get_logger().info("Test duration completed. Stopping...")
            self.timer.cancel()
            return
        
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for human in self.humans:
            pose = Pose()
            pose.position.x = human['x']
            pose.position.y = human['y']
            pose.position.z = 0.0
            pose.orientation = quaternion_from_euler(0.0, 0.0, human['yaw'])
            
            msg.poses.append(pose)
        
        self.human_publisher.publish(msg)
        
        if self.visualize:
            marker_array = self.create_visualization_markers()
            self.marker_publisher.publish(marker_array)
        
        if int(elapsed_time) % 5 == 0 and int(elapsed_time) != int(elapsed_time - 1.0/self.publish_frequency):
            self.get_logger().info(f"Published {len(msg.poses)} humans: {elapsed_time:.1f}/{self.test_duration} seconds elapsed")

def main():
    parser = argparse.ArgumentParser(description='Test script for FlowGrid')
    parser.add_argument('-d', '--duration', type=float, default=60.0,
                        help='Duration of the test in seconds (default: 60.0)')
    parser.add_argument('-f', '--frequency', type=float, default=10.0,
                        help='Publishing frequency in Hz (default: 10.0)')
    parser.add_argument('-v', '--visualize', action='store_true', default=True,
                        help='Enable visualization of simulated humans')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    tester = FlowGridTester(
        test_duration=args.duration,
        publish_frequency=args.frequency,
        visualize=args.visualize
    )
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()