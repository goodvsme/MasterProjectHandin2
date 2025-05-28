#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import json
from pathlib import Path
import tf2_ros
import math
import os
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

class AdaptiveWaypointGenerator(Node):
    def __init__(self):
        super().__init__('adaptive_waypoint_generator')
        
        self.declare_parameter('base_grid_resolution', 1.0)
        self.declare_parameter('min_clearance', 0.4)
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('connection_radius_multiplier', 2.0)
        self.declare_parameter('json_output_path', '~/humans_waypoint_graph.json')
        self.declare_parameter('narrow_corridor_width', 1.0)
        self.declare_parameter('wide_corridor_width', 3.0)
        self.declare_parameter('narrow_resolution_factor', 2.5)
        self.declare_parameter('wide_resolution_factor', 0.5)
        
        self.base_grid_resolution = self.get_parameter('base_grid_resolution').value
        self.min_clearance = self.get_parameter('min_clearance').value
        self.map_topic = self.get_parameter('map_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.connection_radius_multiplier = self.get_parameter('connection_radius_multiplier').value
        self.json_output_path = self.get_parameter('json_output_path').value
        self.narrow_corridor_width = self.get_parameter('narrow_corridor_width').value
        self.wide_corridor_width = self.get_parameter('wide_corridor_width').value
        self.narrow_resolution_factor = self.get_parameter('narrow_resolution_factor').value
        self.wide_resolution_factor = self.get_parameter('wide_resolution_factor').value
        
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_logger().info('Created client for /map_server/map service')
        
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10)
        
        self.waypoint_array_publisher = self.create_publisher(PoseArray, '/waypoint_grid', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/waypoint_markers', 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.map_request_timer = self.create_timer(1.0, self.check_map_service)
        
        self.get_logger().info('Adaptive Waypoint Generator initialized')
        self.map_data = None
        self.inflated_map = None
        self.waypoints = []
        self.graph_edges = []
        
    def check_map_service(self):
        if self.map_data is not None:
            self.map_request_timer.cancel()
            return
            
        if not self.map_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info('Map service not available yet, waiting...')
            return
            
        self.get_logger().info('Map service available, requesting map...')
        self.request_map()
        
    def request_map(self):
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        future.add_done_callback(self.map_service_callback)
        self.get_logger().info('Map request sent')
        
    def map_service_callback(self, future):
        try:
            response = future.result()
            if response:
                self.get_logger().info('Map received from service')
                self.map_callback(response.map)
                self.map_request_timer.cancel()
            else:
                self.get_logger().error('Empty response from map service')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            self.get_logger().error(f'Error type: {type(e).__name__}')
        
    def map_callback(self, msg):
        self.get_logger().info('Received map, generating waypoints')
        self.map_data = msg
        
        if msg.info.resolution <= 0.0:
            self.get_logger().error(f'Invalid map resolution: {msg.info.resolution}')
            return
            
        clearance_cells = int(round(self.min_clearance / msg.info.resolution))
        self.inflated_map, self.distance_transform = self.generate_inflated_map_and_distance(msg, clearance_cells)
        
        self.waypoints = self.generate_adaptive_waypoints(msg, self.distance_transform)
        
        self.graph_edges = self.build_adaptive_graph(self.waypoints, self.inflated_map)
        
        self.publish_waypoints(self.waypoints)
        self.publish_waypoint_markers(self.waypoints, self.graph_edges)
        
        self.export_graph_to_json()

    def generate_inflated_map_and_distance(self, map_data, clearance_cells):
        width = map_data.info.width
        height = map_data.info.height
        
        grid = np.array(map_data.data).reshape((height, width))
        
        binary_grid = (grid > 0).astype(np.uint8)
        
        kernel_size = clearance_cells * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated_binary = cv2.dilate(binary_grid, kernel, iterations=1)
        
        dist_transform = distance_transform_edt(1 - binary_grid) * map_data.info.resolution
        
        inflated_data = (inflated_binary * 100).flatten().tolist()
        
        inflated_map = OccupancyGrid()
        inflated_map.header = map_data.header
        inflated_map.info = map_data.info
        inflated_map.data = inflated_data
        
        return inflated_map, dist_transform

    def generate_adaptive_waypoints(self, map_data, distance_transform):
        waypoints = []
        origin_x = map_data.info.origin.position.x
        origin_y = map_data.info.origin.position.y
        resolution = map_data.info.resolution
        width = map_data.info.width
        height = map_data.info.height
        
        waypoint_grid = np.zeros((height, width), dtype=bool)
        
        inflated_grid = np.array(self.inflated_map.data).reshape((height, width))
        
        free_space = (inflated_grid == 0).astype(np.uint8)
        skeleton = skeletonize(free_space).astype(np.uint8)
        
        for y in range(height):
            for x in range(width):
                if skeleton[y, x] > 0:
                    corridor_width = distance_transform[y, x] * 2
                    
                    if corridor_width < self.min_clearance * 2:
                        continue
                    
                    if corridor_width < self.narrow_corridor_width:
                        min_spacing = self.base_grid_resolution * self.narrow_resolution_factor
                        spacing_cells = int(min_spacing / resolution)
                        
                        area = waypoint_grid[
                            max(0, y - spacing_cells):min(height, y + spacing_cells + 1),
                            max(0, x - spacing_cells):min(width, x + spacing_cells + 1)
                        ]
                        if not np.any(area):
                            world_x = origin_x + x * resolution
                            world_y = origin_y + y * resolution
                            
                            pose = PoseStamped()
                            pose.header.frame_id = self.frame_id
                            pose.header.stamp = self.get_clock().now().to_msg()
                            pose.pose.position.x = world_x
                            pose.pose.position.y = world_y
                            pose.pose.position.z = 0.0
                            pose.pose.orientation.w = 1.0
                            waypoints.append(pose)
                            
                            waypoint_grid[y, x] = True
        
        for y in range(0, height, int(self.base_grid_resolution / resolution)):
            for x in range(0, width, int(self.base_grid_resolution / resolution)):
                if y >= height or x >= width or inflated_grid[y, x] > 0:
                    continue
                
                corridor_width = distance_transform[y, x] * 2
                
                if corridor_width < self.min_clearance * 2:
                    continue
                
                if corridor_width >= self.wide_corridor_width:
                    local_resolution = self.base_grid_resolution * self.wide_resolution_factor
                else:
                    local_resolution = self.base_grid_resolution
                
                spacing_cells = int(local_resolution / resolution)
                
                area = waypoint_grid[
                    max(0, y - spacing_cells//2):min(height, y + spacing_cells//2 + 1),
                    max(0, x - spacing_cells//2):min(width, x + spacing_cells//2 + 1)
                ]
                if not np.any(area):
                    world_x = origin_x + x * resolution
                    world_y = origin_y + y * resolution
                    
                    pose = PoseStamped()
                    pose.header.frame_id = self.frame_id
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.position.x = world_x
                    pose.pose.position.y = world_y
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    waypoints.append(pose)
                    
                    waypoint_grid[y, x] = True
        
        self.get_logger().info(f'Generated {len(waypoints)} adaptively placed waypoints')
        return waypoints

    def bresenham(self, x0, y0, x1, y1):
        line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            line.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return line
    
    def is_valid_edge(self, a_pose, b_pose, inflated_map):
        origin_x = inflated_map.info.origin.position.x
        origin_y = inflated_map.info.origin.position.y
        resolution = inflated_map.info.resolution
        x1 = a_pose.pose.position.x
        y1 = a_pose.pose.position.y
        x2 = b_pose.pose.position.x
        y2 = b_pose.pose.position.y
        
        mx1 = int(round((x1 - origin_x) / resolution))
        my1 = int(round((y1 - origin_y) / resolution))
        mx2 = int(round((x2 - origin_x) / resolution))
        my2 = int(round((y2 - origin_y) / resolution))
        
        line_cells = self.bresenham(mx1, my1, mx2, my2)
        for (mx, my) in line_cells:
            if mx < 0 or mx >= inflated_map.info.width or my < 0 or my >= inflated_map.info.height:
                return False
            idx = my * inflated_map.info.width + mx
            if inflated_map.data[idx] != 0:
                return False
        return True
    
    def build_adaptive_graph(self, waypoints, inflated_map):
        if not waypoints:
            return []
            
        graph_edges = []
        base_connection_radius = self.base_grid_resolution * self.connection_radius_multiplier
        
        positions = [(wp.pose.position.x, wp.pose.position.y) for wp in waypoints]
        kd_tree = KDTree(positions)
        
        for i, wp_i in enumerate(waypoints):
            xi = wp_i.pose.position.x
            yi = wp_i.pose.position.y
            
            indices = kd_tree.query_ball_point([xi, yi], base_connection_radius)
            
            for j in indices:
                if i >= j:
                    continue
                
                wp_j = waypoints[j]
                xj = wp_j.pose.position.x
                yj = wp_j.pose.position.y
                distance = math.hypot(xi - xj, yi - yj)
                
                if self.is_valid_edge(wp_i, wp_j, inflated_map):
                    graph_edges.append((i, j))
        
        connected = set()
        for i, j in graph_edges:
            connected.add(i)
            connected.add(j)
        
        if len(connected) < len(waypoints):
            for i in range(len(waypoints)):
                if i not in connected:
                    min_dist = float('inf')
                    nearest = -1
                    
                    for j in connected:
                        wp_i = waypoints[i]
                        wp_j = waypoints[j]
                        dist = math.hypot(
                            wp_i.pose.position.x - wp_j.pose.position.x,
                            wp_i.pose.position.y - wp_j.pose.position.y
                        )
                        if dist < min_dist and self.is_valid_edge(wp_i, wp_j, inflated_map):
                            min_dist = dist
                            nearest = j
                    
                    if nearest != -1:
                        graph_edges.append((i, nearest))
                        connected.add(i)
        
        self.get_logger().info(f'Built graph with {len(graph_edges)} edges')
        return graph_edges
        
    def publish_waypoints(self, waypoints):
        if not waypoints:
            self.get_logger().warn('No valid waypoints found!')
            return
            
        pose_array = PoseArray()
        pose_array.header.frame_id = self.frame_id
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        for waypoint in waypoints:
            pose_array.poses.append(waypoint.pose)
            
        self.waypoint_array_publisher.publish(pose_array)
        self.get_logger().info(f'Published {len(waypoints)} waypoints')
        
    def publish_waypoint_markers(self, waypoints, graph_edges):
        marker_array = MarkerArray()
        
        for i, waypoint in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = waypoint.pose
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
        
        edge_marker = Marker()
        edge_marker.header.frame_id = self.frame_id
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.ns = "edges"
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.05
        edge_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
        edge_marker.pose.orientation.w = 1.0
        
        for i, j in graph_edges:
            wp_i = waypoints[i]
            wp_j = waypoints[j]
            edge_marker.points.append(wp_i.pose.position)
            edge_marker.points.append(wp_j.pose.position)
        
        marker_array.markers.append(edge_marker)
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f'Published {len(graph_edges)} edges and {len(waypoints)} waypoints as markers')
    
    def export_graph_to_json(self):
        if not self.waypoints or not self.graph_edges:
            self.get_logger().warn('No graph to export!')
            return
            
        output_path = os.path.expanduser(self.json_output_path)
        
        graph_data = {
            "metadata": {
                "frame_id": self.frame_id,
                "timestamp": self.get_clock().now().to_msg().sec,
                "base_grid_resolution": self.base_grid_resolution,
                "min_clearance": self.min_clearance,
                "narrow_corridor_width": self.narrow_corridor_width,
                "wide_corridor_width": self.wide_corridor_width
            },
            "nodes": [],
            "edges": []
        }
        
        for i, waypoint in enumerate(self.waypoints):
            node = {
                "id": i,
                "x": waypoint.pose.position.x,
                "y": waypoint.pose.position.y,
                "z": waypoint.pose.position.z,
                "orientation": {
                    "x": waypoint.pose.orientation.x,
                    "y": waypoint.pose.orientation.y,
                    "z": waypoint.pose.orientation.z,
                    "w": waypoint.pose.orientation.w
                }
            }
            graph_data["nodes"].append(node)
            
        for i, j in self.graph_edges:
            wp_i = self.waypoints[i]
            wp_j = self.waypoints[j]
            xi, yi = wp_i.pose.position.x, wp_i.pose.position.y
            xj, yj = wp_j.pose.position.x, wp_j.pose.position.y
            distance = math.hypot(xi - xj, yi - yj)
            
            edge = {
                "source": i,
                "target": j,
                "distance": distance
            }
            graph_data["edges"].append(edge)
            
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            self.get_logger().info(f'Exported graph to {output_path}')
            
            self.get_logger().info(f'Graph summary: {len(graph_data["nodes"])} nodes, {len(graph_data["edges"])} edges')
        except Exception as e:
            self.get_logger().error(f'Failed to export graph: {str(e)}')
        
    def destroy_node(self):
        self.get_logger().info('Shutting down AdaptiveWaypointGenerator')
        super().destroy_node()
        

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveWaypointGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()