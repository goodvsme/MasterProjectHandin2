#!/usr/bin/env python3
import rclpy
import json
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String, Empty, Float32MultiArray, MultiArrayDimension
import std_srvs.srv
import threading
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from master_project2.srv import UpdateWaypointCost

class WaypointGraphPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_graph_publisher')
        
        self.declare_parameter('base_update_frequency', 0.1)
        self.declare_parameter('waypoints_file', '')
        
        self.base_update_frequency = self.get_parameter('base_update_frequency').value
        self.publish_frequency = self.base_update_frequency / 2.0
        
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
        
        self.marker_pub = self.create_publisher(MarkerArray, 'waypoint_graph', self.config_qos)
        self.waypoint_pub = self.create_publisher(PoseStamped, 'waypoints', self.latest_msg_qos)
        
        self.cost_pub = self.create_publisher(
            Float32MultiArray, 
            'waypoint_costs', 
            self.config_qos
        )
        
        self.waypoint_lock = threading.Lock()
        
        self.marker_cache = {}
        self.changed_nodes = set()
        self.changed_edges = set()
        
        self.timer = self.create_timer(1.0/self.publish_frequency, self.publish_graph)
        
        self.cost_update_subscriber = self.create_subscription(
            String,
            '/waypoint_cost_update_status',
            self.cost_update_status_callback,
            self.latest_msg_qos
        )
        
        self.cost_update_complete_subscriber = self.create_subscription(
            Empty,
            '/waypoint_cost_update_complete',
            self.cost_update_complete_callback,
            self.latest_msg_qos
        )
        
        self.waypoints_file = self.resolve_waypoints_file_path()
        
        self.waypoints = {}
        self.graph = {}
        self.metadata = {}
        self.node_costs = {}
        
        self.load_waypoints_from_json(self.waypoints_file)
        
        self.reload_srv = self.create_service(
            std_srvs.srv.Trigger, 
            'reload_waypoints', 
            self.reload_waypoints_callback
        )
        
        self.update_cost_srv = self.create_service(
            UpdateWaypointCost, 
            'update_waypoint_cost', 
            self.update_waypoint_cost_callback
        )
        
        with self.waypoint_lock:
            self.changed_nodes = set(self.waypoints.keys())
            self.changed_edges = set()
            for node, neighbors in self.graph.items():
                for neighbor in neighbors:
                    self.changed_edges.add(tuple(sorted([node, neighbor])))
        
        self.get_logger().info('WaypointGraphPublisher node initialized')
        self.get_logger().info(f'Base update frequency: {self.base_update_frequency}Hz')
        self.get_logger().info(f'Publish frequency: {self.publish_frequency}Hz')
    
    def publish_costs(self):
        self.get_logger().debug("Publishing waypoint costs")
        
        with self.waypoint_lock:
            max_node_id = 0
            if self.waypoints:
                try:
                    max_node_id = max(map(int, self.waypoints.keys()))
                except ValueError:
                    self.get_logger().warn("Failed to determine maximum node ID")
                    return
            
            costs_array = [-1.0] * (max_node_id + 1)
            
            for node_id, cost in self.node_costs.items():
                try:
                    node_id_int = int(node_id)
                    costs_array[node_id_int] = float(cost)
                except (ValueError, IndexError):
                    self.get_logger().warn(f"Invalid node ID: {node_id}")
                    continue
            
            msg = Float32MultiArray()
            
            dim = MultiArrayDimension()
            dim.label = "waypoint_id"
            dim.size = len(costs_array)
            dim.stride = 1
            msg.layout.dim.append(dim)
            
            msg.data = costs_array
            
            self.cost_pub.publish(msg)
            self.get_logger().debug(f"Published costs for {len(costs_array)} waypoints")
    
    def resolve_waypoints_file_path(self):
        param_file_path = self.get_parameter('waypoints_file').value
        if param_file_path:
            if os.path.exists(param_file_path):
                self.get_logger().info(f"Using waypoints file from parameter: {param_file_path}")
                return param_file_path
            else:
                self.get_logger().warn(f"Specified waypoints file does not exist: {param_file_path}")
        
        try:
            from ament_index_python.packages import get_package_share_directory
            package_name = 'master_project2'
            package_dir = get_package_share_directory(package_name)
            default_path = os.path.join(package_dir, 'config', 'robot_waypoint_graph.json')
            
            if os.path.exists(default_path):
                self.get_logger().info(f"Using waypoints file from package share: {default_path}")
                return default_path
            else:
                self.get_logger().warn(f"Waypoints file not found in package share: {default_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not find package share directory: {e}")
        
        fallback_path = 'config/robot_waypoint_graph.json'
        self.get_logger().warn(f"Using fallback waypoints file path: {fallback_path}")
        return fallback_path
    
    def cost_update_status_callback(self, msg):
        self.get_logger().info(f"Cost update status received: {msg.data}")
        
        self.publish_graph()
    
    def cost_update_complete_callback(self, msg):
        self.get_logger().info("Received waypoint cost update complete notification")
        
        self.publish_graph()

    def reload_waypoints_callback(self, request, response):
        self.get_logger().info("Reload waypoints service called")
        try:
            self.get_logger().info(f"Reloading waypoints from {self.waypoints_file}")
            self.load_waypoints_from_json(self.waypoints_file)
            
            with self.waypoint_lock:
                self.changed_nodes = set(self.waypoints.keys())
                self.changed_edges = set()
                for node, neighbors in self.graph.items():
                    for neighbor in neighbors:
                        self.changed_edges.add(tuple(sorted([node, neighbor])))
            
            self.publish_graph()
            response.success = True
            response.message = f"Successfully reloaded waypoints from {self.waypoints_file}"
        except Exception as e:
            response.success = False
            response.message = f"Error reloading waypoints: {str(e)}"
        return response

    def update_waypoint_cost_callback(self, request, response):
        self.get_logger().info("Waypoint cost update service called")
        try:
            with self.waypoint_lock:
                if request.node_id == -1:
                    updated_nodes = 0
                    max_cost = 1.0

                    for node_id, cost in enumerate(request.costs):
                        node_str_id = str(node_id)
                        if node_str_id in self.waypoints:
                            if node_str_id not in self.node_costs or self.node_costs[node_str_id] != cost:
                                self.node_costs[node_str_id] = cost
                                self.changed_nodes.add(node_str_id)
                                updated_nodes += 1
                                max_cost = max(max_cost, cost)

                    response.success = True
                    response.message = (
                        f"Bulk updated {updated_nodes} waypoint costs in memory. "
                        f"Max cost: {max_cost}"
                    )
                    self.get_logger().info(response.message)
                    return response

                node_id = str(request.node_id)
                new_cost = request.cost
                
                if node_id not in self.waypoints:
                    response.success = False
                    response.message = f"Waypoint {node_id} does not exist."
                    self.get_logger().warn(response.message)
                    return response
                
                if node_id not in self.node_costs or self.node_costs[node_id] != new_cost:
                    self.node_costs[node_id] = new_cost
                    self.changed_nodes.add(node_id)
                    self.get_logger().info(f"Updated cost for node {node_id} to {new_cost} in memory")
            
            self.publish_graph()
            
            response.success = True
            response.message = f"Successfully updated cost for waypoint {node_id}"
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error updating waypoint cost: {str(e)}")
            response.success = False
            response.message = f"Error updating waypoint cost: {str(e)}"
            return response
    
    def handle_error(self, operation, error):
        with self.waypoint_lock:
            self.waypoints = {}
            self.graph = {}
            self.metadata = {}
            self.node_costs = {}
            self.changed_nodes = set()
            self.changed_edges = set()
            
    def load_waypoints_from_json(self, file_path):
        try:
            if not os.path.exists(file_path):
                self.get_logger().error(f"Waypoints file not found: {file_path}")
                with self.waypoint_lock:
                    self.waypoints = {}
                    self.graph = {}
                    self.metadata = {}
                    self.node_costs = {}
                return
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.get_logger().info(f"Successfully loaded waypoints from {file_path}")
            
            with self.waypoint_lock:
                old_waypoints = set(self.waypoints.keys())
                old_edges = set()
                for node, neighbors in self.graph.items():
                    for neighbor in neighbors:
                        old_edges.add(tuple(sorted([node, neighbor])))
                
                self.waypoints = {}
                self.graph = {}
                self.node_costs = {}
                
                for node in data['nodes']:
                    node_id = str(node['id'])
                    self.waypoints[node_id] = (node['x'], node['y'])
                    self.graph[node_id] = []
                    
                    if 'cost' in node:
                        self.node_costs[node_id] = node['cost']
                    else:
                        self.node_costs[node_id] = 1.0
                
                for edge in data['edges']:
                    source_id = str(edge['source'])
                    target_id = str(edge['target'])
                    
                    if target_id not in self.graph[source_id]:
                        self.graph[source_id].append(target_id)
                    if source_id not in self.graph[target_id]:
                        self.graph[target_id].append(source_id)
                
                if 'metadata' in data:
                    self.metadata = data['metadata']
                else:
                    self.metadata = {}
                
                new_waypoints = set(self.waypoints.keys())
                new_edges = set()
                for node, neighbors in self.graph.items():
                    for neighbor in neighbors:
                        new_edges.add(tuple(sorted([node, neighbor])))
                
                self.changed_nodes = new_waypoints | old_waypoints
                self.changed_edges = new_edges | old_edges
                    
            node_count = len(self.waypoints)
            edge_count = len(data.get('edges', []))
            self.get_logger().info(f"Loaded {node_count} waypoints and {edge_count} connections")
            
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON format in waypoints file: {file_path}")
            self.handle_error("loading waypoints", "Invalid JSON format")
        except Exception as e:
            self.get_logger().error(f"Error loading waypoints from JSON: {e}")
            self.handle_error("loading waypoints", e)
    
    def save_waypoints_to_json(self):
        try:
            file_path = self.waypoints_file
            
            with self.waypoint_lock:
                data = {
                    "metadata": self.metadata,
                    "nodes": [],
                    "edges": []
                }
                
                for node_id, (x, y) in self.waypoints.items():
                    node = {
                        "id": int(node_id),
                        "x": x,
                        "y": y,
                        "z": 0.0,
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0
                        }
                    }
                    
                    if node_id in self.node_costs:
                        node["cost"] = self.node_costs[node_id]
                    else:
                        node["cost"] = 1.0
                        
                    data["nodes"].append(node)
                    
                added_edges = set()
                for source_id, targets in self.graph.items():
                    for target_id in targets:
                        edge_id = tuple(sorted([source_id, target_id]))
                        if edge_id not in added_edges:
                            source_coords = self.waypoints[source_id]
                            target_coords = self.waypoints[target_id]
                            distance = self.calculate_distance(source_coords, target_coords)
                            
                            edge = {
                                "source": int(source_id),
                                "target": int(target_id),
                                "distance": distance
                            }
                            data["edges"].append(edge)
                            added_edges.add(edge_id)
            
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.get_logger().info(f"Saved waypoints to {file_path}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving waypoints to JSON: {e}")
            self.get_logger().error(f"Failed to save to path: {file_path}")

    def calculate_distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def create_waypoint_marker(self, node_id, x, y, cost):
        node_id_int = int(node_id)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = node_id_int
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        
        base_size = 0.3
        marker.scale.x = base_size
        marker.scale.y = base_size
        marker.scale.z = base_size
        
        max_cost = 10.0
        normalized_cost = min(1.0, cost / max_cost)
        marker.color = ColorRGBA(
            r=normalized_cost,
            g=1.0 - normalized_cost,
            b=0.0, 
            a=1.0
        )
        
        return marker
    
    def create_edge_marker(self, marker_id, start_coords, end_coords):
        edge_marker = Marker()
        edge_marker.header.frame_id = "map"
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.ns = "edges"
        edge_marker.id = marker_id
        edge_marker.type = Marker.LINE_STRIP
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.05
        edge_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)

        edge_marker.points = [
            Point(x=start_coords[0], y=start_coords[1], z=0.0),
            Point(x=end_coords[0], y=end_coords[1], z=0.0)
        ]
        
        return edge_marker

    def publish_graph(self):
        self.get_logger().debug("Publishing waypoint graph")
        marker_array = MarkerArray()
        marker_id = 1000
        active_markers = set()

        with self.waypoint_lock:
            for node in self.changed_nodes:
                if node in self.waypoints:
                    x, y = self.waypoints[node]
                    cost = self.node_costs.get(node, 1.0)
                    
                    marker_key = f"waypoint_{node}"
                    if marker_key in self.marker_cache:
                        marker = self.marker_cache[marker_key]
                        marker.pose.position.x = x
                        marker.pose.position.y = y
                        
                        max_cost = 10.0
                        normalized_cost = min(1.0, cost / max_cost)
                        marker.color.r = normalized_cost
                        marker.color.g = 1.0 - normalized_cost
                    else:
                        marker = self.create_waypoint_marker(node, x, y, cost)
                        self.marker_cache[marker_key] = marker
                    
                    marker_array.markers.append(marker)
                    active_markers.add(marker_key)
                    
                    pose = PoseStamped()
                    pose.header.frame_id = "map"
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = 0.0
                    self.waypoint_pub.publish(pose)
                else:
                    marker_key = f"waypoint_{node}"
                    if marker_key in self.marker_cache:
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "waypoints"
                        marker.id = int(node)
                        marker.action = Marker.DELETE
                        marker_array.markers.append(marker)
                        del self.marker_cache[marker_key]

            processed_edges = set()
            for edge in self.changed_edges:
                edge_sorted = tuple(sorted(edge))
                node1, node2 = edge_sorted
                
                if edge_sorted in processed_edges:
                    continue
                
                processed_edges.add(edge_sorted)
                marker_key = f"edge_{node1}_{node2}"
                
                if node1 in self.waypoints and node2 in self.waypoints and node2 in self.graph.get(node1, []):
                    start_coords = self.waypoints[node1]
                    end_coords = self.waypoints[node2]
                    
                    if marker_key in self.marker_cache:
                        marker = self.marker_cache[marker_key]
                        marker.points[0].x = start_coords[0]
                        marker.points[0].y = start_coords[1]
                        marker.points[1].x = end_coords[0]
                        marker.points[1].y = end_coords[1]
                    else:
                        while f"edge_{marker_id}" in self.marker_cache.values():
                            marker_id += 1
                        
                        marker = self.create_edge_marker(marker_id, start_coords, end_coords)
                        self.marker_cache[marker_key] = marker
                        marker_id += 1
                    
                    marker_array.markers.append(marker)
                    active_markers.add(marker_key)
                else:
                    if marker_key in self.marker_cache:
                        marker = Marker()
                        marker.header.frame_id = "map"
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = "edges"
                        marker.id = self.marker_cache[marker_key].id
                        marker.action = Marker.DELETE
                        marker_array.markers.append(marker)
                        del self.marker_cache[marker_key]
            
            for marker_key, marker in self.marker_cache.items():
                if marker_key not in active_markers:
                    marker_array.markers.append(marker)
            
            self.changed_nodes.clear()
            self.changed_edges.clear()

        self.marker_pub.publish(marker_array)
        
        self.publish_costs()
        
        self.get_logger().debug(f"Published Waypoint Graph with {len(marker_array.markers)} markers")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointGraphPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()