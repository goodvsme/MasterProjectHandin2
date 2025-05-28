#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Empty, String
import std_srvs.srv
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from master_project2.srv import UpdateWaypointCost
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import time

class WaypointCostUpdater(Node):
    def __init__(self):
        super().__init__('waypoint_cost_updater')
        
        self.declare_parameter('min_cost', 1.0)
        self.declare_parameter('max_cost', 10.0)
        self.declare_parameter('obstacle_cost', 10000.0)
        self.declare_parameter('prediction_threshold', 0.01)
        self.declare_parameter('base_update_frequency', 0.1)
        self.declare_parameter('rapid_update_interval', 0.2)
        self.declare_parameter('map_check_interval', 20.0)
        self.declare_parameter('debug_mode', True)
        
        self.min_cost = self.get_parameter('min_cost').value
        self.max_cost = self.get_parameter('max_cost').value
        self.obstacle_cost = self.get_parameter('obstacle_cost').value
        self.prediction_threshold = self.get_parameter('prediction_threshold').value
        self.base_update_frequency = self.get_parameter('base_update_frequency').value
        self.rapid_update_interval = self.get_parameter('rapid_update_interval').value
        self.map_check_interval = self.get_parameter('map_check_interval').value
        self.debug_mode = False
        
        self.last_update_time = 0.0
        
        self.prediction_grid = None
        self.prediction_grid_info = {
            'width': 0,
            'height': 0,
            'num_directions': 0,
            'resolution': 0.5,
            'origin_x': 0.0,
            'origin_y': 0.0
        }
        
        self.map_data = None
        self.waypoints_in_obstacles = set()
        self.last_map_update = None
        
        self.current_costs = {}
        
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
        
        self.update_cost_client = self.create_client(UpdateWaypointCost, 'update_waypoint_cost')
        
        while not self.update_cost_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for update_waypoint_cost service...')
        
        self.prediction_subscriber = self.create_subscription(
            Float32MultiArray,
            '/prediction_grid',
            self.prediction_callback,
            self.latest_msg_qos
        )
        
        self.waypoints = {}
        self.marker_subscriber = self.create_subscription(
            MarkerArray,
            'waypoint_graph',
            self.waypoint_graph_callback,
            self.config_qos
        )
        
        self.update_subscriber = self.create_subscription(
            Empty,
            '/flow_grid_update_complete',
            self.flow_grid_update_callback,
            self.latest_msg_qos
        )
        
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/cleaned_map',
            self.map_callback,
            self.latest_msg_qos
        )
        
        self.update_status_publisher = self.create_publisher(
            String,
            '/waypoint_cost_update_status',
            self.latest_msg_qos
        )
        
        self.update_complete_publisher = self.create_publisher(
            Empty,
            '/waypoint_cost_update_complete',
            self.latest_msg_qos
        )
        
        update_period = 1.5/self.base_update_frequency
        self.update_timer = self.create_timer(update_period, self.update_timer_callback)
        
        self.map_check_timer = self.create_timer(self.map_check_interval, self.check_waypoints_in_obstacles)
        
        self.debug_timer = self.create_timer(30.0, self.print_debug_status)

    def print_debug_status(self):
        self.get_logger().info("==== DEBUG STATUS ====")
        
        if self.map_data is None:
            self.get_logger().info("Map status: NO MAP DATA RECEIVED YET")
        else:
            time_since_update = "Unknown"
            if hasattr(self, 'last_map_update') and self.last_map_update is not None:
                time_since_update = f"{time.time() - self.last_map_update:.1f} seconds ago"
            
            self.get_logger().info(f"Map status: Map data received {time_since_update}")
            self.get_logger().info(f"  - Map dimensions: {self.map_data.info.width}x{self.map_data.info.height}, resolution: {self.map_data.info.resolution}")
            self.get_logger().info(f"  - Map origin: ({self.map_data.info.origin.position.x}, {self.map_data.info.origin.position.y})")
            
            obstacle_count = sum(1 for val in self.map_data.data if val >= 50)
            unknown_count = sum(1 for val in self.map_data.data if val == -1)
            free_count = sum(1 for val in self.map_data.data if val == 0)
            unique_values = set(self.map_data.data)
            
            self.get_logger().info(f"  - Map content: {obstacle_count} obstacles, {free_count} free, {unknown_count} unknown")
            self.get_logger().info(f"  - Unique values in map: {unique_values}")
        
        self.get_logger().info(f"Waypoints: {len(self.waypoints)} waypoints loaded")
        self.get_logger().info(f"Waypoints in obstacles: {len(self.waypoints_in_obstacles)}")
        
        if self.map_data is not None and self.waypoints:
            self.check_specific_waypoints()

    def check_specific_waypoints(self):
        if self.map_data is None:
            self.get_logger().error("No map data available for checking specific waypoints")
            return
        
        waypoint_ids = list(self.waypoints.keys())
        sample_size = min(5, len(waypoint_ids))
        waypoints_to_check = waypoint_ids[:sample_size]
        
        self.get_logger().info("Checking sample waypoints against map:")
        for wp_id in waypoints_to_check:
            x, y = self.waypoints[wp_id]
            grid_x, grid_y = self.world_to_map_grid(x, y)
            
            if grid_x is not None and grid_y is not None:
                map_value = self.get_map_value(grid_x, grid_y)
                self.get_logger().info(f"  Waypoint {wp_id} at ({x:.2f}, {y:.2f}) → grid ({grid_x}, {grid_y}) → map value: {map_value}")
            else:
                self.get_logger().warn(f"  Waypoint {wp_id} at ({x:.2f}, {y:.2f}) is outside map bounds")

    def map_callback(self, msg):
        occupied_cells = sum(1 for val in msg.data if val > 50)
        unique_values = set(msg.data)
        
        self.get_logger().info("Received updated map data")
        self.get_logger().info(f"Map dimensions: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")
        self.get_logger().info(f"Map origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y})")
        self.get_logger().info(f"Map content: {occupied_cells} occupied cells")
        self.get_logger().info(f"Unique values in map: {unique_values}")
        
        self.last_map_update = time.time()
        
        self.map_data = msg
        
        self.check_waypoints_in_obstacles()

    def check_waypoints_in_obstacles(self):
        if self.map_data is None:
            self.get_logger().warn("Cannot check waypoints in obstacles - no map data available")
            return
            
        if not self.waypoints:
            self.get_logger().warn("Cannot check waypoints in obstacles - no waypoint data available")
            return
            
        self.get_logger().info("Checking if waypoints are in obstacles...")
        
        obstacle_count = sum(1 for val in self.map_data.data if val >= 50)
        unknown_count = sum(1 for val in self.map_data.data if val == -1)
        free_count = sum(1 for val in self.map_data.data if val == 0)
        
        self.get_logger().info(f"Map analysis: {obstacle_count} obstacles, {free_count} free, {unknown_count} unknown")
        
        if obstacle_count < 10:
            self.get_logger().warn("Very few obstacles found in map - check if map is being correctly processed")
        
        old_obstacle_waypoints = self.waypoints_in_obstacles.copy()
        self.waypoints_in_obstacles.clear()
        
        waypoint_map_values = {}
        
        self.get_logger().info(f"Checking {len(self.waypoints)} waypoints against map")
        
        for node_id, (x, y) in self.waypoints.items():
            if node_id in ['0', '1', '2', '3', '4', '5'] or int(node_id) % 20 == 0:
                self.get_logger().debug(f"Checking waypoint {node_id} at world coordinates ({x:.2f}, {y:.2f})")
            
            grid_x, grid_y = self.world_to_map_grid(x, y)
            
            if grid_x is None or grid_y is None:
                if node_id in ['0', '1', '2', '3', '4', '5'] or int(node_id) % 20 == 0:
                    self.get_logger().warn(f"Waypoint {node_id} at ({x:.2f}, {y:.2f}) is outside map bounds")
                continue
            
            if node_id in ['0', '1', '2', '3', '4', '5'] or int(node_id) % 20 == 0:
                self.get_logger().debug(f"Waypoint {node_id} maps to grid coordinates ({grid_x}, {grid_y})")
                
            map_value = self.get_map_value(grid_x, grid_y)
            waypoint_map_values[node_id] = map_value
            
            if map_value > 50:
                self.waypoints_in_obstacles.add(node_id)
                self.get_logger().info(f"Waypoint {node_id} is in obstacle with value {map_value}")
        
        unique_values = set(waypoint_map_values.values())
        self.get_logger().info(f"Map values at waypoints: {unique_values}")
        
        high_value_waypoints = {k: v for k, v in waypoint_map_values.items() if v > 20 and v <= 50}
        if high_value_waypoints:
            self.get_logger().info(f"Waypoints with medium values (not classified as obstacles): {high_value_waypoints}")
        
        if self.waypoints_in_obstacles != old_obstacle_waypoints:
            self.get_logger().info(f"Found {len(self.waypoints_in_obstacles)} waypoints in obstacles. Updating costs...")
            new_obstacles = self.waypoints_in_obstacles - old_obstacle_waypoints
            removed_obstacles = old_obstacle_waypoints - self.waypoints_in_obstacles
            
            if new_obstacles:
                self.get_logger().info(f"New obstacles at waypoints: {new_obstacles}")
            if removed_obstacles:
                self.get_logger().info(f"Removed obstacles at waypoints: {removed_obstacles}")
                
            self.update_waypoint_costs()
        else:
            self.get_logger().info(f"No change in obstacle waypoints (still have {len(self.waypoints_in_obstacles)})")

    def world_to_map_grid(self, x, y):
        if self.map_data is None:
            return None, None
            
        map_info = self.map_data.info
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        resolution = map_info.resolution
        width = map_info.width
        height = map_info.height
        
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        if 0 <= grid_x < width and 0 <= grid_y < height:
            return grid_x, grid_y
        else:
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
                if self._log_counter % 50 == 0:
                    self.get_logger().warn(
                        f"Coordinates ({grid_x}, {grid_y}) are outside map bounds {width}x{height}. "
                        f"World: ({x:.2f}, {y:.2f}), Map origin: ({origin_x:.2f}, {origin_y:.2f}), "
                        f"Resolution: {resolution}"
                    )
            else:
                self._log_counter = 0
                self.get_logger().warn(
                    f"Coordinates are outside map bounds. First occurrence: "
                    f"World: ({x:.2f}, {y:.2f}), Map grid: ({grid_x}, {grid_y}), "
                    f"Map bounds: {width}x{height}, Origin: ({origin_x:.2f}, {origin_y:.2f}), "
                    f"Resolution: {resolution}"
                )
            return None, None

    def get_map_value(self, grid_x, grid_y):
        if self.map_data is None:
            return -1
            
        width = self.map_data.info.width
        index = grid_y * width + grid_x
        
        if 0 <= index < len(self.map_data.data):
            return self.map_data.data[index]
        else:
            if hasattr(self, '_index_log_counter'):
                self._index_log_counter += 1
                if self._index_log_counter % 50 == 0:
                    self.get_logger().warn(f"Invalid map index: {index}, grid: ({grid_x}, {grid_y}), map size: {len(self.map_data.data)}")
            else:
                self._index_log_counter = 0
                self.get_logger().warn(f"Invalid map index: {index}, grid: ({grid_x}, {grid_y}), map size: {len(self.map_data.data)}")
            return -1

    def flow_grid_update_callback(self, msg):
        self.get_logger().info("Received flow grid update notification")
        self.process_update("flow grid update complete")
        
    def process_update(self, update_type):
        current_time = time.time()
        
        if current_time - self.last_update_time < self.rapid_update_interval:
            self.get_logger().debug(f"Rate limiting: skipping {update_type} (too frequent)")
            return
            
        self.get_logger().info(f"Processing {update_type}")
        self.update_waypoint_costs()
        
        self.last_update_time = current_time

    def update_timer_callback(self):
        current_time = time.time()
        
        if self.map_data is None:
            self.get_logger().warn("No map data received yet - can't check for obstacles")
        elif hasattr(self, 'last_map_update') and self.last_map_update is not None:
            time_since_update = current_time - self.last_map_update
            if time_since_update > 60.0:
                self.get_logger().warn(f"Map data is stale - last update was {time_since_update:.1f} seconds ago")
        
        if current_time - self.last_update_time > 2.0/self.base_update_frequency:
            self.get_logger().info('Triggering cost update via fallback timer')
            self.update_waypoint_costs()
            self.last_update_time = current_time

    def waypoint_graph_callback(self, msg):
        self.get_logger().info("Received updated waypoint graph data")
        try:
            waypoint_markers = [m for m in msg.markers if m.ns == "waypoints"]
            
            old_waypoint_count = len(self.waypoints)
            self.waypoints = {}
            
            for marker in waypoint_markers:
                node_id = str(marker.id)
                x = marker.pose.position.x
                y = marker.pose.position.y
                
                self.waypoints[node_id] = (x, y)
                
                normalized_cost = marker.color.r
                estimated_cost = self.min_cost + normalized_cost * (self.max_cost - self.min_cost)
                self.current_costs[node_id] = estimated_cost
            
            if old_waypoint_count != len(self.waypoints):
                self.get_logger().info(f'Updated waypoints from marker array: {len(self.waypoints)} waypoints')
                
                sample_waypoints = {}
                for node_id in list(self.waypoints.keys())[:5]:
                    sample_waypoints[node_id] = self.waypoints[node_id]
                self.get_logger().info(f"Sample waypoints: {sample_waypoints}")
            
            self.check_waypoints_in_obstacles()
            
        except Exception as e:
            self.get_logger().error(f'Error processing waypoint graph: {str(e)}')
    
    def prediction_callback(self, msg):
        self.get_logger().info("Received new prediction grid data")
        try:
            if len(msg.layout.dim) != 3:
                self.get_logger().error('Expected 3D array for prediction grid')
                return
            
            height = int(msg.layout.dim[0].size)
            width = int(msg.layout.dim[1].size)
            num_directions = int(msg.layout.dim[2].size)
            
            self.prediction_grid_info['height'] = height
            self.prediction_grid_info['width'] = width
            self.prediction_grid_info['num_directions'] = num_directions
            
            if len(msg.data) > 3:
                self.prediction_grid_info['origin_x'] = float(msg.data[0])
                self.prediction_grid_info['origin_y'] = float(msg.data[1]) 
                self.prediction_grid_info['resolution'] = float(msg.data[2])
                
                data = np.array(msg.data[3:], dtype=np.float32)
                self.prediction_grid = data.reshape((height, width, num_directions))
                
                self.get_logger().info(
                    f"Prediction grid properties: {width}x{height}x{num_directions}, "
                    f"origin: ({self.prediction_grid_info['origin_x']:.2f}, {self.prediction_grid_info['origin_y']:.2f}), "
                    f"resolution: {self.prediction_grid_info['resolution']}"
                )
                
                if self.debug_mode:
                    max_val = np.max(self.prediction_grid)
                    mean_val = np.mean(self.prediction_grid)
                    significant_cells = np.sum(self.prediction_grid > self.prediction_threshold)
                    
                    self.get_logger().info(
                        f'Prediction Grid: {width}x{height}x{num_directions}, '
                        f'max={max_val:.4f}, mean={mean_val:.4f}, '
                        f'significant cells: {significant_cells}'
                    )
                
                self.process_update("new prediction grid")
            else:
                self.get_logger().error('Prediction grid message does not contain required metadata')
                
        except Exception as e:
            self.get_logger().error(f'Error processing prediction grid: {str(e)}')
    
    def world_to_grid(self, x, y):
        origin_x = self.prediction_grid_info.get('origin_x', 0.0)
        origin_y = self.prediction_grid_info.get('origin_y', 0.0)
        resolution = self.prediction_grid_info.get('resolution', 1.0)
        width = self.prediction_grid_info.get('width', 0)
        height = self.prediction_grid_info.get('height', 0)
        
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        grid_x = max(0, min(grid_x, width - 1))
        grid_y = max(0, min(grid_y, height - 1))
        
        return grid_x, grid_y
    
    def get_prediction_value(self, grid_x, grid_y):
        if self.prediction_grid is None:
            return 0.0
        
        if 0 <= grid_x < self.prediction_grid_info['width'] and 0 <= grid_y < self.prediction_grid_info['height']:
            max_val = np.max(self.prediction_grid[grid_y, grid_x, :])
            return max_val
        else:
            return 0.0
    
    def find_waypoints_in_prediction_grid(self):
        if self.prediction_grid is None or not self.waypoints:
            return {}
        
        waypoints_to_update = {}
        
        origin_x = self.prediction_grid_info['origin_x']
        origin_y = self.prediction_grid_info['origin_y']
        resolution = self.prediction_grid_info['resolution']
        width = self.prediction_grid_info['width']
        height = self.prediction_grid_info['height']
        
        grid_world_width = width * resolution
        grid_world_height = height * resolution
        grid_max_x = origin_x + grid_world_width
        grid_max_y = origin_y + grid_world_height
        
        if self.debug_mode and not hasattr(self, '_grid_logged'):
            self.get_logger().info(f"Grid bounds: ({origin_x}, {origin_y}) to ({grid_max_x}, {grid_max_y})")
            self.get_logger().info(f"Grid dimensions: {width}x{height} cells, {resolution}m resolution")
            self._grid_logged = True
        
        flattened_grid = np.max(self.prediction_grid, axis=2)
        
        significant_indices = np.argwhere(flattened_grid > self.prediction_threshold)
        
        waypoints_in_grid = 0
        waypoints_with_pred = 0
        
        for node_id, (wp_x, wp_y) in self.waypoints.items():
            if not (origin_x <= wp_x < grid_max_x and origin_y <= wp_y < grid_max_y):
                continue
            
            waypoints_in_grid += 1
            
            grid_x, grid_y = self.world_to_grid(wp_x, wp_y)
            
            pred_value = self.get_prediction_value(grid_x, grid_y)
            
            if pred_value > self.prediction_threshold:
                waypoints_with_pred += 1
                waypoints_to_update[node_id] = (wp_x, wp_y, pred_value)
        
        if waypoints_with_pred > 0 or (self.debug_mode and len(significant_indices) > 0):
            self.get_logger().info(
                f"Found {waypoints_with_pred}/{waypoints_in_grid} waypoints with significant predictions "
                f"(threshold: {self.prediction_threshold})"
            )
        
        return waypoints_to_update
    
    def update_waypoint_cost(self, node_id, cost):
        request = UpdateWaypointCost.Request()
        
        if node_id == -1:
            if not self.waypoints:
                self.get_logger().warn('No waypoints available for bulk update')
                return

            try:
                max_node_id = max(map(int, self.waypoints.keys()))
                max_array_size = min(max_node_id + 1, 10000)
            except ValueError:
                self.get_logger().error('Failed to determine maximum node ID from waypoints')
                return
                
            costs = [self.min_cost] * max_array_size
            
            updated_waypoints = 0
            max_cost = self.min_cost
            
            waypoints_in_prediction = self.find_waypoints_in_prediction_grid()
            waypoints_near_prediction = len(waypoints_in_prediction)
            
            if waypoints_near_prediction > 0 and self.debug_mode:
                flattened_grid = np.max(self.prediction_grid, axis=2) if self.prediction_grid is not None else None
                max_prediction = np.max(flattened_grid) if flattened_grid is not None else 0
                significant_cells = np.sum(flattened_grid > self.prediction_threshold) if flattened_grid is not None else 0
                
                self.get_logger().info(
                    f'Prediction Grid: {significant_cells} significant cells, max value: {max_prediction:.4f}'
                )
            
            for node_str_id, (x, y, pred_value) in waypoints_in_prediction.items():
                try:
                    node_id_int = int(node_str_id)
                    
                    if node_str_id in self.waypoints_in_obstacles:
                        new_cost = self.obstacle_cost
                    else:
                        cost_range = self.max_cost - self.min_cost
                        
                        normalized_pred = (pred_value - self.prediction_threshold) / (0.5 - self.prediction_threshold)
                        normalized_pred = max(0.0, min(1.0, normalized_pred))
                        
                        new_cost = self.min_cost + normalized_pred * cost_range
                    
                    if node_str_id in self.waypoints_in_obstacles:
                        pass
                    else:
                        new_cost = max(self.min_cost, min(self.max_cost, new_cost))
                    
                    new_cost = round(new_cost, 2)
                    
                    if 0 <= node_id_int < len(costs):
                        costs[node_id_int] = new_cost
                        updated_waypoints += 1
                        max_cost = max(max_cost, new_cost)
                    
                        if self.debug_mode and (normalized_pred > 0.5 or node_str_id in self.waypoints_in_obstacles):
                            if node_str_id in self.waypoints_in_obstacles:
                                self.get_logger().info(f"Node {node_id_int}: OBSTACLE, cost={new_cost:.2f}")
                            else:
                                self.get_logger().info(f"Node {node_id_int}: pred={pred_value:.4f}, normalized={normalized_pred:.4f}, cost={new_cost:.2f}")
                except ValueError:
                    self.get_logger().warn(f"Skipping invalid node ID: {node_str_id}")
                    continue
            
            for node_str_id in self.waypoints_in_obstacles:
                try:
                    if node_str_id not in waypoints_in_prediction:
                        node_id_int = int(node_str_id)
                        if 0 <= node_id_int < len(costs):
                            costs[node_id_int] = self.obstacle_cost
                            updated_waypoints += 1
                            max_cost = max(max_cost, self.obstacle_cost)
                            if self.debug_mode:
                                self.get_logger().info(f"Node {node_id_int}: OBSTACLE (outside prediction grid), cost={self.obstacle_cost:.2f}")
                except ValueError:
                    self.get_logger().warn(f"Skipping invalid obstacle node ID: {node_str_id}")
                    continue
            
            if updated_waypoints == 0:
                if self.debug_mode:
                    self.get_logger().info('No waypoints with significant predictions or in obstacles to update')
                return
                
            request.node_id = -1
            request.costs = costs
            
            self.get_logger().info(
                f'Bulk updating {updated_waypoints} waypoints, max cost: {max_cost:.2f}, '
                f'waypoints in obstacles: {len(self.waypoints_in_obstacles)}'
            )
            
            future = self.update_cost_client.call_async(request)
            future.add_done_callback(lambda f: self.update_callback(f, updated_waypoints, max_cost))
            return
        
        request.node_id = node_id
        request.cost = cost
        
        future = self.update_cost_client.call_async(request)
        future.add_done_callback(lambda f: self.update_callback(f, node_id, cost))

    def update_callback(self, future, updated_count, max_cost=None):
        try:
            response = future.result()
            if response.success:
                if isinstance(updated_count, int) and max_cost is not None:
                    success_message = f'Successfully bulk updated {updated_count} waypoint costs, max cost: {max_cost:.2f}'
                    self.get_logger().info(success_message)
                    
                    status_msg = String()
                    current_time = self.get_clock().now().seconds_nanoseconds()[0]
                    status_msg.data = f"Updated {updated_count} waypoint costs at {current_time}"
                    self.update_status_publisher.publish(status_msg)
                    
                    self.update_complete_publisher.publish(Empty())
                else:
                    node_id = updated_count
                    success_message = f'Successfully updated cost for node {node_id}'
                    self.get_logger().debug(success_message)
            else:
                error_message = f'Failed to update waypoint costs: {response.message}'
                self.get_logger().warn(error_message)
                
                status_msg = String()
                status_msg.data = f"Failed update: {response.message}"
                self.update_status_publisher.publish(status_msg)
        except Exception as e:
            error_message = f'Service call error: {str(e)}'
            self.get_logger().error(error_message)
            
            status_msg = String()
            status_msg.data = f"Error updating waypoint costs: {str(e)}"
            self.update_status_publisher.publish(status_msg)

    def update_waypoint_costs(self):
        if not self.waypoints:
            self.get_logger().warn('No waypoint data available yet')
            return
        
        self.get_logger().info(f'Processing waypoint cost update')
        
        self.update_waypoint_cost(-1, self.min_cost)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointCostUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()