#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Empty, String
import numpy as np
import math
from scipy.special import i0
import matplotlib.pyplot as plt
import os
import sys
import signal
import datetime
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

class MultibotFlowGrid(Node):
    def __init__(self):
        super().__init__('multibot_flow_grid')
        
        self.declare_parameter('num_directions', 8)
        self.declare_parameter('use_fixed_resolution', True)
        self.declare_parameter('fixed_resolution', 0.1)
        self.declare_parameter('prediction_threshold', 0.01)
        self.declare_parameter('prediction_concentration', 1.0)
        self.declare_parameter('max_propagation_distance', 10)
        self.declare_parameter('distance_decay_factor', 0.95)
        self.declare_parameter('prediction_scale_factor', 10000)
        self.declare_parameter('observation_decay_rate', 0.95)
        self.declare_parameter('observation_base_duration', 0.1)
        self.declare_parameter('save_path', '/tmp')
        self.declare_parameter('base_update_frequency', 0.2)
        self.declare_parameter('rapid_update_interval', 0.1)

        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.rapid_update_interval = self.get_parameter('rapid_update_interval').value

        self.last_full_update_time = self.get_clock().now()
        self.base_update_frequency = self.get_parameter('base_update_frequency').value

        self.num_directions = self.get_parameter('num_directions').value
        self.use_fixed_resolution = self.get_parameter('use_fixed_resolution').value
        self.fixed_resolution = self.get_parameter('fixed_resolution').value
        self.prediction_threshold = self.get_parameter('prediction_threshold').value
        self.concentration = self.get_parameter('prediction_concentration').value
        self.max_propagation_distance = self.get_parameter('max_propagation_distance').value
        self.distance_decay_factor = self.get_parameter('distance_decay_factor').value
        self.decay_rate = self.get_parameter('observation_decay_rate').value
        self.observation_base_duration = self.get_parameter('observation_base_duration').value
        self.save_path = self.get_parameter('save_path').value
        self.prediction_scale_factor = self.get_parameter('prediction_scale_factor').value
        
        self.width = 0
        self.height = 0
        self.origin_x = 0
        self.origin_y = 0
        self.resolution = self.fixed_resolution if self.use_fixed_resolution else 0
        
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
        
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile=self.config_qos)
            
        self.human_subscription = self.create_subscription(
            PoseArray,
            '/raw_human_detections',
            self.human_callback,
            self.latest_msg_qos)
        
        self.prediction_publisher = self.create_publisher(
            Float32MultiArray,
            '/prediction_grid',
            self.latest_msg_qos)
            
        self.update_complete_publisher = self.create_publisher(
            Empty,
            '/flow_grid_update_complete',
            self.latest_msg_qos)
            
        self.cost_update_subscriber = self.create_subscription(
            String,
            '/waypoint_cost_update_status',
            self.cost_update_status_callback,
            self.latest_msg_qos)
        
        self.flow_grid = None
        self.prediction_grid = None
        self.map_data = None  
        self.map_width = 0
        self.map_height = 0
        self.original_map = None  
        
        self.observation_times = None
        self.total_observation_times = None
        
        self.direction_angles = np.linspace(0, 2*np.pi, self.num_directions, endpoint=False)        

        self.timer = self.create_timer(1.0/self.base_update_frequency, self.publish_flow_grid)        
        
        self.decay_timer = self.create_timer(2.0/self.base_update_frequency, self.apply_observation_decay)
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        

    
    def map_callback(self, msg):
        self.original_map = msg
        
        if not self.use_fixed_resolution:
            self.resolution = msg.info.resolution
        
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.map_data = msg.data
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        if self.use_fixed_resolution and self.fixed_resolution != msg.info.resolution:
            world_width = msg.info.width * msg.info.resolution
            world_height = msg.info.height * msg.info.resolution
            self.width = int(world_width / self.fixed_resolution)
            self.height = int(world_height / self.fixed_resolution)
        else:
            self.width = self.map_width
            self.height = self.map_height
        
        self.flow_grid = np.zeros((self.height, self.width, self.num_directions))
        self.prediction_grid = np.zeros((self.height, self.width, self.num_directions))

        self.observation_times = np.zeros((self.height, self.width, self.num_directions))
        self.total_observation_times = np.zeros((self.height, self.width))
    
    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)        
        grid_x = max(0, min(grid_x, self.width - 1))
        grid_y = max(0, min(grid_y, self.height - 1))
        
        return grid_x, grid_y
    
    def angle_to_direction_index(self, angle):
        while angle < 0:
            angle += 2 * np.pi
        while angle >= 2 * np.pi:
            angle -= 2 * np.pi
        
        diffs = np.abs(np.array(self.direction_angles) - angle)
        return np.argmin(diffs)
    
    def direction_index_to_angle(self, index):
        return self.direction_angles[index]
    
    def is_cell_occupied(self, grid_x, grid_y):
        if self.map_data is None:
            return False

        if self.use_fixed_resolution and self.fixed_resolution != self.original_map.info.resolution:
            map_res = self.original_map.info.resolution
            world_x = grid_x * self.resolution + self.origin_x
            world_y = grid_y * self.resolution + self.origin_y
            map_x = int((world_x - self.origin_x) / map_res)
            map_y = int((world_y - self.origin_y) / map_res)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                idx = map_y * self.map_width + map_x
                if idx < len(self.map_data) and self.map_data[idx] > 50:
                    return True
        else:
            if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                idx = grid_y * self.map_width + grid_x
                if idx < len(self.map_data) and self.map_data[idx] > 50:
                    return True
        
        return False
    
    def calculate_proximity_weight(self, distance, max_distance):
        observation_min_weight = 0.1
        decay_range = 1.0 - observation_min_weight
        return max(observation_min_weight, 1.0 - decay_range * (distance / max_distance))

    def human_callback(self, msg):
        if self.flow_grid is None:
            return

        header_stamp = msg.header.stamp
        frame_id = msg.header.frame_id
        
        for i, pose in enumerate(msg.poses):
            try:
                cells_to_update = set()
                
                x = pose.position.x
                y = pose.position.y                
                qx = pose.orientation.x
                qy = pose.orientation.y
                qz = pose.orientation.z
                qw = pose.orientation.w
                
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                grid_x, grid_y = self.world_to_grid(x, y)
                
                if self.is_cell_occupied(grid_x, grid_y):
                    continue
                
                dir_idx = self.angle_to_direction_index(yaw)
                
                observation_duration = self.observation_base_duration
                self.observation_times[grid_y, grid_x, dir_idx] += observation_duration
                self.total_observation_times[grid_y, grid_x] += observation_duration

                cells_to_update.add((grid_x, grid_y, dir_idx))
                
                propagation_radius = 4
                for dy in range(-propagation_radius, propagation_radius + 1):
                    for dx in range(-propagation_radius, propagation_radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = grid_x + dx, grid_y + dy
                        
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.is_cell_occupied(nx, ny):
                                continue

                            distance = math.sqrt(dx**2 + dy**2)
                            if distance <= propagation_radius:
                                weight = self.calculate_proximity_weight(distance, propagation_radius)
                                
                                weighted_duration = observation_duration * weight
                                self.observation_times[ny, nx, dir_idx] += weighted_duration
                                self.total_observation_times[ny, nx] += weighted_duration
                                cells_to_update.add((nx, ny, dir_idx))
                
                min_y = max(0, grid_y - propagation_radius)
                max_y = min(self.height, grid_y + propagation_radius + 1)
                min_x = max(0, grid_x - propagation_radius)
                max_x = min(self.width, grid_x + propagation_radius + 1)
                
                for y in range(min_y, max_y):
                    for x in range(min_x, max_x):
                        if self.total_observation_times[y, x] > 0:
                            for d in range(self.num_directions):
                                self.flow_grid[y, x, d] = (
                                    self.observation_times[y, x, d] / 
                                    self.total_observation_times[y, x]
                                )
                
                for grid_x, grid_y, dir_idx in cells_to_update:
                    self.update_prediction_grid(grid_x, grid_y, dir_idx)
                
            except Exception as e:
                pass
        
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_full_update_time).nanoseconds / 1e9
        
        if time_diff >= self.rapid_update_interval:
            self.publish_flow_grid()
            self.update_complete_publisher.publish(Empty())
            self.last_full_update_time = current_time

    def is_wall_between(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2:
            return False
        
        steep = abs(y2 - y1) > abs(x2 - x1)
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        dx = x2 - x1
        dy = abs(y2 - y1)
        
        error = dx // 2
        y = y1
        y_step = 1 if y1 < y2 else -1
        
        for x in range(x1, x2 + 1):
            if (steep and (y, x) == (y1, x1)) or (not steep and (x, y) == (x1, y1)):
                pass
            elif (steep and (y, x) == (y2, x2)) or (not steep and (x, y) == (x2, y2)):
                pass
            elif self.is_cell_occupied(y if steep else x, x if steep else y):
                return True
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
        
        return False

    def von_mises(self, theta, mu, kappa):
        return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))

    def update_prediction_grid(self, center_x, center_y, observed_dir_idx):
        min_x = max(0, center_x - self.max_propagation_distance)
        max_x = min(self.width, center_x + self.max_propagation_distance + 1)
        min_y = max(0, center_y - self.max_propagation_distance)
        max_y = min(self.height, center_y + self.max_propagation_distance + 1)
        
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                self.prediction_grid[y, x, :] = 0.0
        
        observed_flow_prob = self.flow_grid[center_y, center_x, observed_dir_idx]
        self.prediction_grid[center_y, center_x, observed_dir_idx] = observed_flow_prob
        
        cells_to_process = [(center_x, center_y)]
        processed_cells = set([(center_x, center_y)])
        
        observed_angle = self.direction_index_to_angle(observed_dir_idx)
        
        while cells_to_process:
            x, y = cells_to_process.pop(0)
            
            if self.is_cell_occupied(x, y):
                continue
            
            directions_by_prob = np.argsort(-self.prediction_grid[y, x])
            
            for dir_idx in directions_by_prob:
                if self.prediction_grid[y, x, dir_idx] <= 0:
                    continue
                
                dir_angle = self.direction_index_to_angle(dir_idx)
                dx = round(math.cos(dir_angle))
                dy = round(math.sin(dir_angle))
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < self.width and 0 <= ny < self.height) or (nx, ny) in processed_cells:
                    continue
                
                if self.is_wall_between(x, y, nx, ny):
                    continue
                
                processed_cells.add((nx, ny))
                
                for neighbor_dir in range(self.num_directions):
                    neighbor_angle = self.direction_index_to_angle(neighbor_dir)
                    
                    von_mises_prob = self.von_mises(neighbor_angle, dir_angle, self.concentration)
                    
                    total_prob_sum = sum(self.von_mises(self.direction_index_to_angle(i), dir_angle, self.concentration) 
                                       for i in range(self.num_directions))
                    
                    normalized_prob = von_mises_prob
                    
                    pred_value = normalized_prob * self.prediction_grid[y, x, dir_idx]
                    
                    if pred_value > self.prediction_grid[ny, nx, neighbor_dir]:
                        self.prediction_grid[ny, nx, neighbor_dir] = pred_value
                
                if np.max(self.prediction_grid[ny, nx]) > self.prediction_threshold:
                    cells_to_process.append((nx, ny))
                
                cells_to_process.sort(key=lambda cell: -np.max(self.prediction_grid[cell[1], cell[0]]))

    def apply_observation_decay(self):
        if self.observation_times is None:
            return
            
        self.observation_times *= self.decay_rate
        self.total_observation_times *= self.decay_rate
        
        if self.prediction_grid is not None:
            self.prediction_grid *= self.decay_rate
        
        mask = self.total_observation_times > 0
        for d in range(self.num_directions):
            self.flow_grid[:, :, d] = np.where(
                mask,
                self.observation_times[:, :, d] / self.total_observation_times,
                0
            )
        
        self.publish_flow_grid()
    
    def publish_flow_grid(self):
        self.publish_prediction_grid()
        self.update_complete_publisher.publish(Empty())

    def publish_prediction_grid(self):
        if self.prediction_grid is None:
            return
        
        grid_msg = Float32MultiArray()

        metadata = [
            float(self.origin_x),
            float(self.origin_y),
            float(self.resolution),
        ]

        scaled_prediction = self.prediction_grid * self.prediction_scale_factor
        combined_data = metadata + scaled_prediction.flatten().tolist()
        
        grid_msg.layout.dim.append(MultiArrayDimension(
            label="height", size=self.height, 
            stride=self.width * self.num_directions))
        grid_msg.layout.dim.append(MultiArrayDimension(
            label="width", size=self.width, 
            stride=self.num_directions))
        grid_msg.layout.dim.append(MultiArrayDimension(
            label="directions", size=self.num_directions, 
            stride=1))
        grid_msg.data = combined_data
        self.prediction_publisher.publish(grid_msg)

    def signal_handler(self, sig, frame):
        try:
            self.save_flow_grid_image()
        except Exception as e:
            pass
        rclpy.shutdown()
        sys.exit(0)
        
    def save_flow_grid_image(self):
        if self.flow_grid is None:
            return
            
        try:
            os.makedirs(self.save_path, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
            filename = f"multibot_flow_grid_{timestamp}.png"
            filepath = os.path.join(self.save_path, filename)
            
            if self.flow_grid.size == 0 or self.prediction_grid.size == 0:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "No data collected", 
                         horizontalalignment='center', verticalalignment='center')
                plt.savefig(filepath)
                plt.close()
                return

            plt.switch_backend('Agg')
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            presence = np.sum(self.flow_grid, axis=2)
            max_presence = np.max(presence) if presence.size > 0 else 0
            if max_presence > 0:
                presence = presence / max_presence           
          
            try:
                im1 = axes[0].imshow(presence, cmap='viridis', interpolation='nearest')
                axes[0].set_title('Human Presence Likelihood (Mpres)')
                plt.colorbar(im1, ax=axes[0])
            except Exception as e:
                axes[0].text(0.5, 0.5, "Error plotting presence", 
                         horizontalalignment='center', verticalalignment='center')
            
            try:
                prediction = np.max(self.prediction_grid, axis=2)
                im2 = axes[1].imshow(prediction, cmap='plasma', interpolation='nearest')
                axes[1].set_title('Predicted Human Flow (Mpred)')
                plt.colorbar(im2, ax=axes[1])
            except Exception as e:
                axes[1].text(0.5, 0.5, "Error plotting prediction", 
                         horizontalalignment='center', verticalalignment='center')
            
            for ax in axes:
                ax.set_aspect('equal')
                ax.invert_yaxis()
            
            time_str = timestamp
                
            plt.suptitle(f'Multibot Flow Grid State at {time_str}', fontsize=16)
            plt.tight_layout() 
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)            
            
        except Exception as e:
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error saving image: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center')
                error_path = os.path.join(self.save_path, f"multibot_flow_grid_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(error_path)
                plt.close()
            except Exception as nested_e:
                pass


def main(args=None):
    rclpy.init(args=args)
    flow_grid_node = MultibotFlowGrid()    
    try:
        rclpy.spin(flow_grid_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        try:
            flow_grid_node.save_flow_grid_image()
        except Exception as e:
            pass
        try:
            flow_grid_node.destroy_node()
        except Exception as e:
            pass            
        try:
            rclpy.shutdown()
        except Exception as e:
            pass

if __name__ == '__main__':
    main()