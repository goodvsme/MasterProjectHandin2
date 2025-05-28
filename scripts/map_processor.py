#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import os
import yaml
import math
import time
import threading
from PIL import Image
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Twist, PoseWithCovarianceStamped
from nav_msgs.srv import GetMap
from std_srvs.srv import Trigger

class MapProcessor(Node):
    def __init__(self):
        super().__init__('map_processor')
        
        package_dir = get_package_share_directory('master_project2')
        self.use_sim_time = self.get_parameter('use_sim_time').value
        
        self.declare_parameters(namespace='',
            parameters=[
                ('database_path', os.path.join(package_dir, 'db/mapping.db')),
                ('humans_database_path', os.path.join(package_dir, 'db/humans.db')),
                ('temp_pgm_path', os.path.join(package_dir, 'maps/temp_map.pgm')),
                ('temp_yaml_path', os.path.join(package_dir, 'maps/temp_map.yaml')),
                ('updated_pgm_path', os.path.join(package_dir, 'maps/updated_map.pgm')),
                ('updated_yaml_path', os.path.join(package_dir, 'maps/updated_map.yaml')),
                ('cell_size', 0.1),
                ('d_threshold', 0.15),
                ('confidence_threshold', 7),
                ('batch_size', 50),
                ('processing_rate', 1.0),
                ('map_update_rate', 5.0),
                ('verification_threshold', 5),
                ('max_detection_range', 5.0),
                ('human_radius', 1.0),
                ('save_debug_maps', False),
                ('debug_map_interval', 10),
                ('debug_map_path', os.path.join(package_dir, 'maps/debug/')),
                ('laser_has_roll_180', False),
                ('laser_x_offset', 0.5059),
                ('laser_y_offset', 0.1),
                ('max_scan_pose_time_diff', 0.1)
            ])
        
        self.db_path = self.get_parameter('database_path').value
        self.humans_db_path = self.get_parameter('humans_database_path').value
        self.batch_size = self.get_parameter('batch_size').value
        self.processing_rate = self.get_parameter('processing_rate').value
        self.map_update_rate = self.get_parameter('map_update_rate').value
        self.max_scan_pose_time_diff = self.get_parameter('max_scan_pose_time_diff').value
        self.max_detection_range = self.get_parameter('max_detection_range').value
        self.human_radius = self.get_parameter('human_radius').value
        self.laser_has_roll_180 = self.get_parameter('laser_has_roll_180').value
        self.laser_x_offset = self.get_parameter('laser_x_offset').value
        self.laser_y_offset = self.get_parameter('laser_y_offset').value
        
        self.db_lock = threading.Lock()
        self.init_database()
        
        self.original_map = None
        self.current_map = None
        self.verification_counter = 0
        self.human_positions = []
        
        self.map_client = self.create_client(GetMap, '/map_server/map')
        
        self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            1
        )
        
        self.create_service(
            Trigger, 
            'process_all_data', 
            self.process_all_data_callback
        )
        
        self.create_service(
            Trigger, 
            'reset_cell_confidence', 
            self.reset_cell_confidence_callback
        )
        
        self.create_timer(1.0/self.processing_rate, self.process_batch)
        self.create_timer(self.map_update_rate, self.update_map)
        self.create_timer(30.0, self.log_stats)
        
        if self.get_parameter('save_debug_maps').value:
            os.makedirs(self.get_parameter('debug_map_path').value, exist_ok=True)
        
        self.update_counter = 0
        
        self.stats = {
            'scans_processed': 0,
            'scans_skipped': 0,
            'points_processed': 0,
            'obstacles_detected': 0,
            'free_spaces_detected': 0,
            'processing_time': 0.0,
            'avg_processing_time': 0.0,
            'db_errors': 0
        }
        
        if not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Map service not available, will keep trying')
            self.create_timer(5.0, self.load_initial_map)
        else:
            self.load_initial_map()
        
        self.get_logger().info('MapProcessor initialized')
        self.get_logger().info(f'Mapping database path: {self.db_path}')
        self.get_logger().info(f'Humans database path: {self.humans_db_path}')
        self.get_logger().info(f'Using simulation time: {self.use_sim_time}')
    
    def init_database(self):
        try:
            with self.db_lock:
                if hasattr(self, 'db_conn'):
                    try:
                        self.db_conn.close()
                    except Exception:
                        pass
                
                if hasattr(self, 'humans_db_conn'):
                    try:
                        self.humans_db_conn.close()
                    except Exception:
                        pass
                
                self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.db_cursor = self.db_conn.cursor()
                
                self.humans_db_conn = sqlite3.connect(self.humans_db_path, check_same_thread=False)
                self.humans_db_cursor = self.humans_db_conn.cursor()
                
                self.db_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS status_flags (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        timestamp REAL
                    )
                ''')
                
                self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scan_data'")
                if not self.db_cursor.fetchone():
                    self.get_logger().warn("scan_data table does not exist yet")
                    self.db_cursor.execute('''
                        CREATE TABLE IF NOT EXISTS scan_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            stamp_sec INTEGER,
                            stamp_nanosec INTEGER,
                            frame_id TEXT,
                            angle_min REAL,
                            angle_max REAL,
                            angle_increment REAL,
                            range_min REAL,
                            range_max REAL,
                            ranges BLOB,
                            processed INTEGER DEFAULT 0
                        )
                    ''')
                    
                    self.db_cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_scan_time ON scan_data(stamp_sec, stamp_nanosec)
                    ''')
                    
                    self.db_cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_scan_processed ON scan_data(processed)
                    ''')
                
                self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_data'")
                if not self.db_cursor.fetchone():
                    self.get_logger().warn("pose_data table does not exist yet")
                    self.db_cursor.execute('''
                        CREATE TABLE IF NOT EXISTS pose_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            stamp_sec INTEGER,
                            stamp_nanosec INTEGER,
                            frame_id TEXT,
                            position_x REAL,
                            position_y REAL,
                            position_z REAL,
                            orientation_x REAL,
                            orientation_y REAL,
                            orientation_z REAL,
                            orientation_w REAL
                        )
                    ''')
                    
                    self.db_cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_pose_time ON pose_data(stamp_sec, stamp_nanosec)
                    ''')
                
                self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cell_confidence'")
                if not self.db_cursor.fetchone():
                    self.get_logger().warn("cell_confidence table does not exist yet")
                    self.db_cursor.execute('''
                        CREATE TABLE IF NOT EXISTS cell_confidence (
                            cell_x INTEGER NOT NULL,
                            cell_y INTEGER NOT NULL,
                            occupied_count INTEGER DEFAULT 0,
                            free_count INTEGER DEFAULT 0,
                            ros_time_sec INTEGER,
                            ros_time_nanosec INTEGER,
                            PRIMARY KEY (cell_x, cell_y)
                        )
                    ''')
                
                self.db_conn.commit()
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Database connection error: {str(e)}')
            self.stats['db_errors'] += 1
            self.create_timer(5.0, self.init_database, oneshot=True)
    
    def load_initial_map(self):
        if not self.map_client.service_is_ready():
            self.get_logger().info('Map service not available, will try again later')
            return
            
        try:
            future = self.map_client.call_async(GetMap.Request())
            future.add_done_callback(self.map_callback_async)
        except Exception as e:
            self.get_logger().error(f'Error requesting map: {str(e)}')
    
    def map_callback_async(self, future):
        try:
            response = future.result()
            self.map_callback(response.map)
        except Exception as e:
            self.get_logger().error(f'Map service call failed: {str(e)}')
    
    def map_callback(self, msg):
        if self.original_map is None:
            self.original_map = msg
            self.get_logger().info('Original map received')
        
        self.current_map = msg
        self.get_logger().info('Map updated')
        
        self.save_map(temp=True)
        self.save_map(temp=False)
    
    def get_robot_turning_status(self):
        try:
            with self.db_lock:
                self.db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='status_flags'")
                if not self.db_cursor.fetchone():
                    self.get_logger().debug("status_flags table does not exist yet, assuming robot is not turning")
                    return False
                    
                self.db_cursor.execute('''
                    SELECT value, timestamp FROM status_flags
                    WHERE key = 'is_turning'
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''')
                
                result = self.db_cursor.fetchone()
                if result:
                    value, timestamp = result
                    current_time = self.get_clock().now().nanoseconds * 1e-9
                    if current_time - timestamp < 5.0:
                        return value == 'true'
                
                return False
                
        except sqlite3.Error as e:
            self.get_logger().warn(f'Error checking turning status: {str(e)} - assuming robot is not turning')
            self.stats['db_errors'] += 1
            return False
    
    def process_batch(self):
        if self.current_map is None:
            return
            
        try:
            batch_start_time = time.time()
            
            with self.db_lock:
                self.db_cursor.execute('''
                    SELECT id, stamp_sec, stamp_nanosec, frame_id,
                           angle_min, angle_max, angle_increment,
                           range_min, range_max, ranges
                    FROM scan_data
                    WHERE processed = 0
                    ORDER BY stamp_sec ASC, stamp_nanosec ASC, id ASC
                    LIMIT ?
                ''', (self.batch_size,))
                
                scans = self.db_cursor.fetchall()
            
            if not scans:
                return
                
            self.get_logger().debug(f'Processing batch of {len(scans)} scans')
            
            self.update_human_positions()
            
            for scan in scans:
                scan_id, stamp_sec, stamp_nanosec = scan[0], scan[1], scan[2]
                
                pose = self.find_matching_pose(stamp_sec, stamp_nanosec)
                
                if pose is None:
                    with self.db_lock:
                        self.db_cursor.execute('''
                            UPDATE scan_data
                            SET processed = 1
                            WHERE id = ?
                        ''', (scan_id,))
                        self.db_conn.commit()
                    
                    self.stats['scans_skipped'] += 1
                    continue
                
                self.process_scan(scan, pose)
                
                with self.db_lock:
                    self.db_cursor.execute('''
                        UPDATE scan_data
                        SET processed = 1
                        WHERE id = ?
                    ''', (scan_id,))
                    self.db_conn.commit()
                
                self.stats['scans_processed'] += 1
            
            if len(scans) > 0:
                with self.db_lock:
                    self.db_cursor.execute('SELECT COUNT(*) FROM scan_data')
                    total_scans = self.db_cursor.fetchone()[0]
                    
                    self.db_cursor.execute('SELECT COUNT(*) FROM scan_data WHERE processed = 0')
                    unprocessed_scans = self.db_cursor.fetchone()[0]
                    
                    self.db_cursor.execute('SELECT COUNT(*) FROM cell_confidence')
                    confidence_cells = self.db_cursor.fetchone()[0]
                
                self.get_logger().info(
                    f"Processed batch of {len(scans)} scans. " +
                    f"Database has {total_scans} total scans, {unprocessed_scans} unprocessed. " +
                    f"Cell confidence table has {confidence_cells} cells."
    )
            
            processing_time = time.time() - batch_start_time
            self.stats['processing_time'] = processing_time
            
            if self.stats['avg_processing_time'] == 0.0:
                self.stats['avg_processing_time'] = processing_time
            else:
                self.stats['avg_processing_time'] = (
                    0.9 * self.stats['avg_processing_time'] + 0.1 * processing_time
                )
                
            if len(scans) > 0:
                self.get_logger().debug(
                    f'Processed {len(scans)} scans in {processing_time:.3f}s '
                    f'({processing_time/len(scans)*1000:.1f}ms/scan)'
                )
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Error processing batch: {str(e)}')
            self.stats['db_errors'] += 1
            self.init_database()
    
    def update_human_positions(self):
        try:
            current_time = self.get_clock().now()
            current_sec = current_time.seconds_nanoseconds()[0]
            current_nanosec = current_time.seconds_nanoseconds()[1]
            recent_window = 20.0
            
            with sqlite3.connect(self.humans_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    WITH ranked_detections AS (
                        SELECT 
                            position_x, position_y,
                            unique_id,
                            ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY ros_time_sec DESC, ros_time_nanosec DESC) as rn
                        FROM human_detections
                        WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                        AND unique_id IS NOT NULL
                    )
                    SELECT position_x, position_y
                    FROM ranked_detections
                    WHERE rn = 1
                ''', (current_sec, current_nanosec, recent_window))
                
                self.human_positions = [(row[0], row[1]) for row in cursor.fetchall()]
                
                cursor.execute('''
                    SELECT position_x, position_y
                    FROM human_detections
                    WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                    AND unique_id IS NULL
                ''', (current_sec, current_nanosec, recent_window))
                
                self.human_positions.extend([(row[0], row[1]) for row in cursor.fetchall()])
                
                self.get_logger().debug(f'Using {len(self.human_positions)} human positions from humans.db')
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Error getting human positions from humans.db: {str(e)}')
            self.human_positions = []
    
    def find_matching_pose(self, scan_sec, scan_nanosec):
        try:
            scan_time = scan_sec + scan_nanosec * 1e-9
            
            with self.db_lock:
                self.db_cursor.execute('''
                    SELECT position_x, position_y, position_z,
                           orientation_x, orientation_y, orientation_z, orientation_w
                    FROM pose_data
                    WHERE stamp_sec = ? AND stamp_nanosec = ?
                    LIMIT 1
                ''', (scan_sec, scan_nanosec))
                
                result = self.db_cursor.fetchone()
                if result:
                    return self.create_pose_from_db(result)
                
                self.db_cursor.execute('''
                    SELECT position_x, position_y, position_z,
                           orientation_x, orientation_y, orientation_z, orientation_w,
                           stamp_sec, stamp_nanosec, ABS(? - stamp_sec - stamp_nanosec * 1e-9) as time_diff
                    FROM pose_data
                    WHERE frame_id = 'map' AND ABS(? - stamp_sec - stamp_nanosec * 1e-9) <= ?
                    ORDER BY time_diff ASC
                    LIMIT 1
                ''', (scan_time, scan_time, self.max_scan_pose_time_diff))
                
                result = self.db_cursor.fetchone()
                if result:
                    time_diff = result[9]
                    self.get_logger().debug(f'Found AMCL pose with time diff: {time_diff*1000:.1f}ms')
                    return self.create_pose_from_db(result[:7])
                
                self.db_cursor.execute('''
                    SELECT position_x, position_y, position_z,
                           orientation_x, orientation_y, orientation_z, orientation_w,
                           stamp_sec, stamp_nanosec, ABS(? - stamp_sec - stamp_nanosec * 1e-9) as time_diff
                    FROM pose_data
                    WHERE frame_id != 'map' AND ABS(? - stamp_sec - stamp_nanosec * 1e-9) <= ?
                    ORDER BY time_diff ASC
                    LIMIT 1
                ''', (scan_time, scan_time, self.max_scan_pose_time_diff))
                
                result = self.db_cursor.fetchone()
                if result:
                    time_diff = result[9]
                    self.get_logger().debug(f'Found odom pose with time diff: {time_diff*1000:.1f}ms')
                    return self.create_pose_from_db(result[:7])
                
                return None
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Error finding matching pose: {str(e)}')
            self.stats['db_errors'] += 1
            return None
    
    def create_pose_from_db(self, db_result):
        pose = Pose()
        pose.position.x = db_result[0]
        pose.position.y = db_result[1]
        pose.position.z = db_result[2]
        pose.orientation.x = db_result[3]
        pose.orientation.y = db_result[4]
        pose.orientation.z = db_result[5]
        pose.orientation.w = db_result[6]
        return pose
    
    def process_scan(self, scan, pose):
        scan_id, _, _, frame_id = scan[0], scan[1], scan[2], scan[3]
        angle_min, angle_max, angle_increment = scan[4], scan[5], scan[6]
        range_min, range_max, ranges_blob = scan[7], scan[8], scan[9]
        
        try:
            ranges = np.frombuffer(ranges_blob, dtype=np.float16)
        except:
            ranges = np.frombuffer(ranges_blob, dtype=np.float32)
        
        robot_x = pose.position.x
        robot_y = pose.position.y
        
        q = [
            pose.orientation.x,
            pose.orientation.y, 
            pose.orientation.z,
            pose.orientation.w
        ]
        
        _, _, robot_yaw = self.quaternion_to_euler(q)
        
        all_changed_points = []
        points_processed = 0
        
        for i, range_val in enumerate(ranges):
            if range_val < range_min or range_val > self.max_detection_range or not np.isfinite(range_val):
                continue
                
            scan_angle = angle_min + i * angle_increment
            
            if self.laser_has_roll_180:
                point_laser_x = -range_val * np.cos(scan_angle)
                point_laser_y = -range_val * np.sin(scan_angle)
            else:
                point_laser_x = range_val * np.cos(scan_angle)
                point_laser_y = -range_val * np.sin(scan_angle)
            
            
            cos_yaw = np.cos(robot_yaw)
            sin_yaw = np.sin(robot_yaw)
            laser_map_x = robot_x + (self.laser_x_offset * cos_yaw - self.laser_y_offset * sin_yaw)
            laser_map_y = robot_y + (self.laser_x_offset * sin_yaw + self.laser_y_offset * cos_yaw)

            point_map_x = laser_map_x + (point_laser_x * cos_yaw - point_laser_y * sin_yaw)
            point_map_y = laser_map_y + (point_laser_x * sin_yaw + point_laser_y * cos_yaw)
            
            measured_point = (point_map_x, point_map_y)
            points_processed += 1
            
            if self.near_human(measured_point):
                continue
            
            expected_points = self.calculate_expected_points(scan_angle, pose)
            
            if self.detect_change(measured_point, expected_points):
                all_changed_points.append(measured_point)
        
        if all_changed_points:
            occupied_cells = []
            free_cells = []
            
            for point in all_changed_points:
                cells = self.get_cells_along_beam(point, pose)
                for i, cell in enumerate(cells):
                    grid_x, grid_y = self.world_to_grid(cell)
                    if i == len(cells)-1:
                        occupied_cells.append((grid_x, grid_y))
                    else:
                        free_cells.append((grid_x, grid_y))
            
            self.batch_update_cell_confidence(occupied_cells, free_cells)
            
            self.stats['obstacles_detected'] += len(occupied_cells)
            self.stats['free_spaces_detected'] += len(free_cells)
        
        self.stats['points_processed'] += points_processed
    
    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def calculate_expected_points(self, angle, pose):
        expected_points = []
        n_exp = 3
        delta_theta = 0.1
        
        q = pose.orientation
        
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        for l in range(-n_exp, n_exp+1):
            perturbed_angle = angle + l * delta_theta
            global_angle = yaw + perturbed_angle
            
            expected_range = self.raycast(global_angle, self.max_detection_range, pose)
            if expected_range is not None:
                x = pose.position.x + expected_range * np.cos(global_angle)
                y = pose.position.y - expected_range * np.sin(global_angle)
                expected_points.append((x, y))
                
        return expected_points
    
    def raycast(self, angle, max_range, pose):
        res = self.current_map.info.resolution
        current_x = pose.position.x
        current_y = pose.position.y
        
        for r in np.arange(0, max_range, res):
            x = current_x + r * np.cos(angle)
            y = current_y - r * np.sin(angle)
            
            grid_x, grid_y = self.world_to_grid((x, y))
            idx = self.grid_to_index(grid_x, grid_y)
            if idx is not None and idx < len(self.current_map.data) and self.current_map.data[idx] == 100:
                return r
        return max_range
    
    def get_cells_along_beam(self, end_point, pose):
        cells = []
        resolution = self.current_map.info.resolution
        start_x = pose.position.x
        start_y = pose.position.y
        
        dx = end_point[0] - start_x
        dy = end_point[1] - start_y
        steps = max(abs(dx), abs(dy)) / resolution
        
        for i in range(int(steps) + 1):
            x = start_x + i*dx/steps if steps > 0 else start_x
            y = start_y + i*dy/steps if steps > 0 else start_y
            cell = (round(x/resolution)*resolution, round(y/resolution)*resolution)
            if cell not in cells:
                cells.append(cell)
        return cells
    
    def detect_change(self, measured_point, expected_points):
        if not expected_points:
            return False
        min_distance = min([np.linalg.norm(np.array(measured_point) - np.array(p)) 
                          for p in expected_points])
        return min_distance > self.get_parameter('d_threshold').value
    
    def batch_update_cell_confidence(self, occupied_cells, free_cells):
        try:
            ros_time_sec, ros_time_nanosec = self.get_current_ros_time()
            occupied_updates = len(occupied_cells)
            free_updates = len(free_cells)
            
            with self.db_lock:
                self.db_cursor.execute('BEGIN TRANSACTION')
                
                for grid_x, grid_y in occupied_cells:
                    self.db_cursor.execute('''
                        INSERT INTO cell_confidence (cell_x, cell_y, occupied_count, free_count, ros_time_sec, ros_time_nanosec)
                        VALUES (?, ?, 1, 0, ?, ?)
                        ON CONFLICT(cell_x, cell_y) DO UPDATE SET
                            occupied_count = occupied_count + 1,
                            free_count = CASE WHEN free_count > 0 THEN free_count - 1 ELSE 0 END,
                            ros_time_sec = ?,
                            ros_time_nanosec = ?
                    ''', (grid_x, grid_y, ros_time_sec, ros_time_nanosec, ros_time_sec, ros_time_nanosec))
                                
                for grid_x, grid_y in free_cells:
                    self.db_cursor.execute('''
                        INSERT INTO cell_confidence (cell_x, cell_y, occupied_count, free_count, ros_time_sec, ros_time_nanosec)
                        VALUES (?, ?, 0, 1, ?, ?)
                        ON CONFLICT(cell_x, cell_y) DO UPDATE SET
                            free_count = free_count + 1,
                            occupied_count = CASE WHEN occupied_count > 0 THEN occupied_count - 1 ELSE 0 END,
                            ros_time_sec = ?,
                            ros_time_nanosec = ?
                    ''', (grid_x, grid_y, ros_time_sec, ros_time_nanosec, ros_time_sec, ros_time_nanosec))
                
                self.db_conn.commit()



            self.get_logger().debug(
                f"Updated cell confidence: {occupied_updates} occupied cells, " +
                f"{free_updates} free cells"
)
                
        except sqlite3.Error as e:
            with self.db_lock:
                if self.db_conn.in_transaction:
                    self.db_conn.rollback()
            self.get_logger().error(f"Database batch update error: {str(e)}")
            self.stats['db_errors'] += 1
    
    def update_map(self):
        if self.current_map is None or self.original_map is None:
            return
            
        self.update_counter += 1
        if self.update_counter % 10 == 0:
            self.get_logger().info(f'Map update cycle: {self.update_counter}')
        
        try:
            conf_th = self.get_parameter('confidence_threshold').value
            
            with self.db_lock:
                self.db_cursor.execute('''
                    SELECT cell_x, cell_y, occupied_count, free_count 
                    FROM cell_confidence
                    WHERE occupied_count >= ? OR free_count >= ?
                ''', (conf_th, conf_th))
                
                rows = self.db_cursor.fetchall()
            
            updated_map = OccupancyGrid()
            updated_map.header = self.current_map.header
            updated_map.header.stamp = self.get_clock().now().to_msg()
            updated_map.info = self.current_map.info
            updated_map.data = list(self.current_map.data)
            
            updates_applied = 0
            
            width = updated_map.info.width
            height = updated_map.info.height
            for y in range(height):
                for x in range(width):
                    idx = y * width + x
                    if idx < len(self.original_map.data) and idx < len(updated_map.data):
                        if self.original_map.data[idx] == 100:
                            updated_map.data[idx] = 100
            
            for cell_x, cell_y, occ, free in rows:
                idx = self.grid_to_index(cell_x, cell_y)
                if idx is None:
                    continue
                    
                if idx < len(self.original_map.data) and self.original_map.data[idx] == 100:
                    continue
                    
                if occ >= conf_th:
                    updated_map.data[idx] = 100
                    updates_applied += 1
                elif free >= conf_th:
                    updated_map.data[idx] = 0
                    updates_applied += 1
            
            self.current_map = updated_map
            
            if updates_applied > 0:
                self.get_logger().debug(f'Applied {updates_applied} updates to map')
            
            if self.get_parameter('save_debug_maps').value and self.update_counter % self.get_parameter('debug_map_interval').value == 0:
                self.save_debug_map()
                    
            self.save_map(temp=True)
            
            if self.verification_counter >= self.get_parameter('verification_threshold').value:
                self.get_logger().info(f'Verification threshold reached ({self.verification_counter}), saving final map')
                self.save_map(temp=False)
                self.verification_counter = 0
            else:
                self.verification_counter += 1
            if self.get_parameter('save_debug_maps').value and self.update_counter % 5 == 0:
                debug_path = os.path.join(self.get_parameter('debug_map_path').value, f"confidence_dump_{self.update_counter}.txt")
                try:
                    with open(debug_path, 'w') as f:
                        with self.db_lock:
                            self.db_cursor.execute('''
                                SELECT cell_x, cell_y, occupied_count, free_count 
                                FROM cell_confidence
                                WHERE occupied_count >= 1 OR free_count >= 1
                                ORDER BY cell_x, cell_y
                            ''')
                            
                            f.write("cell_x,cell_y,occupied_count,free_count\n")
                            for row in self.db_cursor.fetchall():
                                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
                    
                    self.get_logger().info(f"Saved confidence dump to {debug_path}")
                except Exception as e:
                    self.get_logger().error(f"Error saving confidence dump: {str(e)}")
        except Exception as e:
            self.get_logger().error(f'Error updating map: {str(e)}')
    
    def get_current_ros_time(self):
        now = self.get_clock().now().to_msg()
        return (now.sec, now.nanosec)
    
    def save_map(self, temp=True):
        if temp:
            self.save_temp_map()
        else:
            self.save_final_map()

    def save_temp_map(self):
        try:
            with self.db_lock:
                self.db_cursor.execute('SELECT cell_x, cell_y, occupied_count, free_count FROM cell_confidence')
                rows = self.db_cursor.fetchall()
                
            width = self.current_map.info.width
            height = self.current_map.info.height
            img_data = np.full((height, width), 205, dtype=np.uint8)
            
            for cell_x, cell_y, occ, free in rows:
                total = occ + free
                if total == 0:
                    continue
                    
                if occ > free:
                    confidence = min(255, int(255 * (occ / total)))
                else:
                    confidence = max(0, 255 - int(255 * (free / total)))
                
                if 0 <= cell_y < height and 0 <= cell_x < width:
                    img_data[cell_y, cell_x] = confidence

            image = Image.fromarray(img_data)
            image.save(self.get_parameter('temp_pgm_path').value)
            
            yaml_data = {
                'image': os.path.basename(self.get_parameter('temp_pgm_path').value),
                'resolution': self.current_map.info.resolution,
                'origin': [
                    self.current_map.info.origin.position.x,
                    self.current_map.info.origin.position.y,
                    0.0
                ],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.196
            }
            
            with open(self.get_parameter('temp_yaml_path').value, 'w') as f:
                yaml.dump(yaml_data, f)
                
        except Exception as e:
            self.get_logger().error(f"Error saving temp map: {str(e)}")

    def save_final_map(self):
        try:
            if self.original_map is None or self.current_map is None:
                self.get_logger().error("Cannot save final map: original or current map is None")
                return
                
            width = self.current_map.info.width
            height = self.current_map.info.height
            
            img_data = np.array(self.current_map.data, dtype=np.int8).reshape((height, width))
            
            original_data = np.array(self.original_map.data, dtype=np.int8).reshape((height, width))
            
            img_data = np.where(original_data == 100, 100, img_data)
            
            pgm_data = np.where(img_data == -1, 205, img_data)
            pgm_data = np.where(img_data == 0, 255, pgm_data)
            pgm_data = np.where(img_data == 100, 0, pgm_data)
            
            image = Image.fromarray(pgm_data.astype(np.uint8))
            image.save(self.get_parameter('updated_pgm_path').value)
            
            self.get_logger().info(f"Saved final map to {self.get_parameter('updated_pgm_path').value}")
            
            yaml_data = {
                'image': os.path.basename(self.get_parameter('updated_pgm_path').value),
                'resolution': self.current_map.info.resolution,
                'origin': [
                    self.current_map.info.origin.position.x,
                    self.current_map.info.origin.position.y,
                    0.0
                ],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.196
            }
            
            with open(self.get_parameter('updated_yaml_path').value, 'w') as f:
                yaml.dump(yaml_data, f)
                
        except Exception as e:
            self.get_logger().error(f"Error saving final map: {str(e)}")
    
    def save_debug_map(self):
        try:
            width = self.current_map.info.width
            height = self.current_map.info.height
            
            img_data = np.zeros((height, width, 3), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    idx = y * width + x
                    if idx < len(self.current_map.data):
                        value = self.current_map.data[idx]
                        if value == 100:
                            img_data[y, x] = [0, 0, 0]
                        elif value == 0:
                            img_data[y, x] = [255, 255, 255]
                        else:
                            img_data[y, x] = [128, 128, 128]
            
            for y in range(height):
                for x in range(width):
                    idx = y * width + x
                    if idx < len(self.original_map.data) and self.original_map.data[idx] == 100:
                        img_data[y, x] = [0, 0, 200]
            
            with self.db_lock:
                self.db_cursor.execute('''
                    SELECT cell_x, cell_y, occupied_count, free_count
                    FROM cell_confidence
                    WHERE occupied_count >= 1 OR free_count >= 1
                ''')
                
                for cell_x, cell_y, occ, free in self.db_cursor.fetchall():
                    if not (0 <= cell_y < height and 0 <= cell_x < width):
                        continue
                        
                    total = occ + free
                    if total == 0:
                        continue
                        
                    if occ > free:
                        intensity = min(255, int(255 * (occ / (occ + free))))
                        img_data[cell_y, cell_x] = [intensity, 0, 0]
                    else:
                        intensity = min(255, int(255 * (free / (occ + free))))
                        img_data[cell_y, cell_x] = [0, intensity, 0]
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.get_parameter('debug_map_path').value, f"debug_map_{timestamp}.png")
            image = Image.fromarray(img_data)
            image.save(filename)
            
            self.get_logger().info(f"Saved debug map to {filename}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving debug map: {str(e)}")
    
    def world_to_grid(self, point):
        res = self.current_map.info.resolution
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        grid_x = int((point[0] - origin_x) / res)
        grid_y = int((point[1] - origin_y) / res)
        return grid_x, grid_y

    def grid_to_index(self, grid_x, grid_y):
        width = self.current_map.info.width
        height = self.current_map.info.height
        if 0 <= grid_x < width and 0 <= grid_y < height:
            return grid_y * width + grid_x
        return None

    def near_human(self, point):
        for human in self.human_positions:
            if np.linalg.norm(np.array(point) - np.array(human)) < self.human_radius:
                return True
        return False
    
    def process_all_data_callback(self, request, response):
        try:
            with self.db_lock:
                self.db_cursor.execute('SELECT COUNT(*) FROM scan_data WHERE processed = 0')
                unprocessed_count = self.db_cursor.fetchone()[0]
            
            if unprocessed_count > 0:
                self.get_logger().info(f"Processing all {unprocessed_count} unprocessed scans")
                
                start_time = time.time()
                processed = 0
                
                while processed < unprocessed_count:
                    self.process_batch()
                    
                    with self.db_lock:
                        self.db_cursor.execute('SELECT COUNT(*) FROM scan_data WHERE processed = 0')
                        remaining = self.db_cursor.fetchone()[0]
                    
                    processed = unprocessed_count - remaining
                    
                    if processed % 100 == 0:
                        elapsed = time.time() - start_time
                        self.get_logger().info(f"Processed {processed}/{unprocessed_count} scans ({processed/unprocessed_count*100:.1f}%) in {elapsed:.1f}s")
                    
                    if remaining == 0:
                        break
                
                self.update_map()
                
                elapsed = time.time() - start_time
                response.success = True
                response.message = f"Processed {processed} scans in {elapsed:.1f}s"
                self.get_logger().info(f"Finished processing all data in {elapsed:.1f}s")
            else:
                response.success = True
                response.message = "No unprocessed scans to process"
                self.get_logger().info("No unprocessed scans to process")
                
        except Exception as e:
            response.success = False
            response.message = f"Error processing data: {str(e)}"
            self.get_logger().error(f"Error in process_all_data: {str(e)}")
            
        return response
    
    def reset_cell_confidence_callback(self, request, response):
        try:
            with self.db_lock:
                self.db_cursor.execute('DELETE FROM cell_confidence')
                self.db_conn.commit()
                
            response.success = True
            response.message = "Cell confidence data reset"
            self.get_logger().info("Cell confidence data reset by service call")
            
            self.verification_counter = 0
            
        except Exception as e:
            response.success = False
            response.message = f"Reset failed: {str(e)}"
            self.get_logger().error(f"Error resetting cell confidence: {str(e)}")
            
        return response
    
    def log_stats(self):
        self.get_logger().info(
            f"Stats: {self.stats['scans_processed']} scans processed, "
            f"{self.stats['scans_skipped']} skipped. "
            f"Points: {self.stats['points_processed']}, "
            f"Detections: {self.stats['obstacles_detected']} obstacles, "
            f"{self.stats['free_spaces_detected']} free spaces. "
            f"Avg time: {self.stats['avg_processing_time']*1000:.1f}ms/batch"
        )
        
        self.stats['scans_processed'] = 0
        self.stats['scans_skipped'] = 0
        self.stats['points_processed'] = 0
        self.stats['obstacles_detected'] = 0
        self.stats['free_spaces_detected'] = 0
        
        try:
            with self.db_lock:
                self.db_cursor.execute('SELECT COUNT(*) FROM scan_data WHERE processed = 0')
                unprocessed_count = self.db_cursor.fetchone()[0]
                
                self.db_cursor.execute('SELECT COUNT(*) FROM cell_confidence')
                confidence_count = self.db_cursor.fetchone()[0]
                
            self.get_logger().info(
                f"Database: {unprocessed_count} unprocessed scans, "
                f"{confidence_count} cells with confidence data"
            )
                
        except sqlite3.Error as e:
            self.get_logger().error(f"Error getting database stats: {str(e)}")
            self.stats['db_errors'] += 1
    
    def __del__(self):
        if hasattr(self, 'db_conn'):
            try:
                self.db_conn.close()
            except Exception:
                pass
        if hasattr(self, 'humans_db_conn'):
            try:
                self.humans_db_conn.close()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = MapProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()