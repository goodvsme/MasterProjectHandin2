#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import os
import time
import threading
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ScanCollector(Node):
    def __init__(self):
        super().__init__('scan_collector')
        
        package_dir = get_package_share_directory('master_project2')
        
        self.declare_parameters(namespace='',
            parameters=[
                ('database_path', os.path.join(package_dir, 'db/mapping.db')),
                ('max_scans', 1000),
                ('max_poses', 5000),
                ('max_data_age', 300.0),
                ('cleanup_interval', 60.0),
                ('enable_cmd_vel_tracking', True),
                ('compression_enabled', True),
                ('buffer_size', 10),
                ('use_reliable_qos', True),
                ('start_on_motion', True),
                ('motion_linear_threshold', 0.05),
                ('motion_angular_threshold', 0.05)
            ])
        
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.max_scans = self.get_parameter('max_scans').value
        self.max_poses = self.get_parameter('max_poses').value
        self.max_data_age = self.get_parameter('max_data_age').value
        self.cleanup_interval = self.get_parameter('cleanup_interval').value
        self.enable_cmd_vel_tracking = self.get_parameter('enable_cmd_vel_tracking').value
        self.compression_enabled = self.get_parameter('compression_enabled').value
        self.start_on_motion = self.get_parameter('start_on_motion').value
        self.motion_linear_threshold = self.get_parameter('motion_linear_threshold').value
        self.motion_angular_threshold = self.get_parameter('motion_angular_threshold').value
        
        self.db_path = self.get_parameter('database_path').value
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.db_lock = threading.Lock()
        self.init_database()
        
        if self.get_parameter('use_reliable_qos').value:
            self.qos_profile = rclpy.qos.QoSProfile(
                history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
                depth=self.get_parameter('buffer_size').value,
                reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
                durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
            )
        else:
            self.qos_profile = 10
        
        self.is_moving = False
        self.collection_active = not self.start_on_motion
        self.collection_ever_activated = False
        
        self.scan_sub = Subscriber(self, LaserScan, '/scan', qos_profile=self.qos_profile)
        self.pose_sub = Subscriber(self, PoseWithCovarianceStamped, '/amcl_pose', qos_profile=self.qos_profile)

        self.ts = ApproximateTimeSynchronizer(
            [self.scan_sub, self.pose_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            10
        )
        
        if self.enable_cmd_vel_tracking:
            self.create_subscription(
                Twist, 
                '/cmd_vel', 
                self.cmd_vel_callback, 
                10
            )
            self.is_turning = False
            self.cmd_vel_history = []
            self.max_cmd_vel_history = 100
        
        self.create_service(
            Trigger, 
            'reset_database', 
            self.reset_database_callback
        )
        
        self.create_service(
            Trigger, 
            'start_collection', 
            self.start_collection_callback
        )

        self.create_service(
            Trigger, 
            'stop_collection', 
            self.stop_collection_callback
        )
        
        self.create_timer(self.cleanup_interval, self.cleanup_old_data)
        
        self.stats = {
            'scans_collected': 0,
            'poses_collected': 0,
            'last_scan_time': None,
            'last_pose_time': None,
            'db_errors': 0
        }
        self.create_timer(10.0, self.log_stats)
        
        self.get_logger().info('ScanCollector initialized')
        self.get_logger().info(f'Database path: {self.db_path}')
        self.get_logger().info(f'Using simulation time: {self.use_sim_time}')
        self.get_logger().info(f'Start on motion: {self.start_on_motion}')

    def synchronized_callback(self, scan_msg, pose_msg):
        self.scan_callback(scan_msg)
        self.amcl_pose_callback(pose_msg)
        
    def init_database(self):
        try:
            with self.db_lock:
                if hasattr(self, 'db_conn'):
                    try:
                        self.db_conn.close()
                    except Exception:
                        pass
                
                self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.db_cursor = self.db_conn.cursor()
                
                self.db_cursor.execute('PRAGMA journal_mode=WAL')
                
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
                
                self.db_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS status_flags (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        timestamp REAL
                    )
                ''')
                
                self.db_cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_scan_time ON scan_data(stamp_sec, stamp_nanosec)
                ''')
                
                self.db_cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_pose_time ON pose_data(stamp_sec, stamp_nanosec)
                ''')
                
                self.db_cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_scan_processed ON scan_data(processed)
                ''')
                
                self.db_conn.commit()
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Database initialization error: {str(e)}')
            self.stats['db_errors'] += 1
            self.create_timer(1.0, self.init_database, oneshot=True)
    
    def get_current_ros_time(self):
        now = self.get_clock().now().to_msg()
        return (now.sec, now.nanosec)
    
    def scan_callback(self, msg):
        if not self.collection_active:
            return
            
        try:
            stamp_sec = msg.header.stamp.sec
            stamp_nanosec = msg.header.stamp.nanosec
            
            if self.compression_enabled:
                ranges_blob = np.array(msg.ranges, dtype=np.float16).tobytes()
            else:
                ranges_blob = np.array(msg.ranges, dtype=np.float32).tobytes()
            
            with self.db_lock:
                self.db_cursor.execute('''
                    INSERT INTO scan_data (
                        stamp_sec, stamp_nanosec, frame_id,
                        angle_min, angle_max, angle_increment,
                        range_min, range_max, ranges
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stamp_sec, stamp_nanosec, msg.header.frame_id,
                    msg.angle_min, msg.angle_max, msg.angle_increment,
                    msg.range_min, msg.range_max, ranges_blob
                ))
                self.db_conn.commit()
            
            self.stats['scans_collected'] += 1
            self.stats['last_scan_time'] = f"{stamp_sec}.{stamp_nanosec//1000000:03d}"
            
        except sqlite3.Error as e:
            self.get_logger().error(f'Error storing scan data: {str(e)}')
            self.stats['db_errors'] += 1
            self.init_database()
    
    def amcl_pose_callback(self, msg):
        if not self.collection_active:
            return
            
        try:
            stamp_sec = msg.header.stamp.sec
            stamp_nanosec = msg.header.stamp.nanosec
            
            pose = msg.pose.pose
            
            with self.db_lock:
                self.db_cursor.execute('''
                    INSERT INTO pose_data (
                        stamp_sec, stamp_nanosec, frame_id,
                        position_x, position_y, position_z,
                        orientation_x, orientation_y, orientation_z, orientation_w
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stamp_sec, stamp_nanosec, msg.header.frame_id,
                    pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                ))
                self.db_conn.commit()
            
            self.stats['poses_collected'] += 1
            self.stats['last_pose_time'] = f"{stamp_sec}.{stamp_nanosec//1000000:03d}"
            
        except sqlite3.Error as e:
            self.get_logger().error(f'Error storing pose data: {str(e)}')
            self.stats['db_errors'] += 1
            self.init_database()
    
    def odom_callback(self, msg):
        if not self.collection_active:
            return
            
        if self.stats.get('last_pose_time') is None or \
           (time.time() - float(self.stats['last_pose_time'])) > 1.0:
            try:
                stamp_sec = msg.header.stamp.sec
                stamp_nanosec = msg.header.stamp.nanosec
                
                pose = msg.pose.pose
                
                with self.db_lock:
                    self.db_cursor.execute('''
                        INSERT INTO pose_data (
                            stamp_sec, stamp_nanosec, frame_id,
                            position_x, position_y, position_z,
                            orientation_x, orientation_y, orientation_z, orientation_w
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        stamp_sec, stamp_nanosec, msg.header.frame_id,
                        pose.position.x, pose.position.y, pose.position.z,
                        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
                    ))
                    self.db_conn.commit()
                
            except sqlite3.Error as e:
                self.get_logger().error(f'Error storing odom data: {str(e)}')
                self.stats['db_errors'] += 1
    
    def cmd_vel_callback(self, msg):
        if not self.enable_cmd_vel_tracking:
            return
            
        current_time = self.get_clock().now().nanoseconds * 1e-9
        
        self.cmd_vel_history.append((current_time, msg))
        
        if len(self.cmd_vel_history) > self.max_cmd_vel_history:
            self.cmd_vel_history = self.cmd_vel_history[-self.max_cmd_vel_history:]
        
        was_moving = self.is_moving
        linear_moving = (abs(msg.linear.x) > self.motion_linear_threshold or 
                        abs(msg.linear.y) > self.motion_linear_threshold)
        angular_moving = abs(msg.angular.z) > self.motion_angular_threshold
        self.is_moving = linear_moving or angular_moving
        
        was_turning = self.is_turning
        self.is_turning = abs(msg.angular.z) > 0.1
        
        if self.start_on_motion and not self.collection_ever_activated and self.is_moving:
            self.collection_active = True
            self.collection_ever_activated = True
            self.get_logger().info('Robot started moving - activating data collection permanently')
        
        try:
            with self.db_lock:
                if was_turning != self.is_turning:
                    self.db_cursor.execute('''
                        INSERT OR REPLACE INTO status_flags (key, value, timestamp)
                        VALUES (?, ?, ?)
                    ''', (
                        'is_turning', 
                        'true' if self.is_turning else 'false',
                        current_time
                    ))
                    
                    if self.is_turning:
                        self.get_logger().debug('Robot started turning')
                    else:
                        self.get_logger().debug('Robot stopped turning')
                
                if was_moving != self.is_moving:
                    self.db_cursor.execute('''
                        INSERT OR REPLACE INTO status_flags (key, value, timestamp)
                        VALUES (?, ?, ?)
                    ''', (
                        'is_moving', 
                        'true' if self.is_moving else 'false',
                        current_time
                    ))
                    
                    if self.is_moving:
                        self.get_logger().debug('Robot started moving')
                    else:
                        self.get_logger().debug('Robot stopped moving')
                
                self.db_cursor.execute('''
                    INSERT OR REPLACE INTO status_flags (key, value, timestamp)
                    VALUES (?, ?, ?)
                ''', (
                    'collection_active', 
                    'true' if self.collection_active else 'false',
                    current_time
                ))
                
                self.db_conn.commit()
        except sqlite3.Error as e:
            self.get_logger().error(f'Error storing motion status: {str(e)}')
            self.stats['db_errors'] += 1
    
    def cleanup_old_data(self):
        try:
            with self.db_lock:
                current_time = self.get_clock().now().nanoseconds * 1e-9
                cutoff_sec = int(current_time) - int(self.max_data_age)
                
                self.db_cursor.execute('''
                    DELETE FROM scan_data 
                    WHERE stamp_sec < ? AND processed = 1
                ''', (cutoff_sec,))
                
                self.db_cursor.execute('''
                    DELETE FROM pose_data 
                    WHERE stamp_sec < ?
                ''', (cutoff_sec,))
                
                self.db_cursor.execute('''
                    DELETE FROM scan_data
                    WHERE id NOT IN (
                        SELECT id FROM scan_data
                        ORDER BY stamp_sec DESC, stamp_nanosec DESC
                        LIMIT ?
                    ) AND processed = 1
                ''', (self.max_scans,))
                
                self.db_cursor.execute('''
                    DELETE FROM pose_data
                    WHERE id NOT IN (
                        SELECT id FROM pose_data
                        ORDER BY stamp_sec DESC, stamp_nanosec DESC
                        LIMIT ?
                    )
                ''', (self.max_poses,))
                
                self.db_conn.commit()
                
                self.get_logger().debug('Database cleanup completed')
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Error during database cleanup: {str(e)}')
            self.stats['db_errors'] += 1
            self.init_database()
    
    def reset_database_callback(self, request, response):
        try:
            with self.db_lock:
                self.db_cursor.execute('DROP TABLE IF EXISTS scan_data')
                self.db_cursor.execute('DROP TABLE IF EXISTS pose_data')
                self.db_cursor.execute('DROP TABLE IF EXISTS status_flags')
                
                self.init_database()
                
                if self.start_on_motion:
                    self.collection_active = False
                    self.collection_ever_activated = False
                
                response.success = True
                response.message = 'Database reset successful'
                self.get_logger().info('Database reset by service call')
        except Exception as e:
            response.success = False
            response.message = f'Database reset failed: {str(e)}'
            self.get_logger().error(f'Database reset failed: {str(e)}')
            self.stats['db_errors'] += 1
            
        return response
    
    def start_collection_callback(self, request, response):
        self.collection_active = True
        self.collection_ever_activated = True
        response.success = True
        response.message = 'Data collection started'
        self.get_logger().info('Data collection started by service call')
        return response

    def stop_collection_callback(self, request, response):
        self.collection_active = False
        response.success = True
        response.message = 'Data collection stopped'
        self.get_logger().info('Data collection stopped by service call')
        return response
    
    def log_stats(self):
        self.get_logger().info(
            f"Stats: {self.stats['scans_collected']} scans, "
            f"{self.stats['poses_collected']} poses collected. "
            f"Last scan: {self.stats['last_scan_time']}, "
            f"Last pose: {self.stats['last_pose_time']}, "
            f"DB errors: {self.stats['db_errors']}, "
            f"Collection active: {self.collection_active}"
        )
        
        try:
            with self.db_lock:
                self.db_cursor.execute('SELECT COUNT(*) FROM scan_data')
                scan_count = self.db_cursor.fetchone()[0]
                
                self.db_cursor.execute('SELECT COUNT(*) FROM pose_data')
                pose_count = self.db_cursor.fetchone()[0]
                
                self.db_cursor.execute('SELECT COUNT(*) FROM scan_data WHERE processed = 0')
                unprocessed_count = self.db_cursor.fetchone()[0]
                
                self.get_logger().info(
                    f"Database contains {scan_count} scans ({unprocessed_count} unprocessed), "
                    f"{pose_count} poses"
                )
                
        except sqlite3.Error as e:
            self.get_logger().error(f'Error getting database stats: {str(e)}')
            self.stats['db_errors'] += 1
    
    def __del__(self):
        if hasattr(self, 'db_conn'):
            try:
                self.db_conn.close()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = ScanCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()