#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sqlite3
import datetime
import math
import os
import yaml
import time
import numpy as np
from PIL import Image
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Int32
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as ROSTime

class DatabaseHandler(Node):
    def __init__(self):
        super().__init__('database_handler')
        
        package_dir = get_package_share_directory('master_project2')
        db_dir = os.path.join(package_dir, 'db')
        maps_dir = os.path.join(package_dir, 'maps')
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(maps_dir, exist_ok=True)

        self.declare_parameters(namespace='',
            parameters=[
                ('database_path', os.path.join(db_dir, 'humans.db')),
                ('deduplication_window', 1.0),
                ('radius_threshold', 0.1),
                ('publish_interval', 1.0),
                
                ('temp_map_yaml', os.path.join(maps_dir, 'temp_map.yaml')),
                ('updated_map_yaml', os.path.join(maps_dir, 'updated_map.yaml')),
                ('cleaned_map_yaml', os.path.join(maps_dir, 'cleaned_map.yaml')),
                ('map_publish_interval', 3),
                
                ('recent_detection_window', 20.0),
                ('republish_interval', 0.1)
            ])
        
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(f"Using simulation time: {self.use_sim_time}")

        self.db_conn = sqlite3.connect(self.get_parameter('database_path').value)
        self.db_cursor = self.db_conn.cursor()
        self.init_database()
        
        mapping_db_path = os.path.join(db_dir, 'mapping.db')
        self.mapping_db_conn = sqlite3.connect(mapping_db_path)
        self.mapping_db_cursor = self.mapping_db_conn.cursor()

        self.temp_map = None
        self.updated_map = None
        self.cleaned_map = None
        self.load_maps()

        qos_profile = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE
        )
        
        self.temp_map_pub = self.create_publisher(OccupancyGrid, '/temp_map', qos_profile)
        self.updated_map_pub = self.create_publisher(OccupancyGrid, '/updated_map', qos_profile)
        self.cleaned_map_pub = self.create_publisher(OccupancyGrid, '/cleaned_map', qos_profile)
        self.human_count_pub = self.create_publisher(Int32, '/human_count', 10)
        self.human_positions_pub = self.create_publisher(PoseArray, '/human_positions', 10)

        self.human_positions_subscription_count = 0
        
        self.create_subscription(
            PoseArray, '/raw_human_detections',
            self.store_detections, 10)

        self.create_timer(
            self.get_parameter('republish_interval').value,
            self.publish_recent_detections
        )
        self.create_timer(
            self.get_parameter('map_publish_interval').value,
            self.publish_maps
        )

        self.create_timer(300.0, self.get_database_stats)
        
        self.history_srv = self.create_service(
            Trigger, 
            'request_detection_history', 
            self.handle_history_request
        )
        
        self.get_logger().info('Database handler initialized')
        self.get_logger().info(f'Using humans database: {self.get_parameter("database_path").value}')
        self.get_logger().info(f'Reading maps from: {mapping_db_path}')

    def init_database(self):
        try:
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS human_detections (
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
                    orientation_w REAL,
                    unique_id INTEGER DEFAULT NULL,
                    ros_time_sec INTEGER,
                    ros_time_nanosec INTEGER
                )
            ''')
            
            self.db_cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_rostime ON human_detections(ros_time_sec, ros_time_nanosec)
            ''')
            
            self.db_cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_unique_id ON human_detections(unique_id)
            ''')
            
            self.db_conn.commit()
        except Exception as e:
            self.get_logger().error(f"Database initialization error: {str(e)}")

    def get_current_ros_time(self):
        now = self.get_clock().now().to_msg()
        return (now.sec, now.nanosec)

    def is_time_within_window(self, time_sec, time_nanosec, window_seconds):
        current_time = self.get_clock().now()
        current_sec = current_time.seconds_nanoseconds()[0]
        current_nanosec = current_time.seconds_nanoseconds()[1]
        
        time_total = time_sec + (time_nanosec / 1e9)
        current_total = current_sec + (current_nanosec / 1e9)
        
        return (current_total - time_total) <= window_seconds

    def load_maps(self):
        self.temp_map = self.load_map(self.get_parameter('temp_map_yaml').value, is_temp=True)
        self.updated_map = self.load_map(self.get_parameter('updated_map_yaml').value, is_temp=False)
        self.cleaned_map = self.load_map(self.get_parameter('cleaned_map_yaml').value, is_temp=False)

    def load_map(self, yaml_path, is_temp):
        try:
            if not os.path.exists(yaml_path):
                self.get_logger().warn(f"Map file not found: {yaml_path}")
                return None
                
            with open(yaml_path, 'r') as f:
                map_meta = yaml.safe_load(f)

            msg = OccupancyGrid()
            msg.header.frame_id = 'map'
            msg.info.resolution = map_meta.get('resolution', 0.05)
            
            map_width = 1201
            map_height = 2101
            
            if is_temp:
                msg.info.width = map_width
                msg.info.height = map_height
                msg.data = [-1] * (msg.info.width * msg.info.height)
                
                try:
                    self.mapping_db_cursor.execute('''
                        SELECT cell_x, cell_y, occupied_count, free_count 
                        FROM cell_confidence
                    ''')
                    for cell_x, cell_y, occ, free in self.mapping_db_cursor.fetchall():
                        idx = cell_y * msg.info.width + cell_x
                        if idx < len(msg.data):
                            total = occ + free
                            if total == 0:
                                continue
                            if occ > free:
                                msg.data[idx] = min(100, int(100 * (occ / total)))
                            else:
                                msg.data[idx] = max(-1, int(-100 * (free / total)))
                except Exception as e:
                    self.get_logger().warn(f"Error reading cell confidence data: {str(e)}")
            else:
                pgm_path = os.path.join(os.path.dirname(yaml_path), map_meta['image'])
                img = Image.open(pgm_path)
                map_data = np.array(img)
                msg.info.width = map_data.shape[1]
                msg.info.height = map_data.shape[0]
                
                occupancy_data = np.zeros_like(map_data, dtype=np.int8)
                
                occupancy_data[map_data == 0] = 100
                
                occupancy_data[map_data == 255] = 0
                
                gray_mask = (map_data > 0) & (map_data < 255)
                if np.any(gray_mask):
                    occupancy_data[gray_mask] = 100 - (map_data[gray_mask] * 100 // 255)
                
                msg.data = occupancy_data.flatten().tolist()

            origin = map_meta.get('origin', [0.0, 0.0, 0.0])
            msg.info.origin.position.x = origin[0]
            msg.info.origin.position.y = origin[1]
            return msg
            
        except Exception as e:
            self.get_logger().error(f"Error loading map: {str(e)}")
            return None

    def publish_maps(self):
        self.load_maps()
        
        current_time = self.get_clock().now().to_msg()
        
        if self.temp_map:
            self.temp_map.header.stamp = current_time
            self.temp_map_pub.publish(self.temp_map)
            self.get_logger().debug("Published temp map")
            
        if self.updated_map:
            self.updated_map.header.stamp = current_time
            self.updated_map_pub.publish(self.updated_map)
            self.get_logger().debug("Published updated map")

        if self.cleaned_map:
            self.cleaned_map.header.stamp = current_time
            self.cleaned_map_pub.publish(self.cleaned_map)
            self.get_logger().debug("Published cleaned map")

    def store_detections(self, msg):
        max_retries = 3
        retry_delay = 0.1
        
        ros_time_sec, ros_time_nanosec = self.get_current_ros_time()
        
        for attempt in range(max_retries):
            try:
                self.db_cursor.execute('BEGIN IMMEDIATE TRANSACTION')
                
                new_detections = 0
                for pose in msg.poses:
                    deduplication_window = self.get_parameter('deduplication_window').value
                    
                    self.db_cursor.execute('''
                        SELECT id, position_x, position_y, unique_id, ros_time_sec, ros_time_nanosec
                        FROM human_detections 
                        WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                    ''', (ros_time_sec, ros_time_nanosec, deduplication_window))

                    is_duplicate = False
                    unique_id = None
                    for existing in self.db_cursor.fetchall():
                        distance = math.sqrt(
                            (pose.position.x - existing[1])**2 +
                            (pose.position.y - existing[2])**2
                        )
                        if distance < self.get_parameter('radius_threshold').value:
                            is_duplicate = True
                            unique_id = existing[3] or existing[0]
                            break

                    if not is_duplicate:
                        self.db_cursor.execute('''
                            INSERT INTO human_detections (
                                stamp_sec, stamp_nanosec, frame_id,
                                position_x, position_y, position_z,
                                orientation_x, orientation_y, orientation_z, orientation_w,
                                unique_id, ros_time_sec, ros_time_nanosec
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id,
                            pose.position.x, pose.position.y, pose.position.z,
                            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
                            None, ros_time_sec, ros_time_nanosec
                        ))
                        new_detection_id = self.db_cursor.lastrowid
                        unique_id = unique_id or new_detection_id
                        new_detections += 1

                    if unique_id:
                        self.db_cursor.execute('''
                            UPDATE human_detections 
                            SET unique_id = ?
                            WHERE unique_id = ? OR id = ?
                        ''', (unique_id, unique_id, unique_id))

                self.db_conn.commit()
                
                if new_detections > 0:
                    self.get_logger().debug(f"Stored {new_detections} new human detections")
                    self.publish_recent_detections() 
                
                return

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    self.get_logger().warn(f"Database locked, retry attempt {attempt+1}/{max_retries}")
                    if self.db_conn.in_transaction:
                        self.db_conn.rollback()
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.get_logger().error("Max retries reached, detection storage failed")
                
                if self.db_conn.in_transaction:
                    self.db_conn.rollback()
                self.get_logger().error(f"Detection storage failed: {str(e)}")
                return
                
            except Exception as e:
                if self.db_conn.in_transaction:
                    self.db_conn.rollback()
                self.get_logger().error(f"Detection storage failed: {str(e)}")
                return

    def publish_recent_detections(self):
        try:
            recent_window = self.get_parameter('recent_detection_window').value
            current_ros_time = self.get_clock().now()
            current_sec = current_ros_time.seconds_nanoseconds()[0]
            current_nanosec = current_ros_time.seconds_nanoseconds()[1]
            
            self.db_cursor.execute('''
                WITH ranked_detections AS (
                    SELECT 
                        id, position_x, position_y, position_z,
                        orientation_x, orientation_y, orientation_z, orientation_w,
                        unique_id,
                        ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY ros_time_sec DESC, ros_time_nanosec DESC) as rn
                    FROM human_detections
                    WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                    AND unique_id IS NOT NULL
                )
                SELECT 
                    id, position_x, position_y, position_z,
                    orientation_x, orientation_y, orientation_z, orientation_w
                FROM ranked_detections
                WHERE rn = 1
            ''', (current_sec, current_nanosec, recent_window))
            
            detections = list(self.db_cursor.fetchall())
            
            self.db_cursor.execute('''
                SELECT 
                    id, position_x, position_y, position_z,
                    orientation_x, orientation_y, orientation_z, orientation_w
                FROM human_detections
                WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                AND unique_id IS NULL
            ''', (current_sec, current_nanosec, recent_window))
            
            detections.extend(self.db_cursor.fetchall())
            
            if not detections:
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                empty_pose_array.header.frame_id = 'map'
                self.human_positions_pub.publish(empty_pose_array)
                
                count_msg = Int32()
                count_msg.data = 0
                self.human_count_pub.publish(count_msg)
                return

            pose_array = PoseArray()
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.header.frame_id = 'map'

            for detection in detections:
                pose = Pose()
                pose.position.x = detection[1]
                pose.position.y = detection[2]
                pose.position.z = detection[3]
                pose.orientation.x = detection[4]
                pose.orientation.y = detection[5]
                pose.orientation.z = detection[6]
                pose.orientation.w = detection[7]
                pose_array.poses.append(pose)

            self.human_positions_pub.publish(pose_array)
            
            count_msg = Int32()
            count_msg.data = len(detections)
            self.human_count_pub.publish(count_msg)
            
            self.get_logger().debug(f"Published {len(detections)} recent human detections (simulation time window: {recent_window}s)")

        except Exception as e:
            self.get_logger().error(f"Failed to publish recent detections: {str(e)}")

    def handle_history_request(self, request, response):
        try:
            response.success = True
            response.message = "Historical data is stored in the database but not exposed via service yet. Use the human_positions topic for recent detections."
            return response
        except Exception as e:
            self.get_logger().error(f"Error handling history request: {str(e)}")
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

    def get_database_stats(self):
        try:
            self.db_cursor.execute("SELECT COUNT(*) FROM human_detections")
            total_detections = self.db_cursor.fetchone()[0]
            
            recent_window = self.get_parameter('recent_detection_window').value
            current_ros_time = self.get_clock().now()
            current_sec = current_ros_time.seconds_nanoseconds()[0]
            current_nanosec = current_ros_time.seconds_nanoseconds()[1]
            
            self.db_cursor.execute(
                """
                SELECT COUNT(*) FROM human_detections 
                WHERE (? - ros_time_sec + (? - ros_time_nanosec) / 1.0e9) <= ?
                """,
                (current_sec, current_nanosec, recent_window)
            )
            recent_detections = self.db_cursor.fetchone()[0]
            
            try:
                self.mapping_db_cursor.execute("SELECT COUNT(*) FROM cell_confidence")
                confidence_cells = self.mapping_db_cursor.fetchone()[0]
            except Exception:
                confidence_cells = "Unknown (mapping database not available)"
            
            self.db_cursor.execute(
                "SELECT MIN(ros_time_sec), MIN(ros_time_nanosec) FROM human_detections"
            )
            oldest_time = self.db_cursor.fetchone()
            
            if oldest_time[0] is not None:
                oldest_detection_time = f"{oldest_time[0]}.{oldest_time[1]//1000000:03d}s"
                current_time_str = f"{current_sec}.{current_nanosec//1000000:03d}s"
                time_diff = (current_sec - oldest_time[0]) + ((current_nanosec - oldest_time[1])/1e9)
                time_diff_str = f"{time_diff:.1f}s ago"
                oldest_detection = f"{oldest_detection_time} (sim time, {time_diff_str} from current {current_time_str})"
            else:
                oldest_detection = "No detections"
            
            self.get_logger().info(
                f"Database stats: {total_detections} total human detections " +
                f"({recent_detections} recent within {recent_window}s sim time), " +
                f"{confidence_cells} confidence cells in mapping database, " +
                f"oldest detection: {oldest_detection}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to get database stats: {str(e)}")

    def __del__(self):
        if hasattr(self, 'db_conn'):
            try:
                self.db_conn.close()
            except Exception:
                pass
        if hasattr(self, 'mapping_db_conn'):
            try:
                self.mapping_db_conn.close()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = DatabaseHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()