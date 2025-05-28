#!/usr/bin/env python3
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseArray, Pose, Point, Quaternion
import os
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from ament_index_python.packages import get_package_share_directory
from scipy.spatial import KDTree
import sqlite3
from rclpy.time import Time

class MultibotHumanDetector(Node):
    def __init__(self):
        super().__init__('multibot_human_detector')

        # Configuration parameters for detection
        self.declare_parameter('use_map_frame', True)
        self.declare_parameter('human_radius', 0.3)
        self.declare_parameter('visualize', True)
        self.declare_parameter('duplicate_threshold', 0.5)
        self.declare_parameter('robot_namespace', '')  
        self.declare_parameter('multi_robot', False)  

        # Configuration parameters for orientation inference
        self.declare_parameter('distance_threshold', 1.0)  
        self.declare_parameter('min_movement', 0.1)     

        # Load parameters
        self.use_map_frame = self.get_parameter('use_map_frame').value
        self.human_radius = self.get_parameter('human_radius').value
        self.visualize = self.get_parameter('visualize').value
        self.duplicate_threshold = self.get_parameter('duplicate_threshold').value
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.min_movement = self.get_parameter('min_movement').value
        self.robot_namespace = self.get_parameter('robot_namespace').value  
        self.multi_robot = self.get_parameter('multi_robot').value  
        
        try:
            self.use_sim_time = self.get_parameter('use_sim_time').value
        except:
            self.use_sim_time = False
            self.get_logger().info("Could not get use_sim_time parameter, defaulting to False")

        self.bridge = CvBridge()
        self.detections = []
        self.current_detections = {'left': None, 'right': None}
        
        self.previous_positions = []
        self.previous_poses = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load YOLO model
        package_name = 'master_project2'
        package_share_directory = get_package_share_directory(package_name)
        weights_path = os.path.join(package_share_directory, 'config', 'yolo', 'best.pt')
        self.model = YOLO(weights_path)

        ns_prefix = f"{self.robot_namespace}/" if self.robot_namespace else ""
        
        self.camera_params = {
            'left': {  
                'position': np.array([0.3369, 0.0679, 0.0585]),
                'rpy': np.array([1.5707, 0.0698, -0.5277]),
                'rgb': None,
                'depth': None,
                'lock': False,
                'fov_h': math.radians(87),
                'fov_v': math.radians(58),
                'min_depth': 0.1,
                'max_depth': 10.0,
                'frame_id': f"{ns_prefix}left_camera_depth_optical_frame",
                'rotated_width': None,
                'rotated_height': None
            },
            'right': {  
                'position': np.array([0.3334, -0.047, 0.0585]),
                'rpy': np.array([1.5707, 0.0698, 0.5277]),
                'rgb': None,
                'depth': None,
                'lock': False,
                'fov_h': math.radians(87),
                'fov_v': math.radians(58),
                'min_depth': 0.1,
                'max_depth': 10.0,
                'frame_id': f"{ns_prefix}right_camera_depth_optical_frame",
                'rotated_width': None,
                'rotated_height': None
            }
        }

        # QoS profile
        qos_profile = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        )

        if self.robot_namespace:
            left_rgb_topic = f'/{self.robot_namespace}/left_camera/color/image_raw'
            right_rgb_topic = f'/{self.robot_namespace}/right_camera/color/image_raw'
            left_depth_topic = f'/{self.robot_namespace}/left_camera/depth/image_raw'
            right_depth_topic = f'/{self.robot_namespace}/right_camera/depth/image_raw'
            detections_topic = f'/{self.robot_namespace}/human_detections'
        else:
            left_rgb_topic = '/left_camera/color/image_raw'
            right_rgb_topic = '/right_camera/color/image_raw'
            left_depth_topic = '/left_camera/depth/image_raw'
            right_depth_topic = '/right_camera/depth/image_raw'
            detections_topic = '/human_detections'

        self.left_sub = self.create_subscription(
            Image, left_rgb_topic,
            lambda msg: self.image_callback(msg, 'left'), qos_profile)
        self.right_sub = self.create_subscription(
            Image, right_rgb_topic,
            lambda msg: self.image_callback(msg, 'right'), qos_profile)
        self.left_depth_sub = self.create_subscription(
            Image, left_depth_topic,
            lambda msg: self.depth_callback(msg, 'left'), qos_profile)
        self.right_depth_sub = self.create_subscription(
            Image, right_depth_topic,
            lambda msg: self.depth_callback(msg, 'right'), qos_profile)

        self.detections_pub = self.create_publisher(PoseArray, detections_topic, 10)
        
        self.get_logger().info(f'Multibot human detector initialized for namespace: "{self.robot_namespace}"')
        self.get_logger().info(f'Subscribed to: {left_rgb_topic}, {right_rgb_topic}, {left_depth_topic}, {right_depth_topic}')
        self.get_logger().info(f'Publishing to: {detections_topic}')

    def image_callback(self, msg, camera):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rotated_rgb = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            self.camera_params[camera]['rgb'] = rotated_rgb
            self.camera_params[camera]['rotated_width'] = rotated_rgb.shape[1]
            self.camera_params[camera]['rotated_height'] = rotated_rgb.shape[0]
            self.process_camera(camera)
        except Exception as e:
            self.get_logger().error(f"{camera} image error: {str(e)}")
            self.camera_params[camera]['rgb'] = None

    def depth_callback(self, msg, camera):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            rotated_depth = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
            self.camera_params[camera]['depth'] = rotated_depth
            self.process_camera(camera)
        except Exception as e:
            self.get_logger().error(f"{camera} depth error: {str(e)}")
            self.camera_params[camera]['depth'] = None

    def process_camera(self, camera):
        if self.camera_params[camera]['lock']:
            return
        if self.camera_params[camera]['rgb'] is None or self.camera_params[camera]['depth'] is None:
            return

        self.camera_params[camera]['lock'] = True
        try:
            results = self.model(self.camera_params[camera]['rgb'], verbose=False)
            camera_detections = []
            for result in results[0].boxes.data:
                confidence = float(result[4])
                class_id = int(result[5])
                if class_id == 0 and confidence > 0.5:  
                    bbox = result[:4].cpu().numpy().astype(int)
                    position = self.get_3d_position(bbox, camera)
                    if position is not None:
                        camera_detections.append(position)
            self.current_detections[camera] = camera_detections
        except Exception as e:
            self.get_logger().error(f"Processing error for {camera}: {str(e)}")
            self.current_detections[camera] = []
        finally:
            self.camera_params[camera]['lock'] = False

        if None not in self.current_detections.values():
            combined = self.current_detections['left'] + self.current_detections['right']
            filtered = self.filter_duplicates(combined)
            self.detections = filtered
            
            self.process_and_publish_detections()
            
            self.current_detections = {'left': None, 'right': None}

    def process_and_publish_detections(self):
        current_positions = [(d.point.x, d.point.y) for d in self.detections]
        
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"  
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        inferred_orientations = {}
        
        if self.previous_positions and current_positions:
            prev_tree = KDTree(self.previous_positions)
            
            for i, (curr_x, curr_y) in enumerate(current_positions):
                orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                if len(self.previous_positions) > 0:
                    distance, idx = prev_tree.query((curr_x, curr_y))
                    
                    if distance <= self.distance_threshold:
                        prev_x, prev_y = self.previous_positions[idx]
                        
                        dx = curr_x - prev_x
                        dy = curr_y - prev_y
                        movement_distance = math.sqrt(dx*dx + dy*dy)
                        
                        if movement_distance >= self.min_movement:
                            yaw = math.atan2(dy, dx)
                            
                            qz = math.sin(yaw / 2.0)
                            qw = math.cos(yaw / 2.0)
                            
                            orientation.z = qz
                            orientation.w = qw
                            
                            self.get_logger().debug(f"Inferred orientation: yaw={yaw:.2f} for human at ({curr_x:.2f}, {curr_y:.2f})")
                        else:
                            if idx < len(self.previous_poses):
                                prev_orientation = self.previous_poses[idx].orientation
                                if prev_orientation.w != 0.0 or prev_orientation.z != 0.0:
                                    orientation = prev_orientation
                                    self.get_logger().debug(f"Using previous orientation for human at ({curr_x:.2f}, {curr_y:.2f})")
                            
                inferred_orientations[i] = orientation
        
        for i, detection in enumerate(self.detections):
            pose = Pose()
            pose.position = detection.point
            
            if i in inferred_orientations:
                pose.orientation = inferred_orientations[i]
            else:
                pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            if self.multi_robot and self.robot_namespace:
                self.get_logger().debug(f"Detection from robot: {self.robot_namespace}")
                
            pose_array.poses.append(pose)
        
        self.detections_pub.publish(pose_array)
        
        self.previous_positions = current_positions
        self.previous_poses = pose_array.poses
        
        oriented_count = sum(1 for i in range(len(pose_array.poses)) if i in inferred_orientations and 
                               (inferred_orientations[i].z != 0.0 or inferred_orientations[i].w != 1.0))
        if len(pose_array.poses) > 0:
            self.get_logger().info(f"Published {len(pose_array.poses)} human detections, {oriented_count} with inferred orientation")

    def get_3d_position(self, bbox, camera):
        try:
            width = self.camera_params[camera]['rotated_width']
            height = self.camera_params[camera]['rotated_height']
            if None in (width, height):
                return None

            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            depth_image = self.camera_params[camera]['depth']
            if depth_image is None:
                return None

            if (center_y >= depth_image.shape[0] or 
                center_x >= depth_image.shape[1] or
                center_y < 0 or center_x < 0):
                return None

            depth_value = depth_image[center_y, center_x]
            try:
                depth_meters = float(depth_value)
            except (TypeError, ValueError) as e:
                self.get_logger().warning(f"Invalid depth value: {depth_value} ({type(depth_value)})")
                return None

            if not (self.camera_params[camera]['min_depth'] <= depth_meters <= self.camera_params[camera]['max_depth']):
                return None

            depth_meters += self.human_radius

            original_width = height  
            original_height = width  

            fx = 674.416  
            fy = 674.416  
            original_cx = 640.5    
            original_cy = 360.5      

            new_cx = original_cy  
            new_cy = original_width - original_cx - 1 

            x_n = (center_x - new_cx) / fx
            y_n = (center_y - new_cy) / fy

            X = x_n * depth_meters
            Y = y_n * depth_meters
            Z = depth_meters

            X_corrected = Y   
            Y_corrected = -X  
            Z_corrected = Z

            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_params[camera]['frame_id']
            point_camera.header.stamp = self.get_clock().now().to_msg()
            point_camera.point = Point(
                x=float(X_corrected),
                y=float(Y_corrected),
                z=float(Z_corrected)
            )

            target_frame = 'map' if self.use_map_frame else 'base_link'
            
            if not self.use_map_frame and self.robot_namespace:
                target_frame = f"{self.robot_namespace}/{target_frame}"
                
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point_camera.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            return do_transform_point(point_camera, transform)

        except tf2_ros.LookupException as e:
            self.get_logger().warning(f"Transform lookup failed: {str(e)}")
            return None
        except tf2_ros.ConnectivityException as e:
            self.get_logger().warning(f"Transform connectivity error: {str(e)}")
            return None
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warning(f"Transform extrapolation error: {str(e)}")
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected error in 3D calculation: {str(e)}")
            return None

    def filter_duplicates(self, detections):
        filtered = []
        for det in detections:
            if not any(self.is_duplicate(det, f) for f in filtered):
                filtered.append(det)
        return filtered

    def is_duplicate(self, det1, det2):
        dx = det1.point.x - det2.point.x
        dy = det1.point.y - det2.point.y
        dz = det1.point.z - det2.point.z
        return math.sqrt(dx**2 + dy**2 + dz**2) < self.duplicate_threshold

def main(args=None):
    rclpy.init(args=args)
    node = MultibotHumanDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()