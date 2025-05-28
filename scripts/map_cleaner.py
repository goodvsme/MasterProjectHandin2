#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import os
import yaml
import time
from ament_index_python.packages import get_package_share_directory
from nav_msgs.srv import GetMap
import traceback

class EnhancedMapCleaner(Node):
    def __init__(self):
        super().__init__('map_cleaner')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('noise_removal_threshold', 4),
                ('dilation_kernel_size', 5),
                ('erosion_kernel_size', 5),
                ('line_detection_threshold', 10),
                ('corner_detection_quality', 0.1),
                ('min_line_length', 3),
                ('max_line_gap', 5),
                ('large_gap_connection', True),
                ('max_large_gap', 7),
                ('fill_enclosed_areas', True),
                ('corner_connection_dist', 15),
                
                ('maps_directory', 'maps'),
                ('input_map_yaml', 'maps/updated_map.yaml'),
                ('output_map_yaml', 'maps/cleaned_map.yaml'),
                ('preserve_unknowns', True),
                ('process_interval', 5.0),
                ('save_debug_images', True),
                
                ('remove_human_obstacles', False),
                ('human_min_size', 15),
                ('human_max_size', 30),
                ('human_circularity_min', 0.5),
                ('human_aspect_ratio_min', 1.2),
                ('human_aspect_ratio_max', 3.0),
                ('human_solidity_min', 0.7),
            ]
        )

        self.noise_removal_threshold = self.get_parameter('noise_removal_threshold').value
        self.dilation_kernel_size = self.get_parameter('dilation_kernel_size').value
        self.erosion_kernel_size = self.get_parameter('erosion_kernel_size').value
        self.line_detection_threshold = self.get_parameter('line_detection_threshold').value
        self.corner_detection_quality = self.get_parameter('corner_detection_quality').value
        self.min_line_length = self.get_parameter('min_line_length').value
        self.max_line_gap = self.get_parameter('max_line_gap').value
        self.large_gap_connection = self.get_parameter('large_gap_connection').value
        self.max_large_gap = self.get_parameter('max_large_gap').value
        self.maps_directory = self.get_parameter('maps_directory').value
        self.should_fill_enclosed_areas = self.get_parameter('fill_enclosed_areas').value
        self.corner_connection_dist = self.get_parameter('corner_connection_dist').value
        self.input_map_yaml = self.get_parameter('input_map_yaml').value
        self.output_map_yaml = self.get_parameter('output_map_yaml').value
        self.preserve_unknowns = self.get_parameter('preserve_unknowns').value
        self.process_interval = self.get_parameter('process_interval').value
        self.save_debug_images = self.get_parameter('save_debug_images').value
        
        self.remove_human_obstacles = self.get_parameter('remove_human_obstacles').value
        self.human_min_size = self.get_parameter('human_min_size').value
        self.human_max_size = self.get_parameter('human_max_size').value
        self.human_circularity_min = self.get_parameter('human_circularity_min').value
        self.human_aspect_ratio_min = self.get_parameter('human_aspect_ratio_min').value
        self.human_aspect_ratio_max = self.get_parameter('human_aspect_ratio_max').value
        self.human_solidity_min = self.get_parameter('human_solidity_min').value

        package_dir = get_package_share_directory('master_project2')
        self.maps_dir = os.path.join(package_dir, self.maps_directory)
        os.makedirs(self.maps_dir, exist_ok=True)
        
        if not os.path.isabs(self.input_map_yaml):
            self.input_map_yaml = os.path.join(package_dir, self.input_map_yaml)
        if not os.path.isabs(self.output_map_yaml):
            self.output_map_yaml = os.path.join(package_dir, self.output_map_yaml)

        if self.save_debug_images:
            self.debug_dir = os.path.join(self.maps_dir, 'debug')
            os.makedirs(self.debug_dir, exist_ok=True)

        self.declare_parameter('original_map_yaml', 'maps/map.yaml')
        self.declare_parameter('flip_original_map', True)
        self.original_map_yaml = self.get_parameter('original_map_yaml').value
        self.flip_original_map = self.get_parameter('flip_original_map').value
        
        if not os.path.isabs(self.original_map_yaml):
            self.original_map_yaml = os.path.join(package_dir, self.original_map_yaml)
        
        self.map_client = self.create_client(GetMap, '/map_server/map')

        self.original_map = None
        self.original_map_info = None

        self.timer = self.create_timer(self.process_interval, self.process_map)

        self.get_logger().info('Enhanced Map Cleaner initialized')
        self.get_logger().info(f'Input map: {self.input_map_yaml}')
        self.get_logger().info(f'Output map: {self.output_map_yaml}')
        self.get_logger().info(f'Human obstacle removal: {"Enabled" if self.remove_human_obstacles else "Disabled"}')
        if self.remove_human_obstacles:
            self.get_logger().info(f'Human obstacle size range: {self.human_min_size}-{self.human_max_size} pixels')

    def save_debug_image(self, image, step_number, step_name, sub_step=None):
        if not self.save_debug_images:
            return
            
        if not hasattr(self, 'debug_dir'):
            self.debug_dir = os.path.join(self.maps_dir, 'debug')
            os.makedirs(self.debug_dir, exist_ok=True)
        
        if sub_step:
            filename = f"{step_number:02d}_{sub_step}_{step_name}.png"
        else:
            filename = f"{step_number:02d}_{step_name}.png"
        
        filepath = os.path.join(self.debug_dir, filename)
        
        annotated_image = image.copy()
        
        bar_height = 40
        label_image = np.ones((image.shape[0] + bar_height, image.shape[1]), dtype=np.uint8) * 200
        label_image[bar_height:, :] = annotated_image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Step {step_number}"
        if sub_step:
            text += f".{sub_step}"
        text += f": {step_name}"
        
        cv2.putText(label_image, text, (10, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imwrite(filepath, label_image)
        return filepath

    def create_comparison_image(self, before_image, after_image, step_number, step_name):
        if not self.save_debug_images:
            return
            
        h, w = before_image.shape[:2]
        
        comparison = np.ones((h + 40, w * 2 + 10), dtype=np.uint8) * 200
        
        comparison[40:40+h, 0:w] = before_image
        comparison[40:40+h, w+10:w*2+10] = after_image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, f"Step {step_number}: {step_name}", (10, 30), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison, "Before", (10, h+30), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(comparison, "After", (w+10, h+30), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        filename = f"{step_number:02d}_comparison_{step_name}.png"
        filepath = os.path.join(self.debug_dir, filename)
        cv2.imwrite(filepath, comparison)
        return filepath
    
    def create_processing_summary(self):
        if not self.save_debug_images:
            return
            
        key_steps = [
            (2, "b", "updated_map_input"),
            (4, None, "new_obstacles_binary"),
            (7, "c", "after_connecting_gaps"),
            (8, "c", "after_rectangle_filling"),
            (10, None, "cleaned_binary_final"),
            (11, None, "final_output_map")
        ]
        
        images = []
        labels = []
        for step_num, sub_step, step_name in key_steps:
            if sub_step:
                filename = f"{step_num:02d}_{sub_step}_{step_name}.png"
            else:
                filename = f"{step_num:02d}_{step_name}.png"
                
            filepath = os.path.join(self.debug_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape[0] > img.shape[1] // 10:
                        for i in range(min(100, img.shape[0])):
                            if np.mean(img[i, :]) > 190:
                                img = img[i+1:, :]
                                break
                    
                    images.append(img)
                    
                    if step_num == 2 and sub_step == "b":
                        labels.append("1. Input Map")
                    elif step_num == 4:
                        labels.append("2. New Obstacles")
                    elif step_num == 7 and sub_step == "c":
                        labels.append("3. Line Enhanced")
                    elif step_num == 8 and sub_step == "c":
                        labels.append("4. Shapes Filled")
                    elif step_num == 10:
                        labels.append("5. Noise Removed")
                    elif step_num == 11:
                        labels.append("6. Final Output")
        
        if len(images) < 4:
            return
            
        height = min(img.shape[0] for img in images)
        resized_images = []
        for img in images:
            aspect = img.shape[1] / img.shape[0]
            new_width = int(height * aspect)
            resized = cv2.resize(img, (new_width, height))
            resized_images.append(resized)
        
        num_images = len(resized_images)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        col_widths = []
        for col in range(cols):
            col_images = [resized_images[i] for i in range(col, num_images, cols) if i < num_images]
            if col_images:
                col_widths.append(max(img.shape[1] for img in col_images))
            else:
                col_widths.append(0)
                
        total_width = sum(col_widths) + (cols - 1) * 20
        title_height = 60
        row_height = height + 40
        total_height = title_height + rows * row_height + (rows - 1) * 10
        
        summary = np.ones((total_height, total_width), dtype=np.uint8) * 240
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(summary, "Map Cleaning Process Summary", 
                   (total_width // 2 - 200, 40), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        for i, (img, label) in enumerate(zip(resized_images, labels)):
            row = i // cols
            col = i % cols
            
            x = sum(col_widths[:col]) + col * 20
            y = title_height + row * (row_height + 10)
            
            x_centered = x + (col_widths[col] - img.shape[1]) // 2
            
            summary[y:y+img.shape[0], x_centered:x_centered+img.shape[1]] = img
            
            cv2.putText(summary, label, 
                       (x + col_widths[col] // 2 - 70, y + img.shape[0] + 30), 
                       font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        filepath = os.path.join(self.debug_dir, "00_processing_summary.png")
        cv2.imwrite(filepath, summary)
        
        self.get_logger().info(f"Created processing summary at {filepath}")
        return filepath

    def get_original_map(self):
        try:
            if os.path.exists(self.original_map_yaml):
                self.get_logger().info(f"Loading original map from file: {self.original_map_yaml}")
                
                with open(self.original_map_yaml, 'r') as f:
                    map_meta = yaml.safe_load(f)
                
                pgm_path = os.path.join(os.path.dirname(self.original_map_yaml), map_meta['image'])
                
                if not os.path.exists(pgm_path):
                    self.get_logger().warn(f'Original map PGM not found: {pgm_path}')
                else:
                    original_pgm = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
                    
                    if original_pgm is None:
                        self.get_logger().error(f'Failed to load original PGM file: {pgm_path}')
                    else:
                        if self.flip_original_map:
                            original_pgm = cv2.flip(original_pgm, 0)
                            
                        class DummyMapInfo:
                            def __init__(self, resolution):
                                self.resolution = resolution
                                
                        map_info = DummyMapInfo(map_meta.get('resolution', 0.1))
                        
                        return original_pgm, map_info
            
            self.get_logger().info("Falling back to map server to get original map")
            
            if not self.map_client.wait_for_service(timeout_sec=3.0):
                self.get_logger().error('Map service not available')
                return None, None
                
            request = GetMap.Request()
            future = self.map_client.call_async(request)
            
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is None:
                self.get_logger().error('Failed to get map from server')
                return None, None
                
            response = future.result()
            
            map_data = np.array(response.map.data).reshape(
                response.map.info.height, 
                response.map.info.width
            )
            
            original_pgm = np.ones_like(map_data, dtype=np.uint8) * 205
            original_pgm[map_data == 0] = 255
            original_pgm[map_data == 100] = 0
            
            if self.flip_original_map:
                self.get_logger().info("Flipping original map vertically to match updated map orientation")
                original_pgm = cv2.flip(original_pgm, 0)
            
            self.get_logger().info(f"Retrieved original map from server: {original_pgm.shape[1]}x{original_pgm.shape[0]}")
            
            return original_pgm, response.map.info
            
        except Exception as e:
            self.get_logger().error(f'Error getting original map: {e}')
            traceback.print_exc()
            return None, None
        

    def process_map(self):
        try:
            if self.original_map is None:
                self.original_map, self.original_map_info = self.get_original_map()
                if self.original_map is None:
                    self.get_logger().error("Couldn't get original map, aborting processing")
                    return
                    
                self.save_debug_image(self.original_map, 1, "original_map")

            if not os.path.exists(self.input_map_yaml):
                self.get_logger().warn(f'Input map YAML not found: {self.input_map_yaml}')
                return
                
            with open(self.input_map_yaml, 'r') as f:
                map_meta = yaml.safe_load(f)
                
            pgm_path = os.path.join(os.path.dirname(self.input_map_yaml), map_meta['image'])
            
            if not os.path.exists(pgm_path):
                self.get_logger().warn(f'PGM file not found: {pgm_path}')
                return
                
            try:
                mtime = os.path.getmtime(pgm_path)
                if hasattr(self, 'last_mtime') and mtime <= self.last_mtime:
                    self.get_logger().info(f'Input map unchanged, skipping processing')
                    return
                self.last_mtime = mtime
            except:
                pass
                
            self.get_logger().info(f'Processing updated map: {pgm_path}')
                
            updated_map = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
            
            if updated_map is None:
                self.get_logger().error(f'Failed to load PGM file: {pgm_path}')
                return
            
            if self.original_map.shape != updated_map.shape:
                self.get_logger().error(f"Map size mismatch! Original: {self.original_map.shape}, " +
                                      f"Updated: {updated_map.shape}")
                return
            
            obstacles = np.sum(updated_map == 0)
            free = np.sum(updated_map >= 254)
            unknown = np.sum((updated_map > 0) & (updated_map < 254))
            total = updated_map.size
            
            self.get_logger().info(f'Loaded updated map: {updated_map.shape[1]}x{updated_map.shape[0]}, ' +
                                 f'{obstacles} obstacles ({obstacles/total*100:.1f}%), ' +
                                 f'{free} free ({free/total*100:.1f}%), ' +
                                 f'{unknown} unknown ({unknown/total*100:.1f}%)')
            
            self.save_debug_image(self.original_map, 2, "original_map_for_comparison", "a")
            self.save_debug_image(updated_map, 2, "updated_map_input", "b")
            
            self.create_comparison_image(self.original_map, updated_map, 2, "original_vs_updated")
            
            new_obstacles = (updated_map == 0) & (self.original_map != 0)
            
            new_obstacles_viz = np.zeros_like(updated_map, dtype=np.uint8)
            new_obstacles_viz[new_obstacles] = 255
            self.save_debug_image(new_obstacles_viz, 3, "new_obstacles_only")
            
            new_obstacle_count = np.sum(new_obstacles)
            self.get_logger().info(f'Identified {new_obstacle_count} new obstacle pixels to clean')
            
            if new_obstacle_count == 0:
                self.get_logger().info('No new obstacles to process, skipping')
                return
            
            binary_map = np.zeros_like(updated_map, dtype=np.uint8)
            binary_map[new_obstacles] = 255
            
            self.save_debug_image(binary_map, 4, "new_obstacles_binary")
            
            start_time = time.time()
            cleaned_binary = self.clean_binary_map(binary_map, map_meta['resolution'])
            processing_time = time.time() - start_time
            
            self.save_debug_image(cleaned_binary, 10, "cleaned_binary_final")
            
            self.create_comparison_image(binary_map, cleaned_binary, 10, "before_after_cleaning")
            
            cleaned_obstacles = (cleaned_binary == 255)
            
            cleaned_count = np.sum(cleaned_obstacles)
            self.get_logger().info(f'After cleaning: {cleaned_count} obstacle pixels ' +
                                 f'({(cleaned_count-new_obstacle_count)/new_obstacle_count*100:.1f}% change)')
            
            output_map = self.original_map.copy()
            output_map[cleaned_obstacles] = 0
            
            self.save_debug_image(output_map, 11, "final_output_map")
            
            self.create_comparison_image(self.original_map, output_map, 11, "original_vs_output")
            
            output_filename = os.path.basename(self.output_map_yaml).replace('.yaml', '.pgm')
            output_pgm_path = os.path.join(os.path.dirname(self.output_map_yaml), output_filename)
            
            cv2.imwrite(output_pgm_path, output_map)
            
            cleaned_yaml_data = map_meta.copy()
            cleaned_yaml_data['image'] = output_filename
            
            with open(self.output_map_yaml, 'w') as yaml_file:
                yaml.dump(cleaned_yaml_data, yaml_file, default_flow_style=False)
            
            obstacles_cleaned = np.sum(output_map == 0)
            free_cleaned = np.sum(output_map >= 254)
            unknown_cleaned = np.sum((output_map > 0) & (output_map < 254))
            
            self.get_logger().info(f'Map cleaning complete in {processing_time:.2f} seconds')
            self.get_logger().info(f'Cleaned map stats: {obstacles_cleaned} obstacles ({obstacles_cleaned/total*100:.1f}%), ' +
                                 f'{free_cleaned} free ({free_cleaned/total*100:.1f}%), ' +
                                 f'{unknown_cleaned} unknown ({unknown_cleaned/total*100:.1f}%)')
            self.get_logger().info(f'Saved to {output_pgm_path} and {self.output_map_yaml}')
            
            self.create_processing_summary()
            
        except Exception as e:
            self.get_logger().error(f'Error processing map: {str(e)}')
            traceback.print_exc()

    def clean_binary_map(self, binary_map, resolution):
        original_map = binary_map.copy()
        
        line_enhanced_map = self.enhance_lines(binary_map)
        self.save_debug_image(line_enhanced_map, 5, "lines_enhanced")
        
        self.create_comparison_image(binary_map, line_enhanced_map, 5, "line_enhancement_comparison")
        
        if self.should_fill_enclosed_areas:
            filled_map = self.fill_enclosed_shapes(line_enhanced_map)
            self.save_debug_image(filled_map, 6, "enclosed_shapes_filled")
            
            self.create_comparison_image(line_enhanced_map, filled_map, 6, "shape_filling_comparison")
        else:
            filled_map = line_enhanced_map
        
        cleaned_map = self.remove_noise(filled_map)
        self.save_debug_image(cleaned_map, 7, "noise_removed")
        
        self.create_comparison_image(filled_map, cleaned_map, 7, "noise_removal_comparison")
        
        final_cleanup = self.final_cleanup(cleaned_map, resolution)
        
        return final_cleanup

    def calculate_shape_metrics(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return {
                'area': area,
                'circularity': 0,
                'aspect_ratio': 1,
                'solidity': 0,
                'is_convex': False
            }
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 1
        
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio
            
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        is_convex = cv2.isContourConvex(contour)
        
        return {
            'area': area,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'is_convex': is_convex
        }
        
    def identify_human_obstacles(self, binary_map):
        human_mask = np.zeros_like(binary_map)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8)
        
        human_count = 0
        human_pixels = 0
        
        human_viz = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
        
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            
            if size < self.human_min_size or size > self.human_max_size:
                continue
            
            component_mask = np.zeros_like(binary_map, dtype=np.uint8)
            component_mask[labels == label] = 255
            
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            metrics = self.calculate_shape_metrics(largest_contour)
            
            is_human = (
                metrics['area'] >= self.human_min_size and
                metrics['area'] <= self.human_max_size and
                metrics['circularity'] >= self.human_circularity_min and
                metrics['aspect_ratio'] >= self.human_aspect_ratio_min and
                metrics['aspect_ratio'] <= self.human_aspect_ratio_max and
                metrics['solidity'] >= self.human_solidity_min
            )
            
            if is_human:
                human_mask[labels == label] = 255
                human_count += 1
                human_pixels += size
                
                human_viz[labels == label] = [0, 0, 255]
                
                if self.save_debug_images:
                    debug_file = os.path.join(self.debug_dir, f"06_human_{label}.png")
                    cv2.imwrite(debug_file, component_mask)
                    
                    with open(os.path.join(self.debug_dir, f"06_human_{label}_metrics.txt"), 'w') as f:
                        for key, value in metrics.items():
                            f.write(f"{key}: {value}\n")
        
        if human_count > 0:
            self.get_logger().info(f"Identified {human_count} human-shaped obstacles ({human_pixels} pixels)")
            
            if self.save_debug_images:
                self.save_debug_image(human_mask, 6, "human_obstacles_mask", "a")
                
                human_viz_gray = cv2.cvtColor(human_viz, cv2.COLOR_BGR2GRAY)
                self.save_debug_image(human_viz_gray, 6, "human_obstacles_visualization", "b")
                
        return human_mask
    
    def remove_small_obstacles(self, binary_map, size_threshold=10):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8)
        
        cleaned_map = np.zeros_like(binary_map)
        
        removed_obstacles_viz = np.zeros_like(binary_map)
        
        components_removed = 0
        total_pixels_removed = 0
        human_like_removed = 0
        
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            remove_component = False
            
            component_mask = None
            
            if size <= size_threshold:
                remove_component = True
            elif self.remove_human_obstacles and size <= self.human_max_size * 1.2:
                component_mask = np.zeros_like(binary_map, dtype=np.uint8)
                component_mask[labels == label] = 255
                
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    metrics = self.calculate_shape_metrics(largest_contour)
                    
                    if (metrics['circularity'] >= self.human_circularity_min * 0.8 and
                        metrics['aspect_ratio'] >= self.human_aspect_ratio_min * 0.8 and
                        metrics['aspect_ratio'] <= self.human_aspect_ratio_max * 1.2):
                        
                        remove_component = True
                        human_like_removed += 1
                        
                        removed_obstacles_viz[labels == label] = 255
            
            if remove_component:
                components_removed += 1
                total_pixels_removed += size
                
                if not np.any(removed_obstacles_viz[labels == label]):
                    removed_obstacles_viz[labels == label] = 128
            else:
                cleaned_map[labels == label] = 255
        
        if self.save_debug_images and components_removed > 0:
            self.save_debug_image(removed_obstacles_viz, 5, "removed_obstacles", "a")
        
        self.get_logger().info(f"Removed {components_removed} small components ({total_pixels_removed} pixels) " +
                             f"including {human_like_removed} human-like shapes")
        return cleaned_map

    def remove_noise(self, binary_map):
        if self.save_debug_images:
            self.save_debug_image(binary_map, 9, "before_noise_removal", "a")
            
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8)
        
        cleaned_map = np.zeros_like(binary_map)
        
        removed_noise_viz = np.zeros_like(binary_map)
        
        noise_components_removed = 0
        noise_pixels_removed = 0
        
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            
            if size > self.noise_removal_threshold:
                cleaned_map[labels == label] = 255
            else:
                noise_components_removed += 1
                noise_pixels_removed += size
                
                removed_noise_viz[labels == label] = 255
        
        if self.save_debug_images and noise_components_removed > 0:
            self.save_debug_image(removed_noise_viz, 9, "removed_noise", "b")
            
        self.get_logger().info(f"Removed {noise_components_removed} noise components ({noise_pixels_removed} pixels)")
        return cleaned_map

    def enhance_lines(self, binary_map):
        result_map = binary_map.copy()
        
        line_overlay = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
        line_overlay[:,:,0] = binary_map
        line_overlay[:,:,1] = binary_map
        line_overlay[:,:,2] = binary_map
        
        lines = cv2.HoughLinesP(
            binary_map, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.line_detection_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        line_endpoints = []
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            total_lines = len(lines)
            self.get_logger().info(f'Found {total_lines} lines in obstacles')
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                is_horizontal = dy < dx * 0.09  
                is_vertical = dx < dy * 0.09
                
                if is_horizontal:
                    horizontal_lines.append(line)
                    cv2.line(line_overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
                elif is_vertical:
                    vertical_lines.append(line)
                    cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                else:
                    cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            if self.save_debug_images:
                line_overlay_gray = cv2.cvtColor(line_overlay, cv2.COLOR_BGR2GRAY)
                self.save_debug_image(line_overlay_gray, 7, "detected_lines", "a")
            
            horizontal_count = len(horizontal_lines)
            vertical_count = len(vertical_lines)
            
            self.get_logger().info(f'Filtered to {horizontal_count + vertical_count} strictly axis-aligned lines ' +
                                 f'({horizontal_count} horizontal, {vertical_count} vertical)')
            
            for line in horizontal_lines + vertical_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_map, (x1, y1), (x2, y2), 255, 1)
                line_endpoints.append(((x1, y1), (x2, y2)))
        
        if self.save_debug_images:
            self.save_debug_image(result_map, 7, "after_line_drawing", "b")
        
        if self.large_gap_connection and line_endpoints and len(line_endpoints) < 1000:
            result_with_gaps_connected = self.connect_large_gaps(result_map, line_endpoints)
            
            if self.save_debug_images:
                self.save_debug_image(result_with_gaps_connected, 7, "after_connecting_gaps", "c")
                
            return result_with_gaps_connected
        elif len(line_endpoints) >= 1000:
            self.get_logger().warn(f'Skipping large gap connection due to too many lines ({len(line_endpoints)})')
            
        return result_map
    
    def connect_large_gaps(self, map_image, line_endpoints):
        self.get_logger().info(f'Attempting to connect large gaps between {len(line_endpoints)} lines')
        result_map = map_image.copy()
        
        connections_viz = np.zeros((map_image.shape[0], map_image.shape[1], 3), dtype=np.uint8)
        connections_viz[:,:,0] = map_image
        connections_viz[:,:,1] = map_image
        connections_viz[:,:,2] = map_image
        
        max_connections = 200
        connections_made = 0
        horizontal_connections = 0
        vertical_connections = 0
        
        for i, ((x1, y1), (x2, y2)) in enumerate(line_endpoints):
            if connections_made >= max_connections:
                self.get_logger().info(f'Reached maximum connection limit of {max_connections}')
                break
                
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 1:
                continue
                
            dx /= length
            dy /= length
            
            is_horizontal = abs(dy) < 0.05
            is_vertical = abs(dx) < 0.05
            
            if not (is_horizontal or is_vertical):
                continue
                
            for start_point, direction in [((x1, y1), (-dx, -dy)), ((x2, y2), (dx, dy))]:
                best_match = None
                best_match_dist = self.max_large_gap + 1
                
                max_comparisons = 100
                comparisons = 0
                
                for j, ((ex1, ey1), (ex2, ey2)) in enumerate(line_endpoints):
                    if i == j or comparisons >= max_comparisons:
                        continue
                        
                    comparisons += 1
                    
                    edx = ex2 - ex1
                    edy = ey2 - ey1
                    elength = np.sqrt(edx*edx + edy*edy)
                    
                    if elength < 1:
                        continue
                        
                    other_is_horizontal = abs(edy) < 0.05
                    other_is_vertical = abs(edx) < 0.05
                    
                    if (is_horizontal and not other_is_horizontal) or (is_vertical and not other_is_vertical):
                        continue
                    
                    edx /= elength
                    edy /= elength
                    
                    for end_point in [(ex1, ey1), (ex2, ey2)]:
                        dist = np.sqrt((start_point[0] - end_point[0])**2 + 
                                     (start_point[1] - end_point[1])**2)
                        
                        if dist <= self.max_large_gap and dist < best_match_dist:
                            if is_horizontal:
                                if abs(start_point[1] - end_point[1]) > 2:
                                    continue
                            
                            if is_vertical:
                                if abs(start_point[0] - end_point[0]) > 2:
                                    continue
                            
                            best_match = end_point
                            best_match_dist = dist
                
                if best_match is not None:
                    cv2.line(result_map, 
                             (int(start_point[0]), int(start_point[1])), 
                             (int(best_match[0]), int(best_match[1])), 
                             255, 1)
                    
                    if is_horizontal:
                        cv2.line(connections_viz, 
                                (int(start_point[0]), int(start_point[1])), 
                                (int(best_match[0]), int(best_match[1])), 
                                (255, 0, 0), 1)
                        horizontal_connections += 1
                    else:
                        cv2.line(connections_viz, 
                                (int(start_point[0]), int(start_point[1])), 
                                (int(best_match[0]), int(best_match[1])), 
                                (0, 255, 0), 1)
                        vertical_connections += 1
                    
                    connections_made += 1
                    
                    if connections_made % 50 == 0:
                        self.get_logger().info(f'Made {connections_made} connections so far')
        
        self.get_logger().info(f'Connected {connections_made} large gaps between lines ' +
                             f'({horizontal_connections} horizontal, {vertical_connections} vertical)')
        
        if self.save_debug_images and connections_made > 0:
            connections_viz_gray = cv2.cvtColor(connections_viz, cv2.COLOR_BGR2GRAY)
            self.save_debug_image(connections_viz_gray, 7, "line_connections", "gap")
            
        return result_map

    def fill_enclosed_shapes(self, binary_map):
        if self.save_debug_images:
            self.save_debug_image(binary_map, 8, "before_filling", "a")
        
        rectangle_filled = self.detect_and_fill_rectangles(binary_map)
        
        if self.save_debug_images:
            self.save_debug_image(rectangle_filled, 8, "after_rectangle_filling", "c")
            self.create_comparison_image(binary_map, rectangle_filled, 8, "rectangle_filling_comparison")
        
        result_map = self.fill_enclosed_areas(rectangle_filled)
        
        if self.save_debug_images:
            self.save_debug_image(result_map, 8, "after_enclosed_filling", "d")
            self.create_comparison_image(rectangle_filled, result_map, 8, "enclosed_filling_comparison")
            
        return result_map
    
    def fill_shapes_with_contours(self, binary_map):
        result_map = binary_map.copy()
        
        contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.save_debug_images:
            contour_debug = np.zeros_like(binary_map)
            cv2.drawContours(contour_debug, contours, -1, 255, 1)
            self.save_debug_image(contour_debug, 8, "detected_contours", "a1")
        
        shapes_viz = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
        shapes_viz[:,:,0] = binary_map
        shapes_viz[:,:,1] = binary_map
        shapes_viz[:,:,2] = binary_map
        
        shapes_filled = 0
        for i, contour in enumerate(contours):
            min_area = 15
            
            contour_area = cv2.contourArea(contour)
            if contour_area < min_area:
                continue
                
            epsilon = min(0.05, 0.02 + (100 / (contour_area + 10))) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            num_corners = len(approx)
            
            if (3 <= num_corners <= 6) or (contour_area > 200):
                cv2.drawContours(result_map, [contour], 0, 255, -1)
                
                if 3 == num_corners:
                    color = (255, 0, 0)
                elif 4 == num_corners:
                    color = (0, 255, 0)
                elif 5 <= num_corners <= 6:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)
                
                cv2.drawContours(shapes_viz, [contour], 0, color, -1)
                shapes_filled += 1
                
                if self.save_debug_images and shapes_filled <= 10:
                    shape_debug = np.zeros_like(binary_map)
                    cv2.drawContours(shape_debug, [contour], 0, 255, -1)
                    self.save_debug_image(shape_debug, 8, f"filled_shape_{i}", "shape")
                    
        if self.save_debug_images and shapes_filled > 0:
            shapes_viz_gray = cv2.cvtColor(shapes_viz, cv2.COLOR_BGR2GRAY)
            self.save_debug_image(shapes_viz_gray, 8, "filled_shapes_visualization", "viz")
                    
        self.get_logger().info(f'Filled {shapes_filled} polygon shapes with contour detection')
        return result_map
        
    def detect_and_fill_rectangles(self, binary_map):
        result_map = binary_map.copy()
        
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles_filled = 0
        total_rectangle_pixels = 0
        
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area < 15:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            
            rect_ratio = contour_area / rect_area if rect_area > 0 else 0
            
            if rect_ratio > 0.4:
                mask = np.zeros_like(binary_map)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                
                pixels_added = np.sum(mask) // 255
                total_rectangle_pixels += pixels_added
                
                result_map = cv2.bitwise_or(result_map, mask)
                
                actual_added = np.sum((result_map & ~binary_map) > 0)
                self.get_logger().info(f'Rectangle {i}: {w}x{h}, {pixels_added} pixels, actual new pixels: {actual_added}')
                
                rectangles_filled += 1
                
                if self.save_debug_images and rectangles_filled <= 10:
                    rect_debug = np.zeros_like(binary_map)
                    cv2.rectangle(rect_debug, (x, y), (x+w, y+h), 255, -1)
                    self.save_debug_image(rect_debug, 8, f"filled_rectangle_{i}", "rect")
                    
                    self.save_debug_image(result_map, 8, f"result_with_rectangle_{i}", "verify")
        
        new_pixels = np.sum(result_map) - np.sum(binary_map)
        self.get_logger().info(f'Added {rectangles_filled} rectangles with {total_rectangle_pixels} pixels. ' +
                            f'Net pixel change: {new_pixels}')
        
        return result_map
        
    def fill_enclosed_areas(self, binary_map):
        result_map = binary_map.copy()
        
        temp_map = binary_map.copy()
        kernel = np.ones((2, 2), np.uint8)
        temp_map = cv2.dilate(temp_map, kernel, iterations=1)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(temp_map), connectivity=8)
        
        enclosed_viz = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
        enclosed_viz[:,:,0] = binary_map
        enclosed_viz[:,:,1] = binary_map
        enclosed_viz[:,:,2] = binary_map
        
        enclosed_count = 0
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            
            if size < 15 or size > 1000:
                continue
                
            component = np.zeros_like(binary_map, dtype=np.uint8)
            component[labels == label] = 255
            
            dilated = cv2.dilate(component, kernel, iterations=1)
            perimeter = dilated & ~component
            
            touching_obstacles = np.sum((perimeter & binary_map) > 0)
            perimeter_size = np.sum(perimeter > 0)
            
            if perimeter_size > 0 and touching_obstacles / perimeter_size > 0.8:
                result_map = cv2.bitwise_or(result_map, component)
                
                enclosed_viz[labels == label] = (0, 0, 255)
                
                enclosed_count += 1
                
                if self.save_debug_images and enclosed_count <= 10:
                    self.save_debug_image(component, 8, f"enclosed_area_{label}", "enclosed")
        
        if self.save_debug_images and enclosed_count > 0:
            enclosed_viz_gray = cv2.cvtColor(enclosed_viz, cv2.COLOR_BGR2GRAY)
            self.save_debug_image(enclosed_viz_gray, 8, "enclosed_areas_visualization", "enclosedviz")
            
        self.get_logger().info(f'Filled {enclosed_count} enclosed areas')
        return result_map
    
    def final_cleanup(self, map_image, resolution):
        if self.save_debug_images:
            self.save_debug_image(map_image, 10, "before_final_cleanup", "a")
            
        result = map_image.copy()
        
        kernel_size = max(3, int(0.3 / resolution))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        obstacles_closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        if self.save_debug_images:
            self.save_debug_image(obstacles_closed, 10, "after_morphological_close", "b")
            self.create_comparison_image(map_image, obstacles_closed, 10, "morphological_close_comparison")
        
        filled_result = obstacles_closed.copy()
        
        h, w = filled_result.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        
        flood_viz = np.zeros((h, w, 3), dtype=np.uint8)
        flood_viz[:,:,0] = obstacles_closed
        flood_viz[:,:,1] = obstacles_closed
        flood_viz[:,:,2] = obstacles_closed
        
        for y in range(0, h, max(1, h//20)):
            cv2.floodFill(filled_result, mask, (0, y), 100)
            cv2.floodFill(filled_result, mask, (w-1, y), 100)
            
            cv2.circle(flood_viz, (0, y), 3, (255, 0, 0), -1)
            cv2.circle(flood_viz, (w-1, y), 3, (255, 0, 0), -1)
        
        for x in range(0, w, max(1, w//20)):
            cv2.floodFill(filled_result, mask, (x, 0), 100)
            cv2.floodFill(filled_result, mask, (x, h-1), 100)
            
            cv2.circle(flood_viz, (x, 0), 3, (255, 0, 0), -1)
            cv2.circle(flood_viz, (x, h-1), 3, (255, 0, 0), -1)
        
        if self.save_debug_images:
            flood_viz_gray = cv2.cvtColor(flood_viz, cv2.COLOR_BGR2GRAY)
            self.save_debug_image(flood_viz_gray, 10, "flood_fill_starting_points", "c")
        
        enclosed_mask = (filled_result == 255)
        
        enclosed_viz = np.zeros_like(obstacles_closed)
        enclosed_viz[enclosed_mask] = 255
        
        if self.save_debug_images:
            self.save_debug_image(enclosed_viz, 10, "final_enclosed_areas", "d")
        
        result[enclosed_mask] = 255
        
        if self.save_debug_images:
            self.create_comparison_image(obstacles_closed, result, 10, "final_fill_comparison")
        
        return result

def main(args=None):
    rclpy.init(args=args)
    
    node = EnhancedMapCleaner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()