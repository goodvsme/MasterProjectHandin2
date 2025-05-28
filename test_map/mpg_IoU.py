import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import re

def load_pgm(file_path, apply_transforms=False):
    """
    Load a PGM file and return as numpy array
    
    Parameters:
    - file_path: Path to the PGM file
    - apply_transforms: Whether to apply rotation and flip transformations
    """
    try:
        with Image.open(file_path) as img:
            map_data = np.array(img)
            
            # Apply transformations only if requested
            if apply_transforms:
                map_data = np.rot90(map_data, k=2)
                map_data = np.fliplr(map_data)
                
            return map_data
    except Exception as e:
        print(f"Error loading PGM file {file_path}: {e}")
        exit(1)

def compute_iou(ground_truth, updated_map, mask=None):
    """Compute Intersection over Union for obstacles (black pixels)"""
    # For ROS2 maps, obstacles are black (0)
    gt_obstacles = ground_truth == 0
    updated_obstacles = updated_map == 0
    
    # Apply mask if provided
    if mask is not None:
        gt_obstacles = np.logical_and(gt_obstacles, mask)
        updated_obstacles = np.logical_and(updated_obstacles, mask)
    
    # Compute intersection and union of obstacles
    intersection = np.logical_and(gt_obstacles, updated_obstacles)
    union = np.logical_or(gt_obstacles, updated_obstacles)
    
    # Calculate IoU
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    # Avoid division by zero
    if union_area == 0:
        return 0, intersection_area, union_area
    
    iou = intersection_area / union_area
    
    return iou, intersection_area, union_area

def find_changes_bounding_box(ground_truth, updated_map, padding=50):
    """Find a bounding box that contains all changes"""
    # Find differences between maps
    gt_obstacles = ground_truth == 0
    updated_obstacles = updated_map == 0
    changes = np.logical_xor(gt_obstacles, updated_obstacles)
    
    # If no changes found, return None
    if not np.any(changes):
        return None
    
    # Find the coordinates of changes
    rows, cols = np.where(changes)
    
    # Calculate bounding box with padding
    min_row = max(0, np.min(rows) - padding)
    max_row = min(ground_truth.shape[0] - 1, np.max(rows) + padding)
    min_col = max(0, np.min(cols) - padding)
    max_col = min(ground_truth.shape[1] - 1, np.max(cols) + padding)
    
    # Return as [top, left, bottom, right]
    return [min_row, min_col, max_row, max_col]

def analyze_map(ground_truth, updated_map, map_name):
    """Analyze a single map and return metrics"""
    # Debug counters to understand what's happening
    gt_obstacle_pixels = np.sum(ground_truth == 0)
    map_obstacle_pixels = np.sum(updated_map == 0)
    
    # Compute global statistics
    iou, intersection_area, union_area = compute_iou(ground_truth, updated_map)
    
    # Calculate global statistics for obstacles
    gt_obstacle_area = np.sum(ground_truth == 0)
    preserved_obstacle_area = np.sum(np.logical_and(ground_truth == 0, updated_map == 0))
    
    # Calculate recall (percentage of ground truth preserved)
    recall = (preserved_obstacle_area / gt_obstacle_area) * 100 if gt_obstacle_area > 0 else 0
    missing_percentage = 100 - recall
    
    # Print detailed pixel counts for debugging
    print(f"Ground truth obstacle pixels: {gt_obstacle_pixels}")
    print(f"Map obstacle pixels: {map_obstacle_pixels}")
    print(f"Intersection (obstacles in both): {intersection_area}")
    print(f"Union (obstacles in either): {union_area}")
    
    # Find region of interest containing changes
    roi = find_changes_bounding_box(ground_truth, updated_map, padding=100)
    
    # Initialize ROI metrics
    roi_iou = 0
    roi_recall = 0
    roi_missing_percentage = 0
    
    # If changes were found, compute ROI metrics
    if roi:
        top, left, bottom, right = roi
        # Create a mask for the ROI
        roi_mask = np.zeros_like(ground_truth, dtype=bool)
        roi_mask[top:bottom, left:right] = True
        
        # Compute ROI-specific statistics
        roi_iou, roi_intersection_area, roi_union_area = compute_iou(ground_truth, updated_map, roi_mask)
        
        # Calculate ROI statistics for obstacles
        roi_gt_obstacle_area = np.sum(np.logical_and(ground_truth == 0, roi_mask))
        roi_preserved_obstacle_area = np.sum(np.logical_and(
            np.logical_and(ground_truth == 0, updated_map == 0), 
            roi_mask
        ))
        
        # Calculate ROI recall
        roi_recall = (roi_preserved_obstacle_area / roi_gt_obstacle_area) * 100 if roi_gt_obstacle_area > 0 else 0
        roi_missing_percentage = 100 - roi_recall
        
        # Debug for ROI
        print(f"ROI obstacle pixels in ground truth: {roi_gt_obstacle_area}")
        print(f"ROI preserved obstacle pixels: {roi_preserved_obstacle_area}")
    else:
        # If no ROI found, use global metrics
        roi_iou = iou
        roi_recall = recall
        roi_missing_percentage = missing_percentage
        roi = [0, 0, ground_truth.shape[0]-1, ground_truth.shape[1]-1]  # Use full map as ROI
    
    # Print results for debugging
    print(f"\n--- {map_name} - Analysis Results ---")
    print(f"Global IoU: {iou:.4f} ({iou*100:.2f}%)")
    print(f"Global Recall: {recall:.2f}%")
    if roi:
        print(f"ROI boundaries [top, left, bottom, right]: {roi}")
        print(f"ROI dimensions: {roi[2]-roi[0]} x {roi[3]-roi[1]}")
        print(f"ROI IoU: {roi_iou:.4f} ({roi_iou*100:.2f}%)")
        print(f"ROI Recall: {roi_recall:.2f}%")
    
    return {
        'name': map_name,
        'iou': iou,
        'recall': recall,
        'missing_percentage': missing_percentage,
        'roi': roi,
        'roi_iou': roi_iou,
        'roi_recall': roi_recall,
        'roi_missing_percentage': roi_missing_percentage
    }

def visualize_map_comparison(ground_truth, compared_map, map_name, output_dir, roi=None):
    """
    Create a visualization showing the overlap between ground truth and compared map.
    
    Colors in confusion matrix terminology:
    - Green: True Positives (obstacles correctly identified in both maps)
    - Red: False Negatives (obstacles in ground truth but missed in compared map)
    - Blue: False Positives (incorrectly identified as obstacles in compared map)
    - White/Gray: True Negatives (correctly identified free space in both maps)
    """
    # For ROS2 maps, obstacles are black (0), free space is non-zero
    gt_obstacles = ground_truth == 0
    map_obstacles = compared_map == 0
    
    # Create RGB visualization
    height, width = ground_truth.shape
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set default background to light gray (True Negatives - free space in both)
    visualization.fill(240)  # Light gray for free space
    
    # Green: True Positives (obstacles correctly identified in both maps)
    true_positives = np.logical_and(gt_obstacles, map_obstacles)
    visualization[true_positives] = [0, 255, 0]  # Green
    
    # Red: False Negatives (obstacles in ground truth but missed in compared map)
    false_negatives = np.logical_and(gt_obstacles, ~map_obstacles)
    visualization[false_negatives] = [255, 0, 0]  # Red
    
    # Blue: False Positives (incorrectly identified as obstacles in compared map)
    false_positives = np.logical_and(~gt_obstacles, map_obstacles)
    visualization[false_positives] = [0, 0, 255]  # Blue
    
    # True Negatives: free space in both maps (already set to light gray)
    true_negatives = np.logical_and(~gt_obstacles, ~map_obstacles)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Main full comparison
    plt.imshow(visualization)
    plt.title(f"Map Comparison: Ground Truth vs {map_name}", fontsize=14)
    
    # Add legend with confusion matrix terminology
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=[0/255, 255/255, 0/255], label='True Positives (obstacles in both)'),
        plt.Rectangle((0, 0), 1, 1, color=[255/255, 0/255, 0/255], label='False Negatives (obstacles in GT, missed in map)'),
        plt.Rectangle((0, 0), 1, 1, color=[0/255, 0/255, 255/255], label='False Positives (incorrect obstacles in map)'),
        plt.Rectangle((0, 0), 1, 1, color=[240/255, 240/255, 240/255], label='True Negatives (free space in both)')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # If ROI is provided, mark it with a rectangle
    if roi is not None:
        top, left, bottom, right = roi
        rect = plt.Rectangle((left, top), right-left, bottom-top, 
                             linewidth=2, edgecolor='yellow', facecolor='none')
        plt.gca().add_patch(rect)
    
    # Calculate metrics for this view
    total_gt_obstacles = np.sum(gt_obstacles)
    total_map_obstacles = np.sum(map_obstacles)
    tp_count = np.sum(true_positives)
    fn_count = np.sum(false_negatives)
    fp_count = np.sum(false_positives)
    tn_count = np.sum(true_negatives)
    
    # Calculate precision and recall
    precision = (tp_count / (tp_count + fp_count)) * 100 if (tp_count + fp_count) > 0 else 0
    recall = (tp_count / (tp_count + fn_count)) * 100 if (tp_count + fn_count) > 0 else 0
    
    # Add metrics annotation
    metrics_text = f"Confusion Matrix Metrics:\n"
    metrics_text += f"Total GT obstacles: {total_gt_obstacles}\n"
    metrics_text += f"Total map obstacles: {total_map_obstacles}\n"
    metrics_text += f"True Positives: {tp_count} pixels ({(tp_count/total_gt_obstacles*100):.2f}% of GT)\n"
    metrics_text += f"False Negatives: {fn_count} pixels ({(fn_count/total_gt_obstacles*100):.2f}% of GT)\n"
    metrics_text += f"False Positives: {fp_count} pixels\n"
    metrics_text += f"Precision: {precision:.2f}%, Recall: {recall:.2f}%"
    
    plt.annotate(metrics_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                 fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the visualization
    vis_path = os.path.join(output_dir, f"{map_name}_comparison.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to {vis_path}")
    plt.close()
    
    # If ROI is provided, create a zoomed view of just the ROI
    if roi is not None:
        top, left, bottom, right = roi
        
        # Check if ROI is valid
        if top >= 0 and left >= 0 and bottom < height and right < width:
            # Crop visualization to ROI
            roi_vis = visualization[top:bottom, left:right]
            
            # Create ROI plot
            plt.figure(figsize=(10, 8))
            plt.imshow(roi_vis)
            plt.title(f"ROI Comparison: Ground Truth vs {map_name}", fontsize=14)
            
            # Add legend
            plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            
            # Calculate metrics for ROI
            roi_gt_obstacles = np.sum(gt_obstacles[top:bottom, left:right])
            roi_map_obstacles = np.sum(map_obstacles[top:bottom, left:right])
            roi_tp = np.sum(true_positives[top:bottom, left:right])
            roi_fn = np.sum(false_negatives[top:bottom, left:right])
            roi_fp = np.sum(false_positives[top:bottom, left:right])
            roi_tn = np.sum(true_negatives[top:bottom, left:right])
            
            # Calculate ROI precision and recall
            roi_precision = (roi_tp / (roi_tp + roi_fp)) * 100 if (roi_tp + roi_fp) > 0 else 0
            roi_recall = (roi_tp / (roi_tp + roi_fn)) * 100 if (roi_tp + roi_fn) > 0 else 0
            
            # Add ROI metrics annotation
            roi_metrics_text = f"ROI Confusion Matrix Metrics:\n"
            roi_metrics_text += f"ROI GT obstacles: {roi_gt_obstacles}\n"
            roi_metrics_text += f"ROI map obstacles: {roi_map_obstacles}\n"
            roi_metrics_text += f"True Positives: {roi_tp} pixels ({(roi_tp/roi_gt_obstacles*100):.2f}% of GT)\n"
            roi_metrics_text += f"False Negatives: {roi_fn} pixels ({(roi_fn/roi_gt_obstacles*100):.2f}% of GT)\n"
            roi_metrics_text += f"False Positives: {roi_fp} pixels\n"
            roi_metrics_text += f"Precision: {roi_precision:.2f}%, Recall: {roi_recall:.2f}%"
            
            plt.annotate(roi_metrics_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                         fontsize=10, verticalalignment='bottom')
            
            plt.tight_layout()
            
            # Save the ROI visualization
            roi_vis_path = os.path.join(output_dir, f"{map_name}_roi_comparison.png")
            plt.savefig(roi_vis_path, dpi=300, bbox_inches='tight')
            print(f"ROI comparison visualization saved to {roi_vis_path}")
            plt.close()
    
    return vis_path

def plot_metrics_over_time(metrics_list, output_dir):
    """Create plots showing IoU and Recall changes over time, with Map 0 as the baseline"""
    if not metrics_list:
        print("No metrics available for plotting.")
        return
    
    if len(metrics_list) < 2:
        print("Need at least Map 0 and one other map to create meaningful plots.")
        return
    
    # Find Map 0 metrics
    map_0_metrics = None
    for metric in metrics_list:
        if metric['name'] == 'map_0':
            map_0_metrics = metric
            break
    
    if not map_0_metrics:
        print("Map 0 metrics not found, cannot create relative plots.")
        return
    
    # Extract data for plotting
    map_names = [metric['name'] for metric in metrics_list]
    
    # Store baseline metrics
    map_0_global_iou = map_0_metrics['iou']
    map_0_roi_iou = map_0_metrics['roi_iou']
    map_0_global_recall = map_0_metrics['recall']
    map_0_roi_recall = map_0_metrics['roi_recall']
    
    # Calculate maximum possible improvement from baseline
    max_global_iou_improvement = 1.0 - map_0_global_iou
    max_roi_iou_improvement = 1.0 - map_0_roi_iou
    max_global_recall_improvement = 100.0 - map_0_global_recall
    max_roi_recall_improvement = 100.0 - map_0_roi_recall
    
    # Calculate relative improvement as percentage of potential improvement
    global_iou_improvements = []
    roi_iou_improvements = []
    global_recall_improvements = []
    roi_recall_improvements = []
    
    for metric in metrics_list:
        # For IoU, calculate improvement as percentage of maximum possible improvement
        if max_global_iou_improvement > 0:
            global_iou_relative = ((metric['iou'] - map_0_global_iou) / max_global_iou_improvement) * 100
        else:
            global_iou_relative = 0
        
        if max_roi_iou_improvement > 0:
            roi_iou_relative = ((metric['roi_iou'] - map_0_roi_iou) / max_roi_iou_improvement) * 100
        else:
            roi_iou_relative = 0
        
        # For Recall, calculate improvement as percentage of maximum possible improvement
        if max_global_recall_improvement > 0:
            global_recall_relative = ((metric['recall'] - map_0_global_recall) / max_global_recall_improvement) * 100
        else:
            global_recall_relative = 0
            
        if max_roi_recall_improvement > 0:
            roi_recall_relative = ((metric['roi_recall'] - map_0_roi_recall) / max_roi_recall_improvement) * 100
        else:
            roi_recall_relative = 0
        
        global_iou_improvements.append(global_iou_relative)
        roi_iou_improvements.append(roi_iou_relative)
        global_recall_improvements.append(global_recall_relative)
        roi_recall_improvements.append(roi_recall_relative)
    
    # Clean up map names for display
    display_names = []
    for name in map_names:
        match = re.search(r'map_cleaned_map_(\d+)', name)
        if match:
            display_names.append(f"Map {match.group(1)}")
        elif name == "map_0":
            display_names.append("Map 0")
        else:
            display_names.append(name)
    
    # 1. Create IoU Improvement plot (percentage of potential improvement)
    plt.figure(figsize=(12, 7))
    plt.plot(display_names, global_iou_improvements, 'bo-', linewidth=2, markersize=8, label='Global IoU')
    plt.plot(display_names, roi_iou_improvements, 'ro-', linewidth=2, markersize=8, label='ROI IoU')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)  # Zero line (Map 0 baseline)
    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5)  # 100% line (perfect score)
    
    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Improvement Relative to Map 0 (%)', fontsize=12)
    plt.title('IoU Improvement as Percentage of Potential Improvement', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Set y-axis limits to range from 0 to slightly above 100
    min_value = min(min(global_iou_improvements), min(roi_iou_improvements))
    if min_value < 0:
        min_value = min_value * 1.1  # Add some padding for negative values
    else:
        min_value = 0
    max_value = max(max(global_iou_improvements), max(roi_iou_improvements))
    if max_value > 0:
        max_value = max(max_value * 1.1, 110)  # At least 110 for positive values
    else:
        max_value = 110
    plt.ylim(min_value, max_value)
    
    plt.tight_layout()
    
    # Save IoU plot
    iou_plot_path = os.path.join(output_dir, "iou_improvement.png")
    plt.savefig(iou_plot_path, dpi=300)
    print(f"IoU improvement plot saved to {iou_plot_path}")
    plt.close()
    
    # 2. Create Recall Improvement plot (percentage of potential improvement)
    plt.figure(figsize=(12, 7))
    plt.plot(display_names, global_recall_improvements, 'go-', linewidth=2, markersize=8, label='Global Recall')
    plt.plot(display_names, roi_recall_improvements, 'mo-', linewidth=2, markersize=8, label='ROI Recall')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)  # Zero line (Map 0 baseline)
    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5)  # 100% line (perfect score)
    
    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Improvement Relative to Map 0 (%)', fontsize=12)
    plt.title('Recall Improvement as Percentage of Potential Improvement', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Set y-axis limits similar to IoU plot
    min_value = min(min(global_recall_improvements), min(roi_recall_improvements))
    if min_value < 0:
        min_value = min_value * 1.1  # Add some padding for negative values
    else:
        min_value = 0
    max_value = max(max(global_recall_improvements), max(roi_recall_improvements))
    if max_value > 0:
        max_value = max(max_value * 1.1, 110)  # At least 110 for positive values
    else:
        max_value = 110
    plt.ylim(min_value, max_value)
    
    plt.tight_layout()
    
    # Save Recall plot
    recall_plot_path = os.path.join(output_dir, "recall_improvement.png")
    plt.savefig(recall_plot_path, dpi=300)
    print(f"Recall improvement plot saved to {recall_plot_path}")
    plt.close()
    
    # 3. Create IoU Change plot (simple percentage point change)
    plt.figure(figsize=(12, 7))
    global_iou_changes = [(metric['iou'] - map_0_global_iou) * 100 for metric in metrics_list]
    roi_iou_changes = [(metric['roi_iou'] - map_0_roi_iou) * 100 for metric in metrics_list]

    plt.plot(display_names, global_iou_changes, 'bo-', linewidth=2, markersize=8, label='Global IoU')
    plt.plot(display_names, roi_iou_changes, 'ro-', linewidth=2, markersize=8, label='ROI IoU')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)  # Zero line (Map 0 baseline)

    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Change from Map 0 (percentage points)', fontsize=12)
    plt.title('IoU Change Relative to Map 0 Baseline', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # Set y-axis limits with reasonable padding
    min_val = min(min(global_iou_changes), min(roi_iou_changes))
    max_val = max(max(global_iou_changes), max(roi_iou_changes))
    padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
    plt.ylim(min_val - padding, max_val + padding)

    plt.tight_layout()

    # Save IoU plot
    iou_change_path = os.path.join(output_dir, "iou_change.png")
    plt.savefig(iou_change_path, dpi=300)
    print(f"IoU change plot saved to {iou_change_path}")
    plt.close()

    # 4. Create Recall Change plot (simple percentage point change)
    plt.figure(figsize=(12, 7))
    global_recall_changes = [metric['recall'] - map_0_global_recall for metric in metrics_list]
    roi_recall_changes = [metric['roi_recall'] - map_0_roi_recall for metric in metrics_list]

    plt.plot(display_names, global_recall_changes, 'go-', linewidth=2, markersize=8, label='Global Recall')
    plt.plot(display_names, roi_recall_changes, 'mo-', linewidth=2, markersize=8, label='ROI Recall')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)  # Zero line (Map 0 baseline)

    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Change from Map 0 (percentage points)', fontsize=12)
    plt.title('Recall Change Relative to Map 0 Baseline', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # Set y-axis limits with reasonable padding
    min_val = min(min(global_recall_changes), min(roi_recall_changes))
    max_val = max(max(global_recall_changes), max(roi_recall_changes))
    padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
    plt.ylim(min_val - padding, max_val + padding)

    plt.tight_layout()

    # Save Recall plot
    recall_change_path = os.path.join(output_dir, "recall_change.png")
    plt.savefig(recall_change_path, dpi=300)
    print(f"Recall change plot saved to {recall_change_path}")
    plt.close()
    
    # 5. Create Absolute IoU plot with auto-scaled y-axis
    plt.figure(figsize=(12, 7))
    plt.plot(display_names, [metric['iou'] * 100 for metric in metrics_list], 'bo-', linewidth=2, markersize=8, label='Global IoU')
    plt.plot(display_names, [metric['roi_iou'] * 100 for metric in metrics_list], 'ro-', linewidth=2, markersize=8, label='ROI IoU')
    
    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Absolute IoU (%)', fontsize=12)
    plt.title('Absolute IoU Values Compared to Ground Truth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Determine data range for better visualization
    min_iou = min(min([metric['iou'] * 100 for metric in metrics_list]), 
                 min([metric['roi_iou'] * 100 for metric in metrics_list]))
    max_iou = max(max([metric['iou'] * 100 for metric in metrics_list]), 
                 max([metric['roi_iou'] * 100 for metric in metrics_list]))
    # Add some padding (0.5% below min, 0.5% above max)
    y_min = max(0, min_iou - 0.5)  # Don't go below 0
    y_max = min(100, max_iou + 0.5)  # Don't go above 100
    plt.ylim(y_min, y_max)  # Zoom in on the actual data range
    
    plt.tight_layout()
    
    # Save Absolute IoU plot
    abs_iou_plot_path = os.path.join(output_dir, "absolute_iou.png")
    plt.savefig(abs_iou_plot_path, dpi=300)
    print(f"Absolute IoU plot saved to {abs_iou_plot_path}")
    plt.close()
    
    # 6. Create Absolute Recall plot with auto-scaled y-axis
    plt.figure(figsize=(12, 7))
    plt.plot(display_names, [metric['recall'] for metric in metrics_list], 'go-', linewidth=2, markersize=8, label='Global Recall')
    plt.plot(display_names, [metric['roi_recall'] for metric in metrics_list], 'mo-', linewidth=2, markersize=8, label='ROI Recall')
    
    plt.xlabel('Map Version', fontsize=12)
    plt.ylabel('Absolute Recall (%)', fontsize=12)
    plt.title('Absolute Recall Values Compared to Ground Truth', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Determine data range for better visualization
    min_recall = min(min([metric['recall'] for metric in metrics_list]), 
                    min([metric['roi_recall'] for metric in metrics_list]))
    max_recall = max(max([metric['recall'] for metric in metrics_list]), 
                    max([metric['roi_recall'] for metric in metrics_list]))
    # Add some padding (0.5% below min, 0.5% above max)
    y_min = max(0, min_recall - 0.5)  # Don't go below 0
    y_max = min(100, max_recall + 0.5)  # Don't go above 100
    plt.ylim(y_min, y_max)  # Zoom in on the actual data range
    
    plt.tight_layout()
    
    # Save Absolute Recall plot
    abs_recall_plot_path = os.path.join(output_dir, "absolute_recall.png")
    plt.savefig(abs_recall_plot_path, dpi=300)
    print(f"Absolute Recall plot saved to {abs_recall_plot_path}")
    plt.close()
    
    return iou_plot_path, recall_plot_path

def save_summary(metrics_list, output_dir):
    """Create a summary text file with all metrics"""
    if not metrics_list:
        print("No metrics available for summary.")
        return
    
    summary_text = "=== MAP COMPARISON SUMMARY (COMPARED TO GROUND TRUTH) ===\n\n"
    summary_text += "Map Name, Global IoU, Global IoU (%), Global Recall (%), ROI IoU, ROI IoU (%), ROI Recall (%)\n"
    
    for metric in metrics_list:
        summary_text += f"{metric['name']}, {metric['iou']:.4f}, {metric['iou']*100:.2f}%, {metric['recall']:.2f}%, "
        summary_text += f"{metric['roi_iou']:.4f}, {metric['roi_iou']*100:.2f}%, {metric['roi_recall']:.2f}%\n"
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "map_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary saved to {summary_path}")
    return summary_path

def save_improvement_summary(metrics_list, output_dir):
    """Create a summary text file focusing only on improvements relative to Map 0"""
    if not metrics_list:
        print("No metrics available for improvement summary.")
        return
    
    if len(metrics_list) < 2:
        print("Need at least Map 0 and one other map to create improvement summary.")
        return
    
    # Find Map 0 metrics
    map_0_metrics = None
    for metric in metrics_list:
        if metric['name'] == 'map_0':
            map_0_metrics = metric
            break
    
    if not map_0_metrics:
        print("Map 0 metrics not found, cannot create improvement summary.")
        return
    
    # Store baseline metrics
    map_0_global_iou = map_0_metrics['iou']
    map_0_roi_iou = map_0_metrics['roi_iou']
    map_0_global_recall = map_0_metrics['recall']
    map_0_roi_recall = map_0_metrics['roi_recall']
    
    # Calculate maximum possible improvement from baseline
    max_global_iou_improvement = 1.0 - map_0_global_iou
    max_roi_iou_improvement = 1.0 - map_0_roi_iou
    max_global_recall_improvement = 100.0 - map_0_global_recall
    max_roi_recall_improvement = 100.0 - map_0_roi_recall
    
    # Calculate improvements for each map
    improvements = []
    
    for metric in metrics_list:
        # Calculate relative improvement (percentage of potential improvement)
        if max_global_iou_improvement > 0:
            global_iou_improvement = ((metric['iou'] - map_0_global_iou) / max_global_iou_improvement) * 100
        else:
            global_iou_improvement = 0
            
        if max_roi_iou_improvement > 0:
            roi_iou_improvement = ((metric['roi_iou'] - map_0_roi_iou) / max_roi_iou_improvement) * 100
        else:
            roi_iou_improvement = 0
            
        if max_global_recall_improvement > 0:
            global_recall_improvement = ((metric['recall'] - map_0_global_recall) / max_global_recall_improvement) * 100
        else:
            global_recall_improvement = 0
            
        if max_roi_recall_improvement > 0:
            roi_recall_improvement = ((metric['roi_recall'] - map_0_roi_recall) / max_roi_recall_improvement) * 100
        else:
            roi_recall_improvement = 0
        
        # Store improvements for this map
        improvements.append({
            'name': metric['name'],
            'global_iou_improvement': global_iou_improvement,
            'roi_iou_improvement': roi_iou_improvement,
            'global_recall_improvement': global_recall_improvement,
            'roi_recall_improvement': roi_recall_improvement
        })
    
    # Create summary text
    summary_text = "=== MAP IMPROVEMENT SUMMARY (RELATIVE TO MAP 0) ===\n\n"
    
    # Add baseline values
    summary_text += "Baseline (Map 0) values:\n"
    summary_text += f"Global IoU: {map_0_global_iou:.4f} ({map_0_global_iou*100:.2f}%)\n"
    summary_text += f"ROI IoU: {map_0_roi_iou:.4f} ({map_0_roi_iou*100:.2f}%)\n"
    summary_text += f"Global Recall: {map_0_global_recall:.2f}%\n"
    summary_text += f"ROI Recall: {map_0_roi_recall:.2f}%\n\n"
    
    # Add maximum possible improvement info
    summary_text += "Maximum possible improvement from baseline:\n"
    summary_text += f"Global IoU: {max_global_iou_improvement*100:.2f}%\n"
    summary_text += f"ROI IoU: {max_roi_iou_improvement*100:.2f}%\n"
    summary_text += f"Global Recall: {max_global_recall_improvement:.2f}%\n"
    summary_text += f"ROI Recall: {max_roi_recall_improvement:.2f}%\n\n"
    
    # Add header for relative improvements
    summary_text += "RELATIVE IMPROVEMENTS (Percentage of potential improvement):\n"
    summary_text += "Map Name, Global IoU Improvement (%), ROI IoU Improvement (%), Global Recall Improvement (%), ROI Recall Improvement (%)\n"
    
    # Add relative improvements
    for imp in improvements:
        summary_text += f"{imp['name']}, {imp['global_iou_improvement']:.2f}, {imp['roi_iou_improvement']:.2f}, "
        summary_text += f"{imp['global_recall_improvement']:.2f}, {imp['roi_recall_improvement']:.2f}\n"
    
    # Add explanation
    summary_text += "\nEXPLANATION:\n"
    summary_text += "- Relative Improvements: How much of the potential improvement from baseline to perfect score has been achieved\n"
    summary_text += "- 0% means same performance as baseline (Map 0)\n"
    summary_text += "- 100% means perfect score achieved (IoU=1.0 or Recall=100%)\n"
    summary_text += "- Negative values indicate degradation compared to baseline\n"
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "summary_improvements.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Improvement summary saved to {summary_path}")
    return summary_path

def main():
    # Base directory
    base_dir = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_map"
    
    # Create results directory in the same location as the PGM files
    results_dir = os.path.join(base_dir, "results")
    vis_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load ground truth map as the reference for all comparisons
    # Ground truth map needs transformation
    ground_truth_path = os.path.join(base_dir, "map.pgm")
    if os.path.exists(ground_truth_path):
        print(f"Loading ground truth map from {ground_truth_path}")
        ground_truth_map = load_pgm(ground_truth_path, apply_transforms=True)
        
        # Debug info about the ground truth map
        print(f"Ground truth map shape: {ground_truth_map.shape}")
        print(f"Ground truth map min/max values: {np.min(ground_truth_map)}/{np.max(ground_truth_map)}")
        print(f"Ground truth obstacle count (pixels == 0): {np.sum(ground_truth_map == 0)}")
        
        # Visualize ground truth map
        plt.figure(figsize=(10, 8))
        plt.imshow(ground_truth_map, cmap='gray')
        plt.colorbar(label='Pixel Value')
        plt.title("Ground Truth Map", fontsize=14)
        gt_vis_path = os.path.join(vis_dir, "ground_truth_map.png")
        plt.tight_layout()
        plt.savefig(gt_vis_path, dpi=300)
        plt.close()
    else:
        print("Error: Ground truth map (map.pgm) not found. Cannot continue.")
        exit(1)
    
    # Storage for metrics
    all_metrics = []
    
    # Load original world map (Map 0) and compare to ground truth
    # Map 0 also needs transformation
    original_world_path = os.path.join(base_dir, "original_world.pgm")
    if os.path.exists(original_world_path):
        print(f"Loading original world map (Map 0) from {original_world_path}")
        original_map = load_pgm(original_world_path, apply_transforms=True)
        
        # Debug info about Map 0
        print(f"Map 0 shape: {original_map.shape}")
        print(f"Map 0 min/max values: {np.min(original_map)}/{np.max(original_map)}")
        print(f"Map 0 obstacle count (pixels == 0): {np.sum(original_map == 0)}")
        
        # Visualize Map 0
        plt.figure(figsize=(10, 8))
        plt.imshow(original_map, cmap='gray')
        plt.colorbar(label='Pixel Value')
        plt.title("Map 0 (Original World)", fontsize=14)
        map0_vis_path = os.path.join(vis_dir, "map_0.png")
        plt.tight_layout()
        plt.savefig(map0_vis_path, dpi=300)
        plt.close()
        
        # Check if maps have the same dimensions
        if ground_truth_map.shape != original_map.shape:
            print(f"Warning: Map dimensions differ. Ground Truth: {ground_truth_map.shape}, Map 0: {original_map.shape}")
            # Attempt to resize if dimensions don't match
            try:
                original_map_img = Image.fromarray(original_map)
                original_map_img = original_map_img.resize(
                    (ground_truth_map.shape[1], ground_truth_map.shape[0]), 
                    Image.LANCZOS
                )
                original_map = np.array(original_map_img)
                print(f"Resized Map 0 to {original_map.shape}")
            except Exception as e:
                print(f"Error resizing Map 0: {e}")
                exit(1)
        
        # Visualize comparison between ground truth and Map 0
        visualize_map_comparison(ground_truth_map, original_map, "map_0", vis_dir)
        
        # Calculate metrics comparing Map 0 to ground truth
        metrics = analyze_map(ground_truth_map, original_map, "map_0")
        all_metrics.append(metrics)
    else:
        print("Warning: Original world map (Map 0) not found. Continuing without it.")
    
    # Find all map_cleaned_map_X.pgm files
    pattern = os.path.join(base_dir, "map_cleaned_map_*.pgm")
    map_files = glob.glob(pattern)
    
    # Sort the files by number
    def extract_number(filename):
        match = re.search(r'map_cleaned_map_(\d+)\.pgm', filename)
        return int(match.group(1)) if match else float('inf')
    
    map_files.sort(key=extract_number)
    
    print(f"Found {len(map_files)} map files to analyze")
    
    # Process each map file
    for map_file in map_files:
        # Extract map name for output
        map_name = os.path.basename(map_file).replace('.pgm', '')
        map_number = extract_number(map_file)
        
        print(f"\nProcessing {map_file}...")
        
        # Load current map - do NOT apply transformations to cleaned maps
        current_map = load_pgm(map_file, apply_transforms=False)
        
        # Debug info about current map
        print(f"Map {map_number} shape: {current_map.shape}")
        print(f"Map {map_number} min/max values: {np.min(current_map)}/{np.max(current_map)}")
        print(f"Map {map_number} obstacle count (pixels == 0): {np.sum(current_map == 0)}")
        
        # Visualize the current map
        plt.figure(figsize=(10, 8))
        plt.imshow(current_map, cmap='gray')
        plt.colorbar(label='Pixel Value')
        plt.title(f"Map {map_number}", fontsize=14)
        map_vis_path = os.path.join(vis_dir, f"{map_name}.png")
        plt.tight_layout()
        plt.savefig(map_vis_path, dpi=300)
        plt.close()
        
        # Check if maps have the same dimensions
        if ground_truth_map.shape != current_map.shape:
            print(f"Warning: Map dimensions differ. Ground Truth: {ground_truth_map.shape}, {map_name}: {current_map.shape}")
            # Attempt to resize if dimensions don't match
            try:
                current_map_img = Image.fromarray(current_map)
                current_map_img = current_map_img.resize(
                    (ground_truth_map.shape[1], ground_truth_map.shape[0]), 
                    Image.LANCZOS
                )
                current_map = np.array(current_map_img)
                print(f"Resized {map_name} to {current_map.shape}")
            except Exception as e:
                print(f"Error resizing {map_name}: {e}")
                continue
        
        # Create visualization of comparison
        roi = find_changes_bounding_box(ground_truth_map, current_map, padding=100)
        visualize_map_comparison(ground_truth_map, current_map, map_name, vis_dir, roi)
        
        # Calculate metrics comparing to ground truth
        metrics = analyze_map(ground_truth_map, current_map, map_name)
        all_metrics.append(metrics)
    
    # Create summary files
    save_summary(all_metrics, results_dir)
    
    # Create improvement summary file
    if len(all_metrics) > 1:
        save_improvement_summary(all_metrics, results_dir)
    
    # Create plots showing metrics progression over time
    if len(all_metrics) > 1:
        plot_metrics_over_time(all_metrics, results_dir)

if __name__ == "__main__":
    main()