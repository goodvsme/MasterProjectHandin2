import numpy as np
import matplotlib
# Use non-interactive backend to avoid Qt/display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import uniform_filter

# ======= PARAMETERS (MODIFY THESE) =======
# Path to ground truth PGM file
GROUND_TRUTH_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_map/map.pgm"

# Path to updated map PGM file (will be compared against ground truth)
UPDATED_MAP_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_map/map_cleaned_map_5.pgm"

# Directory where outputs will be saved
OUTPUT_DIR = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_map/results"

# Path to original world PGM file (without ground truth obstacles)
# Set to None if not available
ORIGINAL_WORLD_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_map/original_world.pgm"

# Padding around region of interest (in pixels)
PADDING = 100
# =========================================

def load_pgm(file_path):
    """Load a PGM file and return as numpy array"""
    try:
        with Image.open(file_path) as img:
            return np.array(img)
    except Exception as e:
        print(f"Error loading PGM file {file_path}: {e}")
        exit(1)

def create_difference_map(ground_truth, updated_map, original_world=None):
    """Create a visualization of differences between obstacle maps"""
    # Define obstacles precisely: black pixels (value 0) are obstacles
    gt_obstacles = ground_truth == 0
    updated_obstacles = updated_map == 0
    
    # Create the difference map with exact pixel-level precision
    # Different categories in the difference map:
    # 0: Free space in both (white) - True Negatives
    # 1: In ground truth only - missing groundtruth (red) - False Negatives
    # 2: In updated map only - added obstacles (blue) - False Positives
    # 3: In both - preserved obstacles (black) - True Positives
    # 4: Correctly added obstacles - subset of preserved obstacles that weren't in original world (green) - True Positives
    diff_map = np.zeros_like(ground_truth, dtype=np.uint8)
    
    # In ground truth only (red) - False Negatives
    missing_gt_mask = np.logical_and(gt_obstacles, ~updated_obstacles)
    diff_map[missing_gt_mask] = 1
    
    # In updated map only (blue) - False Positives
    added_obstacles_mask = np.logical_and(~gt_obstacles, updated_obstacles)
    diff_map[added_obstacles_mask] = 2
    
    # In both maps (black) - True Positives
    preserved_obstacles_mask = np.logical_and(gt_obstacles, updated_obstacles)
    diff_map[preserved_obstacles_mask] = 3
    
    # Category 4 is only set if original_world is provided
    if original_world is not None:
        # Original free space (value != 0) is where there was no obstacle
        original_free = original_world != 0
        
        # Correctly added obstacles:
        # 1. In ground truth (gt_obstacles)
        # 2. In updated map (updated_obstacles)
        # 3. Not in original world (original_free)
        # This is a strict subset of preserved_obstacles_mask
        correctly_added_mask = np.logical_and(preserved_obstacles_mask, original_free)
        
        # Set the correctly added obstacles to category 4 (green) - True Positives
        diff_map[correctly_added_mask] = 4
    
    # Verify that every pixel is assigned exactly one category
    category_count = np.sum([(diff_map == i).astype(int) for i in range(5)], axis=0)
    assert np.all(category_count == 1), "Error: Some pixels were assigned to multiple or no categories"
    
    return diff_map

def create_difference_map_visualization(diff_map, output_path="difference_map_visualization.png"):
    """
    Create a visualization of the difference map with 3 zoomed regions aligned to the left.
    This visualization is meticulous about preserving exact pixel values in zooms.
    
    Args:
        diff_map: The difference map with categories (0-4)
        output_path: Path to save the visualization
    """
    # Force matplotlib to use non-interactive backend
    plt.switch_backend('Agg')
    
    # Get map dimensions
    h, w = diff_map.shape
    
    # Divide the map into quadrants
    h_mid = h // 2
    w_mid = w // 2
    
    quadrants = [
        (0, 0, h_mid, w_mid),           # Top-left
        (0, w_mid, h_mid, w),           # Top-right
        (h_mid, 0, h, w_mid),           # Bottom-left
        (h_mid, w_mid, h, w)            # Bottom-right
    ]
    
    # Categories to prioritize (in order of importance)
    categories_to_find = [4, 1, 2]  # Green, Red, Blue
    
    # Store coordinates of blocks for each zoom
    top_blocks = []
    top_block_categories = []  # Store which category each block represents
    
    # Find the best region in each quadrant
    for q_idx, (q_top, q_left, q_bottom, q_right) in enumerate(quadrants):
        if len(top_blocks) >= 3:  # We only need 3 zooms
            break
            
        best_block = None
        best_density = 0
        best_category = None
        
        # Check each category in priority order
        for category in categories_to_find:
            # Skip if we already have this category and there are other options
            if category in top_block_categories and len(categories_to_find) > len(top_block_categories):
                continue
                
            # Create mask for this category in this quadrant with exact precision
            # First create a mask of zeros for the entire map
            category_mask = np.zeros_like(diff_map, dtype=np.uint8)
            
            # Then set only the quadrant area where the category matches
            quadrant_area = diff_map[q_top:q_bottom, q_left:q_right]
            category_indices = np.where(quadrant_area == category)
            
            # If this category doesn't exist in the quadrant, skip it
            if len(category_indices[0]) == 0:
                continue
                
            # Set the matching pixels in the mask
            # Convert relative indices to absolute
            abs_rows = category_indices[0] + q_top
            abs_cols = category_indices[1] + q_left
            category_mask[abs_rows, abs_cols] = 1
            
            # Find the densest block for this category in this quadrant
            block_size = 50
            h_blocks = max(1, (q_bottom - q_top) // block_size)
            w_blocks = max(1, (q_right - q_left) // block_size)
            
            # Calculate density map
            density = np.zeros((h_blocks, w_blocks))
            for i in range(h_blocks):
                for j in range(w_blocks):
                    # Calculate block boundaries
                    r_start = q_top + i * block_size
                    r_end = min(q_top + (i+1) * block_size, q_bottom)
                    c_start = q_left + j * block_size
                    c_end = min(q_left + (j+1) * block_size, q_right)
                    
                    # Extract block with exact boundaries
                    block = category_mask[r_start:r_end, c_start:c_end]
                    density[i, j] = np.sum(block)
            
            # Find highest density block
            if np.max(density) > best_density:
                # Get coordinates of highest density block
                max_idx = np.argmax(density.flatten())
                i, j = np.unravel_index(max_idx, density.shape)
                
                # Convert to image coordinates with exact pixel values
                r_start = q_top + i * block_size
                c_start = q_left + j * block_size
                
                best_block = (r_start, c_start)
                best_density = np.max(density)
                best_category = category
        
        # If we found a good block in this quadrant, add it
        if best_block is not None and best_density > 0:
            top_blocks.append(best_block)
            top_block_categories.append(best_category)
    
    # If we didn't get enough blocks from quadrants, find more blocks
    # Truncate implementation for brevity - the rest is unchanged
    
    # If we didn't get enough blocks from quadrants, find more blocks in any area
    # that contains categories we haven't shown yet
    if len(top_blocks) < 3:
        # Try to find blocks for categories we haven't shown yet
        for category in categories_to_find:
            if category in top_block_categories or len(top_blocks) >= 3:
                continue
                
            category_mask = (diff_map == category).astype(np.uint8)
            
            # If this category doesn't exist in the map, skip it
            if np.sum(category_mask) == 0:
                continue
                
            # Find best block for this category
            block_size = 50
            h_blocks = max(1, h // block_size)
            w_blocks = max(1, w // block_size)
            
            density = np.zeros((h_blocks, w_blocks))
            for i in range(h_blocks):
                for j in range(w_blocks):
                    r_start = i * block_size
                    r_end = min((i+1) * block_size, h)
                    c_start = j * block_size
                    c_end = min((j+1) * block_size, w)
                    
                    block = category_mask[r_start:r_end, c_start:c_end]
                    density[i, j] = np.sum(block)
                    
            # Find highest density block that's not too close to existing blocks
            flat_indices = np.argsort(density.flatten())[::-1]
            
            for idx in flat_indices:
                if density.flatten()[idx] > 0:
                    i, j = np.unravel_index(idx, density.shape)
                    new_block = (i*block_size, j*block_size)
                    
                    # Check distance from existing blocks
                    too_close = False
                    for existing_block in top_blocks:
                        dist = np.sqrt((new_block[0] - existing_block[0])**2 + 
                                      (new_block[1] - existing_block[1])**2)
                        if dist < block_size * 3:
                            too_close = True
                            break
                            
                    if not too_close:
                        top_blocks.append(new_block)
                        top_block_categories.append(category)
                        break
    
    # Ensure we have at least one zoom region (fallback to center if needed)
    if not top_blocks:
        print("No significant changes found for zoom. Adding default center zoom.")
        center_row = h // 2
        center_col = w // 2
        top_blocks = [(center_row, center_col)]
        top_block_categories = [0]  # Default to "True Negatives" category
    
    # Force at least 3 zoom regions by looking at any changes if needed
    if len(top_blocks) < 3:
        any_change_mask = (diff_map != 0).astype(np.uint8)
        block_size = 50
        
        # Divide map into quadrants we haven't used yet
        remaining_quadrants = []
        for q_idx, (q_top, q_left, q_bottom, q_right) in enumerate(quadrants):
            # Check if we already have a block in this quadrant
            quadrant_used = False
            for block in top_blocks:
                r, c = block
                if q_top <= r < q_bottom and q_left <= c < q_right:
                    quadrant_used = True
                    break
            
            if not quadrant_used:
                remaining_quadrants.append((q_top, q_left, q_bottom, q_right))
        
        # If we have remaining quadrants, try to find blocks there
        for q_top, q_left, q_bottom, q_right in remaining_quadrants:
            if len(top_blocks) >= 3:
                break
                
            # Only look at changes in this quadrant
            quadrant_mask = np.zeros_like(any_change_mask)
            quadrant_mask[q_top:q_bottom, q_left:q_right] = any_change_mask[q_top:q_bottom, q_left:q_right]
            
            # Find densest block of changes
            h_blocks = max(1, (q_bottom - q_top) // block_size)
            w_blocks = max(1, (q_right - q_left) // block_size)
            
            density = np.zeros((h_blocks, w_blocks))
            for i in range(h_blocks):
                for j in range(w_blocks):
                    r_start = q_top + i * block_size
                    r_end = min(q_top + (i+1) * block_size, q_bottom)
                    c_start = q_left + j * block_size
                    c_end = min(q_left + (j+1) * block_size, q_right)
                    
                    block = quadrant_mask[r_start:r_end, c_start:c_end]
                    density[i, j] = np.sum(block)
            
            # Find highest density block
            if np.max(density) > 0:
                max_idx = np.argmax(density.flatten())
                i, j = np.unravel_index(max_idx, density.shape)
                
                # Convert to image coordinates
                r_start = q_top + i * block_size
                c_start = q_left + j * block_size
                
                # Figure out the main category in this block
                r_end = min(r_start + block_size, q_bottom)
                c_end = min(c_start + block_size, q_right)
                block_data = diff_map[r_start:r_end, c_start:c_end]
                
                # Count occurrences of each category (excluding 0)
                category_counts = {}
                for cat in range(1, 5):  # Categories 1-4
                    category_counts[cat] = np.sum(block_data == cat)
                
                # Find most common category (default to 1 if none found)
                main_category = 1
                max_count = 0
                for cat, count in category_counts.items():
                    if count > max_count:
                        max_count = count
                        main_category = cat
                
                # Add this block
                top_blocks.append((r_start, c_start))
                top_block_categories.append(main_category)
    
    # Ensure we don't have duplicates and we have exactly 3 zooms
    # Keep unique pairs of (block, category)
    unique_blocks = []
    unique_categories = []
    for block, category in zip(top_blocks, top_block_categories):
        if block not in unique_blocks:
            unique_blocks.append(block)
            unique_categories.append(category)
    
    # Update our lists
    top_blocks = unique_blocks
    top_block_categories = unique_categories
    
    # If we still don't have enough, add shifted copies
    while len(top_blocks) < 3:
        if len(top_blocks) > 0:
            # Create a shifted copy of the first block
            block = top_blocks[0]
            category = top_block_categories[0]
            shift = (block[0] + 100, block[1] + 100)
            # Make sure it's in bounds
            shift = (min(h-block_size, shift[0]), min(w-block_size, shift[1]))
            top_blocks.append(shift)
            top_block_categories.append(category)
        else:
            # This should never happen since we have the fallback above
            break
    
    # Truncate to 3 zooms if needed
    top_blocks = top_blocks[:3]
    top_block_categories = top_block_categories[:3]
    
    # Define category names for zoom titles
    category_names = {
        0: "True Negatives",
        1: "False Negatives",
        2: "False Positives", 
        3: "True Positives",
        4: "True Positives"
    }
    
    # Create the figure
    fig = plt.figure(figsize=(14, 10))
    
    # Create main axis for difference map
    ax_main = plt.axes([0.35, 0.1, 0.6, 0.8])  # Shifted to the right to make room for zooms on left
    
    # Set up the color map - use exact colors with no interpolation
    colors = ['white', 'red', 'blue', 'black', 'green']  # True Negatives, False Negatives, False Positives, True Positives, True Positives
    cmap = matplotlib.colors.ListedColormap(colors)
    
    # Display the difference map with exact values - ensure no interpolation
    im = ax_main.imshow(diff_map, cmap=cmap, vmin=0, vmax=4, interpolation='none')
    ax_main.axis('off')
    
    # Add colorbar with exact tick positions
    cbar_ax = fig.add_axes([0.35, 0.05, 0.6, 0.02])  # Position below main map
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[0.4, 1.2, 2.0, 2.8, 3.6])
    cbar.set_ticklabels(['True Negatives', 'False Negatives', 'False Positives', 'True Positives', 'True Positives (Correctly Added)'])
    
    # Add zoom views along the left side
    zoom_size = min(100, 50 * 2)
    
    zoom_width = 0.25  # Width as fraction of figure width
    zoom_height = 0.25  # Height as fraction of figure height
    
    # Reorder the zoom blocks if we have enough
    if len(top_blocks) >= 3:
        # Save the original order
        original_blocks = top_blocks.copy()
        original_categories = top_block_categories.copy()
        
        # Reorder to [block1, block3, block2]
        top_blocks = [original_blocks[0], original_blocks[2], original_blocks[1]]
        top_block_categories = [original_categories[0], original_categories[2], original_categories[1]]
    
    # Calculate vertical positions for 3 zooms - top to bottom
    zoom_positions = [0.7, 0.4, 0.1]  # Top, middle, bottom
    
    # Move zoom views closer to main map
    zoom_x = 0.15  # Left aligned but closer to main map
    
    for i, (row, col) in enumerate(top_blocks):
        if i < 3:  # Limit to top 3 zoom areas
            # Define zoom area - ensure it's within bounds with exact pixel positions
            r_start = max(0, row - zoom_size//4)
            r_end = min(h, row + zoom_size)
            c_start = max(0, col - zoom_size//4)
            c_end = min(w, col + zoom_size)
            
            # Position on the left side - use swapped positions
            zoom_y = zoom_positions[i]  # Use the calculated position
            
            # Create zoom axes
            ax_zoom = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])
            
            # Extract zoomed region - exact pixel extraction
            zoom_img = diff_map[r_start:r_end, c_start:c_end].copy()
            
            # Verify zoom content is exactly what we expect
            unique_values = np.unique(zoom_img)
            if not all(val in [0, 1, 2, 3, 4] for val in unique_values):
                print(f"Warning: Zoom {i+1} contains unexpected values: {unique_values}")
            
            # Draw rectangle on main image to show zoom area
            rect = plt.Rectangle((c_start, r_start), c_end-c_start, r_end-r_start,
                               linewidth=1, edgecolor='yellow', facecolor='none')
            ax_main.add_patch(rect)
            
            # Show zoomed image with NO interpolation to preserve pixel values exactly
            ax_zoom.imshow(zoom_img, cmap=cmap, vmin=0, vmax=4, interpolation='none')
            
            # Turn off axis elements but keep spines visible
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            ax_zoom.set_xticklabels([])
            ax_zoom.set_yticklabels([])
            
            # Add yellow outline to zoom image
            for spine in ax_zoom.spines.values():
                spine.set_edgecolor('yellow')
                spine.set_linewidth(2)
                spine.set_visible(True)
            
            # Set title with white background and category name
            category = top_block_categories[i]
            # Number from top to bottom (1,2,3) regardless of content order
            zoom_number = i + 1
            title = ax_zoom.set_title(f'Zoom {zoom_number}: {category_names[category]}', pad=8, loc='center', y=-0.15)
            title.set_bbox(dict(facecolor='white', edgecolor='none', pad=3.0))
            
            # Get the left middle point of the rectangle in data coordinates
            rect_left = c_start
            rect_middle_y = r_start + (r_end - r_start) / 2
            
            # Calculate the connecting points in figure coordinates
            main_bbox = ax_main.get_position()
            zoom_bbox = ax_zoom.get_position()
            
            # Calculate position of rectangle in figure coordinates
            main_xlim = ax_main.get_xlim()
            main_ylim = ax_main.get_ylim()
            
            # Convert data coordinates to normalized axes coordinates (0-1)
            rect_ax_x = (rect_left - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
            rect_ax_y = (rect_middle_y - main_ylim[0]) / (main_ylim[1] - main_ylim[0])
            
            # Convert axes coordinates to figure coordinates
            rect_fig_x = main_bbox.x0 + rect_ax_x * main_bbox.width
            rect_fig_y = main_bbox.y0 + rect_ax_y * main_bbox.height
            
            # Right edge middle of zoom box in figure coordinates
            zoom_fig_x = zoom_bbox.x0 + zoom_bbox.width
            zoom_fig_y = zoom_bbox.y0 + zoom_bbox.height/2
            
            # Draw connecting line
            fig.add_artist(plt.Line2D([rect_fig_x, zoom_fig_x], 
                                     [rect_fig_y, zoom_fig_y], 
                                     transform=fig.transFigure, color='yellow', linestyle='-', linewidth=1.5))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Difference map visualization with zooms saved to {output_path}")
    
    # Save a diagnostic version showing exact pixel values
    diagnostic_path = output_path.replace('.png', '_diagnostic.png')
    plt.figure(figsize=(18, 10))
    
    # Left side: original difference map
    plt.subplot(1, 2, 1)
    plt.imshow(diff_map, cmap=cmap, vmin=0, vmax=4, interpolation='none')
    plt.title('Original Difference Map (Exact Values)')
    plt.colorbar(ticks=[0.4, 1.2, 2.0, 2.8, 3.6]).set_ticklabels(
        ['True Negatives', 'False Negatives', 'False Positives', 'True Positives', 'True Positives (Correctly Added)'])
    
    # Right side: pixel count validation
    unique, counts = np.unique(diff_map, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    
    labels = []
    values = []
    colors_pie = []
    
    for val in range(5):
        if val in pixel_counts:
            labels.append(['True Negatives', 'False Negatives', 'False Positives', 
                         'True Positives', 'True Positives (Correctly Added)'][val])
            values.append(pixel_counts[val])
            colors_pie.append(colors[val])
    
    plt.subplot(1, 2, 2)
    plt.pie(values, labels=labels, colors=colors_pie, autopct='%1.1f%%')
    plt.title('Pixel Category Distribution')
    
    plt.tight_layout()
    plt.savefig(diagnostic_path, dpi=300)
    plt.close()
    
    print(f"Diagnostic visualization saved to {diagnostic_path}")


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

def find_most_relevant_roi(ground_truth, updated_map, padding=100, focus_on_added=True):
    """
    Find a region of interest that contains the highest density of changes,
    with emphasis on added obstacles if focus_on_added is True.
    """
    # Find differences between maps
    gt_obstacles = ground_truth == 0
    updated_obstacles = updated_map == 0
    
    # Focus on added obstacles if requested (for misalignment scenarios)
    if focus_on_added:
        # Only consider areas where obstacles were added (not in ground truth but in updated map)
        changes = np.logical_and(~gt_obstacles, updated_obstacles)
    else:
        # Consider all changes
        changes = np.logical_xor(gt_obstacles, updated_obstacles)
    
    # If no changes found, return None
    if not np.any(changes):
        return None
    
    # Find density of changes by using a sliding window approach
    window_size = 200  # Adjust as needed
    density_map = np.zeros_like(changes, dtype=np.float32)
    
    # Use simple convolution to find density
    density_map = uniform_filter(changes.astype(np.float32), size=window_size)
    
    # Find coordinates of maximum density
    max_coord = np.unravel_index(np.argmax(density_map), density_map.shape)
    center_row, center_col = max_coord
    
    # Calculate ROI with padding
    roi_size = max(window_size * 2, 500)  # Make ROI large enough to capture context
    min_row = max(0, center_row - roi_size//2)
    max_row = min(ground_truth.shape[0] - 1, center_row + roi_size//2)
    min_col = max(0, center_col - roi_size//2)
    max_col = min(ground_truth.shape[1] - 1, center_col + roi_size//2)
    
    # Return as [top, left, bottom, right]
    return [min_row, min_col, max_row, max_col]

def create_roi_difference_map(ground_truth, updated_map, roi, original_world=None):
    """
    Create a visualization of differences between obstacle maps within the ROI only.
    This function extracts the exact ROI from each map and then creates a difference map.
    No interpolation or extrapolation is performed.
    """
    top, left, bottom, right = roi
    
    # Strictly extract the region of interest from each map with exact pixel boundaries
    gt_roi = ground_truth[top:bottom, left:right].copy()
    updated_roi = updated_map[top:bottom, left:right].copy()
    
    # Handle original world if provided
    original_roi = None
    if original_world is not None:
        original_roi = original_world[top:bottom, left:right].copy()
    
    # Create difference map using only the ROI data, with no extrapolation
    roi_diff_map = create_difference_map(gt_roi, updated_roi, original_roi)
    
    # Validate that the ROI difference map has the exact same dimensions as the input ROIs
    assert roi_diff_map.shape == gt_roi.shape, "Error: ROI difference map dimensions mismatch"
    
    return roi_diff_map

def compute_iou(ground_truth, updated_map):
    """Compute Intersection over Union for obstacles (black pixels)"""
    # For ROS2 maps, obstacles are black (0)
    gt_obstacles = ground_truth == 0
    updated_obstacles = updated_map == 0
    
    # Compute intersection and union of obstacles
    intersection = np.logical_and(gt_obstacles, updated_obstacles)
    union = np.logical_or(gt_obstacles, updated_obstacles)
    
    # Calculate IoU
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    # Avoid division by zero
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    
    return iou

def compute_roi_metrics(ground_truth, updated_map, roi, original_world=None):
    """Compute metrics only for the specified region of interest"""
    top, left, bottom, right = roi
    
    # Extract the region of interest from each map
    gt_roi = ground_truth[top:bottom, left:right]
    updated_roi = updated_map[top:bottom, left:right]
    original_roi = None
    if original_world is not None:
        original_roi = original_world[top:bottom, left:right]
    
    # Calculate IoU for the ROI
    iou = compute_iou(gt_roi, updated_roi)
    
    # Calculate statistics for obstacles in ROI
    gt_obstacle_area = np.sum(gt_roi == 0)
    updated_obstacle_area = np.sum(updated_roi == 0)
    preserved_obstacle_area = np.sum(np.logical_and(gt_roi == 0, updated_roi == 0))
    missing_obstacle_area = np.sum(np.logical_and(gt_roi == 0, updated_roi != 0))
    added_obstacle_area = np.sum(np.logical_and(gt_roi != 0, updated_roi == 0))
    
    # Calculate correctly added obstacles if original world is available
    correctly_added_area = 0
    if original_roi is not None:
        # Correctly added obstacles: preserved obstacles that weren't in original world
        preserved_obstacles = np.logical_and(gt_roi == 0, updated_roi == 0)
        correctly_added_area = np.sum(np.logical_and(preserved_obstacles, original_roi != 0))
    
    # Calculate recall (percentage of ground truth preserved)
    recall = (preserved_obstacle_area / gt_obstacle_area) * 100 if gt_obstacle_area > 0 else 0
    
    metrics = {
        "iou": iou,
        "recall": recall,
        "missing_groundtruth": missing_obstacle_area,
        "added_obstacles": added_obstacle_area,
        "preserved_obstacles": preserved_obstacle_area,
        "correctly_added": correctly_added_area
    }
    
    return metrics

def main():
    """Main function to create difference map visualization"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load maps
    print(f"Loading ground truth map from {GROUND_TRUTH_PATH}")
    ground_truth = load_pgm(GROUND_TRUTH_PATH)
    
    print(f"Loading updated map from {UPDATED_MAP_PATH}")
    updated_map = load_pgm(UPDATED_MAP_PATH)
    
    # Apply transformations (rotate 180 degrees and mirror horizontally) to match orientation
    updated_map = np.rot90(updated_map, k=2)
    updated_map = np.fliplr(updated_map)
    
    # Load original world map if provided
    original_world = None
    if ORIGINAL_WORLD_PATH and os.path.exists(ORIGINAL_WORLD_PATH):
        print(f"Loading original world map from {ORIGINAL_WORLD_PATH}")
        original_world = load_pgm(ORIGINAL_WORLD_PATH)
        # Apply the same transformations
        original_world = np.rot90(original_world, k=2)
        original_world = np.fliplr(original_world)
        # Additionally flip vertically since original map is flipped vertically
        original_world = np.flipud(original_world)
    
    # Check if maps have the same dimensions
    if ground_truth.shape != updated_map.shape:
        print(f"Error: Maps have different dimensions. Ground truth: {ground_truth.shape}, Updated: {updated_map.shape}")
        exit(1)
    
    # Create difference map for the full image (for reference)
    print("Creating full difference map...")
    full_diff_map = create_difference_map(ground_truth, updated_map, original_world)
    
    # Validate difference map contains only expected values
    unique_vals = np.unique(full_diff_map)
    print(f"Checking difference map values: {unique_vals}")
    if not all(val in [0, 1, 2, 3, 4] for val in unique_vals):
        print(f"WARNING: Difference map contains unexpected values: {unique_vals}")
    
    # Create visualization with zoomed regions (full map)
    print("Creating full map visualization with zooms...")
    full_zoom_output_path = os.path.join(OUTPUT_DIR, "gt_vs_updated_diff_map_with_zooms.png")
    create_difference_map_visualization(full_diff_map, full_zoom_output_path)
    
    # Find most relevant region with added obstacles (focus on misalignment areas)
    print("Finding most relevant ROI based on added obstacles...")
    relevant_roi = find_most_relevant_roi(ground_truth, updated_map, PADDING, focus_on_added=True)
    
    # Process relevant ROI if found (focused on misalignment)
    if relevant_roi:
        top, left, bottom, right = relevant_roi
        print(f"Found most relevant ROI: [top={top}, left={left}, bottom={bottom}, right={right}]")
        print(f"ROI dimensions: {bottom-top} x {right-left}")
        
        # Create visualization focusing on the ROI (using the full difference map)
        print("Creating ROI view from full difference map...")
        plt.figure(figsize=(12, 10))
        colors = ['white', 'red', 'blue', 'black', 'green'] 
        cmap = matplotlib.colors.ListedColormap(colors)
        
        # Extract exact ROI slice from the full difference map
        roi_view = full_diff_map[top:bottom, left:right].copy()
        
        # Validate ROI view contains only expected values
        roi_unique_vals = np.unique(roi_view)
        print(f"Checking ROI view values: {roi_unique_vals}")
        
        # Show the ROI from the full difference map with no interpolation
        plt.imshow(roi_view, cmap=cmap, vmin=0, vmax=4, interpolation='none')
        plt.axis('off')
        plt.colorbar(ticks=[0.4, 1.2, 2.0, 2.8, 3.6], 
                    label="Map Differences").set_ticklabels(
                    ['True Negatives', 'False Negatives', 'False Positives', 
                     'True Positives', 'True Positives (Correctly Added)'])
        
        roi_output_path = os.path.join(OUTPUT_DIR, "gt_vs_updated_diff_map_relevant_roi.png")
        plt.savefig(roi_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROI difference map saved to {roi_output_path}")
        
        # Create a difference map for ROI only (considering only data in the ROI)
        print("Creating exact difference map of ROI data...")
        roi_diff_map = create_roi_difference_map(ground_truth, updated_map, relevant_roi, original_world)
        
        # Validate ROI difference map
        roi_diff_unique_vals = np.unique(roi_diff_map)
        print(f"Checking ROI difference map values: {roi_diff_unique_vals}")
        
        # Create visualization with zoomed regions (ROI only)
        print("Creating zoom visualization of ROI-only difference map...")
        roi_zoom_output_path = os.path.join(OUTPUT_DIR, "gt_vs_updated_diff_map_roi_with_zooms.png")
        create_difference_map_visualization(roi_diff_map, roi_zoom_output_path)
        
        # Verify that the two approaches produce consistent results
        if roi_view.shape == roi_diff_map.shape:
            matching_pixels = np.sum(roi_view == roi_diff_map)
            total_pixels = roi_view.size
            match_percentage = (matching_pixels / total_pixels) * 100
            print(f"ROI view consistency check: {match_percentage:.2f}% pixels match between methods")
            
            # If significant differences, save a comparison visualization
            if match_percentage < 99.9:
                plt.figure(figsize=(18, 8))
                plt.subplot(1, 3, 1)
                plt.imshow(roi_view, cmap=cmap, vmin=0, vmax=4, interpolation='none')
                plt.title("ROI View from Full Map")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(roi_diff_map, cmap=cmap, vmin=0, vmax=4, interpolation='none')
                plt.title("Direct ROI Difference Map")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                diff = np.zeros_like(roi_view)
                diff[roi_view != roi_diff_map] = 1
                plt.imshow(diff, cmap='gray', interpolation='none')
                plt.title("Differences Between Methods")
                plt.axis('off')
                
                compare_path = os.path.join(OUTPUT_DIR, "roi_method_comparison.png")
                plt.savefig(compare_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Method comparison saved to {compare_path}")
        
        # Calculate and save metrics for ROI only
        roi_metrics = compute_roi_metrics(ground_truth, updated_map, relevant_roi, original_world)
        
        # Save ROI metrics to file
        metrics_text = "=== ROI FOCUSED METRICS ===\n\n"
        metrics_text += f"IoU within ROI: {roi_metrics['iou']:.4f}\n"
        metrics_text += f"Recall within ROI (% of GT preserved): {roi_metrics['recall']:.2f}%\n"
        metrics_text += f"False Negatives within ROI: {roi_metrics['missing_groundtruth']} pixels\n"
        metrics_text += f"False Positives within ROI: {roi_metrics['added_obstacles']} pixels\n"
        metrics_text += f"True Positives within ROI: {roi_metrics['preserved_obstacles']} pixels\n"
        if original_world is not None:
            metrics_text += f"True Positives (Correctly Added) within ROI: {roi_metrics['correctly_added']} pixels\n"
        metrics_text += f"\nRelevant ROI boundaries [top, left, bottom, right]: {relevant_roi}\n"
        metrics_text += f"Relevant ROI dimensions: {bottom-top} x {right-left}\n"
        
        roi_metrics_path = os.path.join(OUTPUT_DIR, "gt_vs_updated_roi_metrics.txt")
        with open(roi_metrics_path, 'w') as f:
            f.write(metrics_text)
        
        print(f"ROI metrics saved to {roi_metrics_path}")
    else:
        print("No relevant ROI found for misalignment correction.")

if __name__ == "__main__":
    main()