import numpy as np
import matplotlib
# Use non-interactive backend to avoid Qt/display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pathlib import Path

def read_pgm(filename):
    """Read PGM file and return as numpy array."""
    try:
        with Image.open(filename) as img:
            return np.array(img)
    except Exception as e:
        print(f"Error loading PGM file {filename}: {e}")
        exit(1)

def check_and_flip_if_needed(original_map, comparison_map):
    """
    Check if the original map needs to be flipped vertically to match orientation.
    
    This is a basic check that compares the difference magnitude between the original
    and the comparison map in both normal and flipped orientations.
    
    Args:
        original_map: The original PGM map
        comparison_map: The map to compare against (updated or refined)
        
    Returns:
        The original map, possibly flipped to match orientation
    """
    # Ensure maps are the same size
    if original_map.shape != comparison_map.shape:
        print("Warning: Maps have different dimensions, resizing original")
        try:
            from PIL import Image
            # Convert to PIL image, resize, and convert back to numpy array
            pil_img = Image.fromarray(original_map)
            pil_img = pil_img.resize((comparison_map.shape[1], comparison_map.shape[0]))
            original_map = np.array(pil_img)
        except Exception as e:
            print(f"Error resizing: {e}")
            # Fallback: simple resize with numpy (lower quality)
            original_map = np.resize(original_map, comparison_map.shape)
    
    # Calculate difference with normal orientation
    normal_diff = np.sum(np.abs(original_map - comparison_map))
    
    # Calculate difference with flipped orientation 
    # Use numpy's flip instead of cv2.flip
    flipped_map = np.flip(original_map, axis=0)  # flip vertically
    flipped_diff = np.sum(np.abs(flipped_map - comparison_map))
    
    # If flipped orientation has less difference, it's likely the correct orientation
    if flipped_diff < normal_diff:
        print("Original map appears to be flipped vertically - flipping to match others")
        return flipped_map
    else:
        return original_map

def compare_maps(original_map, updated_map):
    """
    Compare two maps and identify differences.
    Returns a mask where differences are marked as 1, same pixels as 0.
    """
    # Ensure maps are the same size
    if original_map.shape != updated_map.shape:
        raise ValueError("Maps must have the same dimensions")
    
    # Create difference mask (1 where pixels are different, 0 where they're the same)
    diff_mask = (original_map != updated_map).astype(np.uint8)
    
    return diff_mask

def create_before_after_visualization(original_map, updated_map, diff_mask, output_path="map_comparison.png"):
    """
    Create a before-and-after visualization with differences highlighted.
    
    Args:
        original_map: The original PGM map
        updated_map: The updated PGM map
        diff_mask: Binary mask where 1 indicates different pixels
        output_path: Path to save the visualization
    """
    # Force matplotlib to use non-interactive backend
    plt.switch_backend('Agg')
    
    # Calculate regions with significant changes for zoomed views
    # We'll use a simple approach to find areas with the most changes
    block_size = 50  # Size of blocks to analyze
    change_density = np.zeros((diff_mask.shape[0] // block_size, diff_mask.shape[1] // block_size))
    
    for i in range(change_density.shape[0]):
        for j in range(change_density.shape[1]):
            block = diff_mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            change_density[i, j] = np.sum(block)
    
    # Find coordinates of top 2 blocks with most changes
    flat_indices = np.argsort(change_density.flatten())[-2:]  # Get indices of top 2 blocks
    top_blocks = []
    for idx in flat_indices:
        if change_density.flatten()[idx] > 0:  # Only include if there are changes
            i, j = np.unravel_index(idx, change_density.shape)
            top_blocks.append((i*block_size, j*block_size))
    
    # Create the figure with a custom layout
    h, w = updated_map.shape
    n_zoom = len(top_blocks)
    
    if n_zoom > 0:
        # Create a figure with custom layout - tighter spacing
        fig = plt.figure(figsize=(12, 8))
        
        # Create a 2x2 grid with tighter spacing
        gs = plt.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1], wspace=0.05)
        
        # Main original map
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.imshow(original_map, cmap='gray')
        # Move title to bottom
        ax_orig.set_title('Original Map', pad=8, loc='center', y=-0.1)
        ax_orig.axis('off')
        
        # Main updated map with highlighted changes
        ax_upd = fig.add_subplot(gs[:, 1])
    else:
        # Just create a side-by-side comparison if no significant change regions
        fig, (ax_orig, ax_upd) = plt.subplots(1, 2, figsize=(12, 6))
        ax_orig.imshow(original_map, cmap='gray')
        ax_orig.set_title('Original Map', pad=8, loc='center', y=-0.1)
        ax_orig.axis('off')
    
    # Create a colored visualization of the updated map
    # We'll use a 3-channel RGB image for the updated map
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize grayscale to 0-255
    if updated_map.max() > 0:  # Avoid division by zero
        norm_updated = (updated_map / updated_map.max() * 255).astype(np.uint8)
    else:
        norm_updated = updated_map.astype(np.uint8)
    
    # Copy grayscale values to all channels for base image
    rgb_img[:, :, 0] = norm_updated  # Red channel
    rgb_img[:, :, 1] = norm_updated  # Green channel
    rgb_img[:, :, 2] = norm_updated  # Blue channel
    
    # Color the changed pixels in bright blue (no dilation)
    # Using a more vibrant blue (0, 127, 255) instead of pure blue
    rgb_img[diff_mask == 1, 0] = 0      # Red channel
    rgb_img[diff_mask == 1, 1] = 127    # Green channel - adding some green for better visibility
    rgb_img[diff_mask == 1, 2] = 255    # Blue channel (max)
    
    # Plot updated map with blue highlights
    ax_upd.imshow(rgb_img)
    # Simplified title at the bottom
    ax_upd.set_title('Updated Map', pad=8, loc='center', y=-0.1)
    ax_upd.axis('off')
    
    # Add a dividing line between the original and updated maps
    # that doesn't extend all the way to the top and bottom
    orig_bbox = ax_orig.get_position()
    upd_bbox = ax_upd.get_position()
    
    line_x = (orig_bbox.x1 + upd_bbox.x0) / 2
    line_y_bottom = 0.1  # Start line 10% from bottom
    line_y_top = 0.9     # End line 10% from top
    
    # Add the dividing line
    fig.add_artist(plt.Line2D([line_x, line_x], 
                             [line_y_bottom, line_y_top], 
                             transform=fig.transFigure, color='black', linestyle='-', linewidth=1))
    
    # Get the position of the updated map for overlapping zoom placement
    upd_bbox = ax_upd.get_position()
    upd_width = upd_bbox.width
    upd_height = upd_bbox.height
    upd_x0 = upd_bbox.x0
    upd_y0 = upd_bbox.y0
    
    # Add zoom views stacked vertically, slightly overlapping with the updated map
    zoom_axes = []
    zoom_size = min(block_size * 2, 100)  # Size of zoom window
    
    # Swap zoom 1 and 2 positions as requested
    if len(top_blocks) >= 2:
        top_blocks = [top_blocks[1], top_blocks[0]]  # Swap the order
    
    for i, (row, col) in enumerate(top_blocks):
        if i < 2:  # Limit to top 2 zoom areas
            # Define zoom area - ensure it's within bounds
            r_start = max(0, row - zoom_size//4)
            r_end = min(h, row + zoom_size)
            c_start = max(0, col - zoom_size//4)
            c_end = min(w, col + zoom_size)
            
            # Create a zoomed view that overlaps with the updated map
            # Calculate the position for the zoom area (in figure coordinates)
            zoom_width = 0.2  # Width of zoom box as fraction of figure width
            zoom_height = 0.25  # Height of zoom box as fraction of figure height
            zoom_x = upd_x0 + upd_width - zoom_width * 0.8  # Position zoom to overlap slightly
            
            # Position vertically based on index (swapped)
            # Move zooms a bit downward by adjusting these positions
            if i == 0:  # Now this is Zoom 2 (swapped)
                zoom_y = upd_y0 + upd_height * 0.65  # Top zoom, moved down
            else:  # Now this is Zoom 1 (swapped)
                zoom_y = upd_y0 + upd_height * 0.20  # Bottom zoom, moved down
            
            # Create axes at the desired position
            ax_zoom = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])
            zoom_axes.append(ax_zoom)
            
            # Extract zoomed region
            zoom_img = rgb_img[r_start:r_end, c_start:c_end]
            
            # Draw a rectangle on the main image to show zoom area
            rect = plt.Rectangle((c_start, r_start), c_end-c_start, r_end-r_start, 
                               linewidth=1, edgecolor='yellow', facecolor='none')
            ax_upd.add_patch(rect)
            
            # Show zoomed image
            ax_zoom.imshow(zoom_img)
            
            # Turn off axis elements but KEEP the spines visible
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            ax_zoom.set_xticklabels([])
            ax_zoom.set_yticklabels([])
            
            # Add yellow outline to the zoom image
            for spine in ax_zoom.spines.values():
                spine.set_edgecolor('yellow')
                spine.set_linewidth(2)
                spine.set_visible(True)
                
            # Set title at the bottom with white background
            title = ax_zoom.set_title(f'Zoom {2-i}', pad=8, loc='center', y=-0.15)
            # Add white background with some padding
            title.set_bbox(dict(facecolor='white', edgecolor='none', pad=3.0))
            
            # Get the right middle point of the rectangle in data coordinates
            rect_right = c_end
            rect_middle_y = r_start + (r_end - r_start) / 2
            
            # Transform to figure coordinates
            rect_fig_coords = ax_upd.transData.transform((rect_right, rect_middle_y))
            rect_fig_coords = fig.transFigure.inverted().transform(rect_fig_coords)
            
            # Get zoom axis position in figure coordinates
            zoom_bbox = ax_zoom.get_position()
            
            # Left edge of zoom box
            zoom_fig_x = zoom_bbox.x0
            zoom_fig_y = zoom_bbox.y0 + zoom_bbox.height/2  # Middle of zoom box
            
            # Draw the connecting line - no contour, just yellow
            fig.add_artist(plt.Line2D([rect_fig_coords[0], zoom_fig_x], 
                                     [rect_fig_coords[1], zoom_fig_y], 
                                     transform=fig.transFigure, color='yellow', linestyle='-', linewidth=1.5))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure
    
    print(f"Visualization saved to {output_path}")

def create_refined_map_figure(original_map, refined_map, diff_mask, top_blocks=None, output_path="map_refinement.png"):
    """
    Create a figure showing the result of map refining with changes colored.
    
    Args:
        original_map: The original PGM map
        refined_map: The refined PGM map
        diff_mask: Binary mask where 1 indicates different pixels
        top_blocks: List of (row, col) tuples for zoomed regions (optional)
        output_path: Path to save the visualization
    """
    # Force matplotlib to use non-interactive backend
    plt.switch_backend('Agg')
    
    # Create a colored visualization
    # We'll use a 3-channel RGB image
    h, w = original_map.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize grayscale to 0-255
    if original_map.max() > 0:  # Avoid division by zero
        norm_original = (original_map / original_map.max() * 255).astype(np.uint8)
    else:
        norm_original = original_map.astype(np.uint8)
        
    if refined_map.max() > 0:  # Avoid division by zero
        norm_refined = (refined_map / refined_map.max() * 255).astype(np.uint8)
    else:
        norm_refined = refined_map.astype(np.uint8)
    
    # Copy grayscale values to all channels for base image
    rgb_img[:, :, 0] = norm_refined  # Red channel
    rgb_img[:, :, 1] = norm_refined  # Green channel
    rgb_img[:, :, 2] = norm_refined  # Blue channel
    
    # Color the changed pixels in blue instead of red
    rgb_img[diff_mask == 1, 0] = 0    # Red channel
    rgb_img[diff_mask == 1, 1] = 0    # Green channel
    rgb_img[diff_mask == 1, 2] = 255  # Blue channel (set to maximum for blue)
    
    # Create figure with a similar layout to the before-after comparison
    n_zoom = len(top_blocks) if top_blocks else 0
    
    if n_zoom > 0:
        # Create figure with similar dimensions to the before/after visualization
        fig = plt.figure(figsize=(10, 8))
        
        # Create main axis for refined map
        ax_main = plt.axes([0.1, 0.1, 0.8, 0.8])
        ax_main.imshow(rgb_img)
        ax_main.set_title('Refined Map', pad=8, loc='center', y=-0.1)
        ax_main.axis('off')
        
        # Swap zoom 1 and 2 positions as in before/after visualization
        if len(top_blocks) >= 2:
            top_blocks = [top_blocks[1], top_blocks[0]]  # Swap the order
            
        # Add zoom views
        zoom_size = min(100, 50 * 2)  # Same zoom size as in before/after
        
        for i, (row, col) in enumerate(top_blocks):
            if i < 2:  # Limit to top 2 zoom areas
                # Define zoom area - ensure it's within bounds
                r_start = max(0, row - zoom_size//4)
                r_end = min(h, row + zoom_size)
                c_start = max(0, col - zoom_size//4)
                c_end = min(w, col + zoom_size)
                
                # Create a zoomed view that overlaps with the map
                # Calculate position for zoom views
                zoom_width = 0.2  # Width as fraction of figure width
                zoom_height = 0.25  # Height as fraction of figure height
                
                # Position zooms on the right side similar to before/after
                # Move more to the left to overlap more with the map
                zoom_x = 0.6  # Moved more to the left to overlap map
                
                # Positioning vertically (swapped as in before/after)
                if i == 0:  # Now this is Zoom 2
                    zoom_y = 0.65  # Top zoom
                else:  # Now this is Zoom 1
                    zoom_y = 0.2  # Bottom zoom
                
                # Create zoom axes
                ax_zoom = fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])
                
                # Extract zoomed region
                zoom_img = rgb_img[r_start:r_end, c_start:c_end]
                
                # Draw rectangle on main image to show zoom area
                rect = plt.Rectangle((c_start, r_start), c_end-c_start, r_end-r_start,
                                   linewidth=1, edgecolor='yellow', facecolor='none')
                ax_main.add_patch(rect)
                
                # Show zoomed image
                ax_zoom.imshow(zoom_img)
                
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
                
                # Set title with white background
                title = ax_zoom.set_title(f'Zoom {2-i}', pad=8, loc='center', y=-0.15)
                title.set_bbox(dict(facecolor='white', edgecolor='none', pad=3.0))
                
                # Get the right middle point of the rectangle in data coordinates
                rect_right = c_end
                rect_middle_y = r_start + (r_end - r_start) / 2
                
                # Calculate the connecting points manually in figure coordinates
                # This ensures more precise connections regardless of transformations
                main_bbox = ax_main.get_position()
                zoom_bbox = ax_zoom.get_position()
                
                # Calculate position of rectangle in figure coordinates
                # First convert from data to axes coordinates
                main_xlim = ax_main.get_xlim()
                main_ylim = ax_main.get_ylim()
                
                # Convert data coordinates to normalized axes coordinates (0-1)
                rect_ax_x = (rect_right - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
                rect_ax_y = (rect_middle_y - main_ylim[0]) / (main_ylim[1] - main_ylim[0])
                
                # Convert axes coordinates to figure coordinates
                rect_fig_x = main_bbox.x0 + rect_ax_x * main_bbox.width
                rect_fig_y = main_bbox.y0 + rect_ax_y * main_bbox.height
                
                # Left edge middle of zoom box in figure coordinates
                zoom_fig_x = zoom_bbox.x0
                zoom_fig_y = zoom_bbox.y0 + zoom_bbox.height/2
                
                # Draw connecting line - yellow
                fig.add_artist(plt.Line2D([rect_fig_x, zoom_fig_x], 
                                         [rect_fig_y, zoom_fig_y], 
                                         transform=fig.transFigure, color='yellow', linestyle='-', linewidth=1.5))
    else:
        # Simple figure if no zoom regions
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure
    
    print(f"Refinement visualization saved to {output_path}")

def process_maps(original_path, updated_path, refined_path=None, output_dir="./results"):
    """
    Process three maps: original, updated, and refined.
    
    Args:
        original_path: Path to the original PGM map (without obstacles)
        updated_path: Path to the updated PGM map
        refined_path: Path to the refined PGM map (optional)
        output_dir: Directory to save the output visualizations
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read maps
    original_map = read_pgm(original_path)
    updated_map = read_pgm(updated_path)
    
    # Check if original map needs flipping to match orientation of updated map
    original_map = check_and_flip_if_needed(original_map, updated_map)
    
    # Compare original and updated maps
    update_diff_mask = compare_maps(original_map, updated_map)
    
    # Calculate regions with significant changes for zoomed views
    # We'll use a simple approach to find areas with the most changes
    block_size = 50  # Size of blocks to analyze
    change_density = np.zeros((update_diff_mask.shape[0] // block_size, update_diff_mask.shape[1] // block_size))
    
    for i in range(change_density.shape[0]):
        for j in range(change_density.shape[1]):
            block = update_diff_mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            change_density[i, j] = np.sum(block)
    
    # Find coordinates of top 2 blocks with most changes
    flat_indices = np.argsort(change_density.flatten())[-2:]  # Get indices of top 2 blocks
    top_blocks = []
    for idx in flat_indices:
        if change_density.flatten()[idx] > 0:  # Only include if there are changes
            i, j = np.unravel_index(idx, change_density.shape)
            top_blocks.append((i*block_size, j*block_size))
    
    # Count changes between original and updated
    num_update_changes = np.sum(update_diff_mask)
    total_pixels = update_diff_mask.size
    update_percent_changed = (num_update_changes / total_pixels) * 100
    
    print(f"Original to Updated comparison:")
    print(f"- Total pixels: {total_pixels}")
    print(f"- Changed pixels: {num_update_changes}")
    print(f"- Percentage changed: {update_percent_changed:.2f}%")
    
    # Create original vs updated visualization
    create_before_after_visualization(
        original_map, 
        updated_map, 
        update_diff_mask,
        output_path=f"{output_dir}/before_after_comparison.png"
    )
    
    # If refined map is provided, process it too
    if refined_path:
        refined_map = read_pgm(refined_path)
        
        # We already checked/flipped original_map based on updated_map,
        # so we use the same orientation for refined map comparison
        
        # Compare original and refined maps
        refined_diff_mask = compare_maps(original_map, refined_map)
        
        # Count changes between original and refined
        num_refined_changes = np.sum(refined_diff_mask)
        refined_percent_changed = (num_refined_changes / total_pixels) * 100
        
        print(f"\nOriginal to Refined comparison:")
        print(f"- Total pixels: {total_pixels}")
        print(f"- Changed pixels: {num_refined_changes}")
        print(f"- Percentage changed: {refined_percent_changed:.2f}%")
        
        # Create refined map visualization with changes highlighted
        # Pass the same top_blocks to use the same zoom regions
        create_refined_map_figure(
            original_map, 
            refined_map, 
            refined_diff_mask,
            top_blocks=top_blocks,  # Pass the zoom regions
            output_path=f"{output_dir}/refined_map_visualization.png"
        )

def main():
    """Main function to execute the map comparison pipeline with hardcoded parameters."""
    # Set these parameters to your actual file paths
    original_map_path = "/home/patrickdalager/ros2_ws_master/src/master_project2/t_map_figures/original_map.pgm"
    updated_map_path = "/home/patrickdalager/ros2_ws_master/src/master_project2/t_map_figures/updated_map.pgm"
    refined_map_path = "/home/patrickdalager/ros2_ws_master/src/master_project2/t_map_figures/refined_map.pgm"
    output_dir = "/home/patrickdalager/ros2_ws_master/src/master_project2/t_map_figures/results"                     # Output directory
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process maps
    process_maps(original_map_path, updated_map_path, refined_map_path, output_dir)
    
    print("Map comparison complete!")

if __name__ == "__main__":
    # This runs the main function when the script is executed
    main()