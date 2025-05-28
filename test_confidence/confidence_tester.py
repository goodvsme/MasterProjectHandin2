#!/usr/bin/env python3

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

#-------------------------------------------------------------------
# CONFIGURATION: Edit these values to set your database and output paths
#-------------------------------------------------------------------
# Path to your SQLite database file
DATABASE_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_confidence/db_mapping.db"

# Path where the output image will be saved
# IMPORTANT: Make sure this is a valid directory path (not inside the database file)
OUTPUT_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_confidence/results/confidence_map.png"

# Resolution of the map in meters per pixel
RESOLUTION = 0.1

# Colormap to use ('custom', 'viridis', 'jet', 'plasma', 'inferno', 'magma', 'cividis')
COLORMAP = 'custom'

# Set to True to automatically detect and zoom into region with most activity
AUTO_FOCUS = True

# Manual region selection (only used if AUTO_FOCUS is False)
# These are grid cell coordinates
REGION_X_MIN = None  # Will be set based on data
REGION_X_MAX = None  # Will be set based on data
REGION_Y_MIN = None  # Will be set based on data
REGION_Y_MAX = None  # Will be set based on data

# Number of cells to use for active region detection
ACTIVE_REGION_SIZE = 150

# Limit region to this many cells at most (prevents huge plots)
MAX_REGION_SIZE = 250

# Create both focused and full map
CREATE_FULL_MAP = True
#-------------------------------------------------------------------

def create_confidence_map(db_path, output_path, resolution=0.1, colormap='custom',
                        auto_focus=True, region_coords=None, 
                        active_region_size=150, max_region_size=250,
                        create_full_map=True):
    """
    Create a confidence map visualization from the mapping database.
    
    Args:
        db_path (str): Path to the SQLite database file
        output_path (str): Path to save the output image
        resolution (float): Resolution of the map in meters per pixel
        colormap (str): Colormap to use ('custom', 'viridis', 'jet', etc.)
        auto_focus (bool): Whether to automatically focus on active region
        region_coords (tuple): Manual region coordinates (x_min, x_max, y_min, y_max)
        active_region_size (int): Size of active region to detect (if auto_focus is True)
        max_region_size (int): Maximum size of region to visualize
        create_full_map (bool): Whether to create visualizations of the full map too
    """
    # Connect to the database
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all cell confidence data
    cursor.execute("SELECT cell_x, cell_y, occupied_count, free_count FROM cell_confidence")
    cells = cursor.fetchall()
    
    if not cells:
        print("No cell confidence data found in the database.")
        conn.close()
        return
    
    print(f"Found {len(cells)} cells with confidence data")
    
    # Find the bounds of the map
    min_x = min(cell[0] for cell in cells)
    max_x = max(cell[0] for cell in cells)
    min_y = min(cell[1] for cell in cells)
    max_y = max(cell[1] for cell in cells)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    print(f"Map dimensions: {width} x {height} cells")
    print(f"Map bounds: X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
    
    # Create empty arrays for the confidence map
    # We'll use three channels: occupied probability, free probability, and total observations (for normalization)
    confidence = np.zeros((height, width, 3), dtype=float)
    
    # Create an array to track observation counts for region detection
    observation_count = np.zeros((height, width), dtype=int)
    
    # Fill the confidence map
    for cell_x, cell_y, occupied, free in cells:
        # Adjust coordinates to start from 0
        array_x = cell_x - min_x
        array_y = cell_y - min_y
        
        # Skip cells with no observations
        total_obs = occupied + free
        if total_obs == 0:
            continue
        
        # Store total observations for active region detection
        observation_count[array_y, array_x] = total_obs
        
        # Store occupied probability, free probability, and total observations
        confidence[array_y, array_x, 0] = occupied / total_obs  # Occupied probability
        confidence[array_y, array_x, 1] = free / total_obs      # Free probability
        confidence[array_y, array_x, 2] = total_obs             # Total observations for normalization
    
    # Close the database connection
    conn.close()
    
    # Determine the region to visualize
    if auto_focus:
        # Find the region with the most activity (observations)
        region = find_active_region(observation_count, active_region_size, max_region_size)
        print(f"Auto-detected active region: {region}")
    elif region_coords:
        # Use manually specified region
        region = region_coords
        print(f"Using manual region: {region}")
    else:
        # Use full map but limit size
        if width > max_region_size or height > max_region_size:
            # Find center of mass of activity
            y_indices, x_indices = np.where(observation_count > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                half_size = min(max_region_size // 2, width // 2, height // 2)
                region = (
                    max(0, center_x - half_size),
                    min(width - 1, center_x + half_size),
                    max(0, center_y - half_size),
                    min(height - 1, center_y + half_size)
                )
            else:
                # Fallback if no activity
                region = (0, min(width - 1, max_region_size - 1), 
                          0, min(height - 1, max_region_size - 1))
        else:
            # Use entire map
            region = (0, width - 1, 0, height - 1)
        print(f"Using centered region: {region}")
    
    # Extract the focused region
    x_min, x_max, y_min, y_max = region
    focused_map = confidence[y_min:y_max+1, x_min:x_max+1, :]
    
    # Create the focused visualization
    focused_output_path = output_path
    if create_full_map:
        base, ext = os.path.splitext(output_path)
        focused_output_path = f"{base}_focused{ext}"
    
    create_visualization(focused_map, focused_output_path, resolution, 
                         min_x + x_min, min_y + y_min, colormap)
    
    # Create visualization of the full map if requested
    if create_full_map:
        full_map_path = output_path
        print(f"Creating full map visualization at {full_map_path}")
        create_visualization(confidence, full_map_path, resolution,
                            min_x, min_y, colormap)
    
    print(f"Confidence map saved to {output_path}")

def find_active_region(observation_count, region_size, max_size):
    """
    Find the region with the most activity (observations).
    
    Args:
        observation_count (numpy.ndarray): 2D array with observation counts
        region_size (int): Size of region to find
        max_size (int): Maximum region size
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max) region coordinates
    """
    height, width = observation_count.shape
    max_sum = -1
    best_region = (0, min(width-1, max_size-1), 0, min(height-1, max_size-1))
    
    # Limit region size
    region_size = min(region_size, max_size)
    
    # If map is smaller than desired region, return the whole map
    if width <= region_size and height <= region_size:
        return (0, width-1, 0, height-1)
    
    # Calculate activity in sliding windows to find most active region
    for y in range(max(0, height - region_size + 1)):
        for x in range(max(0, width - region_size + 1)):
            x_max = min(x + region_size - 1, width - 1)
            y_max = min(y + region_size - 1, height - 1)
            
            # Sum of observations in this region
            region_sum = np.sum(observation_count[y:y_max+1, x:x_max+1])
            
            if region_sum > max_sum:
                max_sum = region_sum
                best_region = (x, x_max, y, y_max)
    
    return best_region

def create_visualization(confidence, output_path, resolution, min_x, min_y, colormap_name):
    """
    Create and save separate visualizations of the occupancy map and confidence map.
    
    Args:
        confidence (numpy.ndarray): 3D array with occupied probability, free probability, and observation count
        output_path (str): Path to save the output image
        resolution (float): Resolution of the map in meters per pixel
        min_x, min_y (int): Minimum x and y coordinates for labeling
        colormap_name (str): Name of the colormap to use
    """
    height, width, _ = confidence.shape
    
    # Get the base file path without extension
    base, ext = os.path.splitext(output_path)
    occupancy_path = f"{base}_occupancy{ext}"
    confidence_path = f"{base}_confidence{ext}"
    
    # Calculate the physical extents in meters
    x_min_meters = min_x * resolution
    y_min_meters = min_y * resolution
    x_max_meters = (min_x + width - 1) * resolution
    y_max_meters = (min_y + height - 1) * resolution
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 16,              # Base font size
        'axes.titlesize': 20,         # Title font size
        'axes.labelsize': 18,         # Axis label font size
        'xtick.labelsize': 16,        # X-axis tick label size
        'ytick.labelsize': 16,        # Y-axis tick label size
        'legend.fontsize': 16,        # Legend font size
        'figure.titlesize': 22        # Figure title font size
    })
    
    # Create the occupancy map visualization
    fig_occ, ax_occ = plt.subplots(figsize=(10, 10))
    
    # Create a custom colormap with brighter colors and white background
    if colormap_name == 'custom':
        cmap = LinearSegmentedColormap.from_list('occupancy', [
            (1, 1, 1),      # White for no observations/background
            (0.2, 1, 0.2),  # Bright green for free
            (1, 1, 0),      # Bright yellow for uncertain
            (1, 0.2, 0.2)   # Bright red for occupied
        ])
    else:
        cmap = plt.get_cmap(colormap_name)
    
    # Create occupancy image with white background
    occupancy_img = np.ones((height, width, 4))  # RGBA with white background
    
    # Areas with observations
    has_observations = confidence[:, :, 2] > 0
    
    # All pixels visible (no transparency)
    occupancy_img[:, :, 3] = 1.0
    
    # Use a simpler approach with loops to avoid array shape issues
    for y in range(height):
        for x in range(width):
            if has_observations[y, x]:
                occ_prob = confidence[y, x, 0]
                if colormap_name == 'custom':
                    # Custom coloring: use occ_prob directly (0=free, 1=occupied)
                    occupancy_img[y, x, 0:3] = cmap(occ_prob)[0:3]
                else:
                    # Use the selected colormap
                    occupancy_img[y, x, 0:3] = cmap(occ_prob)[0:3]
    
    # Display the occupancy image
    img_plot = ax_occ.imshow(occupancy_img, origin='lower',
                          extent=[x_min_meters, x_max_meters, y_min_meters, y_max_meters])
    ax_occ.set_title('Occupancy Map')
    ax_occ.set_xlabel('X (m)')
    ax_occ.set_ylabel('Y (m)')
    ax_occ.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend for occupancy map
    if colormap_name == 'custom':
        # Create legend patches
        occupied_patch = mpatches.Patch(color=(1, 0.2, 0.2), label='Occupied')
        free_patch = mpatches.Patch(color=(0.2, 1, 0.2), label='Free')
        mixed_patch = mpatches.Patch(color=(1, 1, 0), label='Mixed')
        # Place legend outside the plot to avoid overlapping with the data
        ax_occ.legend(handles=[occupied_patch, free_patch, mixed_patch], 
                      loc='lower center', bbox_to_anchor=(0.5, -0.15),
                      ncol=3, framealpha=0.7)
    
    # Add physical scale in meters
    scale_x = width * resolution
    scale_y = height * resolution
    ax_occ.text(0.98, 0.02, f'Size: {scale_x:.2f}m × {scale_y:.2f}m', 
             transform=ax_occ.transAxes, horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the occupancy map figure
    plt.tight_layout()
    plt.savefig(occupancy_path, dpi=300, bbox_inches='tight')
    plt.close(fig_occ)
    
    # Create the confidence visualization
    fig_conf, ax_conf = plt.subplots(figsize=(10, 10))
    
    total_obs = confidence[:, :, 2]
    max_obs = np.max(total_obs)
    if max_obs > 0:
        # Normalize by the maximum observation count
        confidence_img = total_obs / max_obs
    else:
        confidence_img = total_obs
    
    # Use inverted viridis colormap (light to dark)
    conf_plot = ax_conf.imshow(confidence_img, cmap='viridis_r', origin='lower',
                             extent=[x_min_meters, x_max_meters, y_min_meters, y_max_meters])
    
    ax_conf.set_title('Observation Confidence')
    ax_conf.set_xlabel('X (m)')
    ax_conf.set_ylabel('Y (m)')
    ax_conf.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add a colorbar for the confidence
    cbar = plt.colorbar(conf_plot, ax=ax_conf)
    cbar.set_label('Normalized Observation Count')
    
    # Add physical scale
    ax_conf.text(0.98, 0.02, f'Size: {scale_x:.2f}m × {scale_y:.2f}m', 
             transform=ax_conf.transAxes, horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the confidence map figure
    plt.tight_layout()
    plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
    plt.close(fig_conf)
    
    # Create and save the combined figure (original behavior)
    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Recreate the occupancy plot in the combined figure
    ax1.imshow(occupancy_img, origin='lower',
              extent=[x_min_meters, x_max_meters, y_min_meters, y_max_meters])
    ax1.set_title('Occupancy Map')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend for occupancy map in combined figure
    if colormap_name == 'custom':
        occupied_patch = mpatches.Patch(color=(1, 0.2, 0.2), label='Occupied')
        free_patch = mpatches.Patch(color=(0.2, 1, 0.2), label='Free')
        mixed_patch = mpatches.Patch(color=(1, 1, 0), label='Mixed')
        # Place legend in a better position that won't overlap with the data
        ax1.legend(handles=[occupied_patch, free_patch, mixed_patch], 
                   loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3, framealpha=0.7)
    
    # Recreate the confidence plot in the combined figure (with inverted colormap)
    conf_plot_combined = ax2.imshow(confidence_img, cmap='viridis_r', origin='lower',
                                  extent=[x_min_meters, x_max_meters, y_min_meters, y_max_meters])
    ax2.set_title('Observation Confidence')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add a colorbar for the confidence
    cbar = fig_combined.colorbar(conf_plot_combined, ax=ax2)
    cbar.set_label('Normalized Observation Count')
    
    # First apply tight_layout to optimize overall spacing
    plt.tight_layout()
    
    # Add some space between the two images
    plt.subplots_adjust(wspace=0.18)
    
    # Save the combined figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Save data as numpy array for later analysis if needed
    np.save(f"{base}_data.npy", confidence)
    
    plt.close(fig_combined)

    
def main():
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create the confidence map using the configured values
    print(f"Using database: {DATABASE_PATH}")
    print(f"Saving output to: {OUTPUT_PATH}")
    print(f"Resolution: {RESOLUTION} meters per pixel")
    print(f"Colormap: {COLORMAP}")
    print(f"Auto-focus on active region: {AUTO_FOCUS}")
    print(f"Create full map: {CREATE_FULL_MAP}")
    
    # Region coordinates (only used if AUTO_FOCUS is False)
    region_coords = None
    if not AUTO_FOCUS and all(x is not None for x in [REGION_X_MIN, REGION_X_MAX, REGION_Y_MIN, REGION_Y_MAX]):
        region_coords = (REGION_X_MIN, REGION_X_MAX, REGION_Y_MIN, REGION_Y_MAX)
    
    create_confidence_map(
        DATABASE_PATH, 
        OUTPUT_PATH, 
        RESOLUTION, 
        COLORMAP,
        auto_focus=AUTO_FOCUS,
        region_coords=region_coords,
        active_region_size=ACTIVE_REGION_SIZE,
        max_region_size=MAX_REGION_SIZE,
        create_full_map=CREATE_FULL_MAP
    )
    
    print("\nDone! To change settings, edit the variables at the top of this script.")

if __name__ == '__main__':
    main()