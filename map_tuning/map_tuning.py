#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
from PIL import Image
import yaml
import time

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Update these parameters with your specific paths and settings

# Input/Output Paths
DB_PATH = '/home/patrickdalager/ros2_ws_master/src/master_project2/map_tuning/db_mapping.db'                  # Path to your SQLite database
GROUND_TRUTH_MAP = '/home/patrickdalager/ros2_ws_master/src/master_project2/map_tuning/map.pgm'   # Path to ground truth map
CURRENT_MAP = '/home/patrickdalager/ros2_ws_master/src/master_project2/map_tuning/map_cleaned_map.pgm'         # Path to current/updated map
MAP_YAML = '/home/patrickdalager/ros2_ws_master/src/master_project2/map_tuning/map_cleaned_map.yaml'                   # Path to map metadata YAML
OUTPUT_DIR = '/home/patrickdalager/ros2_ws_master/src/master_project2/map_tuning/results'          # Directory for results

# Analysis Options
RUN_GRID_SEARCH = True                            # Perform exhaustive parameter search
TRANSFORM_GROUND_TRUTH = True                     # Apply rotation/flip to ground truth map
TRANSFORM_CURRENT_MAP = False                     # Apply rotation/flip to current map

# Logging
VERBOSE = True                                    # Print detailed logs

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(message, force=False):
    """Simple logging function"""
    if VERBOSE or force:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

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
        log(f"Error loading PGM file {file_path}: {e}", force=True)
        exit(1)

def load_map_metadata(yaml_path):
    """Load map metadata from YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        return metadata
    except Exception as e:
        log(f"Error loading map metadata: {e}", force=True)
        return None

def extract_confidence_values(db_path):
    """Extract cell confidence values from the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT cell_x, cell_y, occupied_count, free_count
            FROM cell_confidence
            WHERE occupied_count > 0 OR free_count > 0
        ''')
        
        confidence_data = {}
        for cell_x, cell_y, occ, free in cursor.fetchall():
            confidence_data[(cell_x, cell_y)] = {
                'occupied_count': occ,
                'free_count': free,
                'total': occ + free,
                'confidence': (occ - free) / (occ + free) if (occ + free) > 0 else 0
            }
        
        conn.close()
        return confidence_data
    
    except sqlite3.Error as e:
        log(f"Database error: {e}", force=True)
        return {}

def identify_misclassified_cells(ground_truth, current_map, confidence_data, map_metadata):
    """Identify false positives and false negatives and associate with confidence values"""
    origin_x = map_metadata['origin'][0]
    origin_y = map_metadata['origin'][1]
    resolution = map_metadata['resolution']
    
    height, width = ground_truth.shape
    
    # For ROS2 maps: 0=black=occupied, 255=white=free
    gt_obstacles = ground_truth == 0
    map_obstacles = current_map == 0
    
    false_positives = {}
    false_negatives = {}
    true_positives = {}
    true_negatives = {}
    
    for y in range(height):
        for x in range(width):
            # Skip cells without confidence data
            grid_coords = (x, y)
            if grid_coords not in confidence_data:
                continue
            
            conf_data = confidence_data[grid_coords]
            
            if gt_obstacles[y, x] and map_obstacles[y, x]:
                # True positive
                true_positives[grid_coords] = conf_data
            elif gt_obstacles[y, x] and not map_obstacles[y, x]:
                # False negative
                false_negatives[grid_coords] = conf_data
            elif not gt_obstacles[y, x] and map_obstacles[y, x]:
                # False positive
                false_positives[grid_coords] = conf_data
            else:
                # True negative
                true_negatives[grid_coords] = conf_data
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'true_negatives': true_negatives
    }

def analyze_confidence_distributions(misclassified_cells):
    """Analyze the distribution of confidence values for misclassified cells"""
    categories = ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']
    results = {}
    
    for category in categories:
        cells = misclassified_cells[category]
        if not cells:
            results[category] = None
            continue
        
        # Extract confidence metrics
        occupied_counts = [cell['occupied_count'] for cell in cells.values()]
        free_counts = [cell['free_count'] for cell in cells.values()]
        confidence_values = [cell['confidence'] for cell in cells.values()]
        
        # Calculate statistics
        results[category] = {
            'count': len(cells),
            'occupied_counts': {
                'min': min(occupied_counts) if occupied_counts else 0,
                'max': max(occupied_counts) if occupied_counts else 0,
                'mean': np.mean(occupied_counts) if occupied_counts else 0,
                'median': np.median(occupied_counts) if occupied_counts else 0,
                'percentiles': {
                    '25': np.percentile(occupied_counts, 25) if occupied_counts else 0,
                    '75': np.percentile(occupied_counts, 75) if occupied_counts else 0,
                    '90': np.percentile(occupied_counts, 90) if occupied_counts else 0,
                }
            },
            'free_counts': {
                'min': min(free_counts) if free_counts else 0,
                'max': max(free_counts) if free_counts else 0,
                'mean': np.mean(free_counts) if free_counts else 0,
                'median': np.median(free_counts) if free_counts else 0,
                'percentiles': {
                    '25': np.percentile(free_counts, 25) if free_counts else 0,
                    '75': np.percentile(free_counts, 75) if free_counts else 0,
                    '90': np.percentile(free_counts, 90) if free_counts else 0,
                }
            },
            'confidence_values': confidence_values
        }
    
    return results

def suggest_thresholds(analysis_results, misclassified_cells):
    """Suggest optimal thresholds based on the analysis"""
    recommended_thresholds = {}
    
    # Extract occupied counts for false positives and false negatives
    fp_occupied = [cell['occupied_count'] for cell in misclassified_cells['false_positives'].values()] if misclassified_cells['false_positives'] else []
    fn_occupied = [cell['occupied_count'] for cell in misclassified_cells['false_negatives'].values()] if misclassified_cells['false_negatives'] else []
    
    # If we have enough data, find a threshold that would reduce false positives
    if fp_occupied:
        # A higher threshold would reduce false positives
        # Look at the distribution and suggest the 75th percentile as a starting point
        fp_threshold = np.percentile(fp_occupied, 75)
        recommended_thresholds['reduce_false_positives'] = max(1, int(fp_threshold + 1))
    
    # If we have false negatives, find a threshold that would reduce them
    if fn_occupied:
        # A lower threshold could help with false negatives
        # Look at the distribution and suggest a threshold below the 25th percentile
        fn_threshold = np.percentile(fn_occupied, 25)
        recommended_thresholds['reduce_false_negatives'] = max(1, int(fn_threshold - 1))
    
    # Balance threshold - median between the two if both exist
    if 'reduce_false_positives' in recommended_thresholds and 'reduce_false_negatives' in recommended_thresholds:
        balance_threshold = (recommended_thresholds['reduce_false_positives'] + recommended_thresholds['reduce_false_negatives']) // 2
        recommended_thresholds['balanced'] = max(1, balance_threshold)
    
    # For verification_threshold, suggest based on confidence threshold
    if 'balanced' in recommended_thresholds:
        # If we have a balanced confidence threshold, suggest verification threshold
        # proportional to it to ensure enough time for confidence to build up
        recommended_thresholds['verification_threshold'] = max(3, recommended_thresholds['balanced'] // 2)
    else:
        # Default suggestion
        recommended_thresholds['verification_threshold'] = 5
    
    return recommended_thresholds

def visualize_confidence_distributions(analysis_results, misclassified_cells, output_dir):
    """Visualize the distribution of confidence values for different cell categories"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot histograms of confidence values
    plt.figure(figsize=(12, 10))
    
    categories = ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']
    colors = ['blue', 'red', 'green', 'gray']
    
    for i, category in enumerate(categories):
        if analysis_results[category] is None or not analysis_results[category]['confidence_values']:
            continue
            
        values = analysis_results[category]['confidence_values']
        plt.subplot(2, 2, i+1)
        plt.hist(values, bins=30, alpha=0.7, color=colors[i])
        plt.title(f"{category.replace('_', ' ').title()} (n={len(values)})")
        plt.xlabel('Confidence Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distributions.png'), dpi=300)
    plt.close()
    
    # Plot occupied vs free counts for each category
    plt.figure(figsize=(12, 10))
    
    for i, category in enumerate(categories):
        if analysis_results[category] is None:
            continue
            
        plt.subplot(2, 2, i+1)
        
        occupied_counts = [cell['occupied_count'] for cell in misclassified_cells[category].values()]
        free_counts = [cell['free_count'] for cell in misclassified_cells[category].values()]
        
        plt.scatter(occupied_counts, free_counts, alpha=0.5, s=10, color=colors[i])
        plt.title(f"{category.replace('_', ' ').title()} (n={len(occupied_counts)})")
        plt.xlabel('Occupied Count')
        plt.ylabel('Free Count')
        plt.grid(alpha=0.3)
        
        # Add diagonal reference line
        max_val = max(max(occupied_counts, default=0), max(free_counts, default=0))
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'occupied_vs_free_counts.png'), dpi=300)
    plt.close()
    
    # Plot impact of different confidence thresholds
    if misclassified_cells['false_positives'] and misclassified_cells['false_negatives']:
        thresholds = range(1, 21)  # Try thresholds from 1 to 20
        fp_rates = []
        fn_rates = []
        
        # Get total counts for normalization
        total_fp = len(misclassified_cells['false_positives'])
        total_fn = len(misclassified_cells['false_negatives'])
        
        for threshold in thresholds:
            # Count how many false positives would remain with this threshold
            remaining_fp = sum(1 for cell in misclassified_cells['false_positives'].values() 
                              if cell['occupied_count'] >= threshold)
            
            # Count how many false negatives would be identified with this threshold
            corrected_fn = sum(1 for cell in misclassified_cells['false_negatives'].values() 
                              if cell['occupied_count'] >= threshold)
            
            # Calculate rates
            fp_rate = remaining_fp / total_fp if total_fp > 0 else 0
            fn_correction_rate = corrected_fn / total_fn if total_fn > 0 else 0
            
            fp_rates.append(fp_rate)
            fn_rates.append(fn_correction_rate)
        
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, fp_rates, 'b-', label='False Positive Rate')
        plt.plot(thresholds, fn_rates, 'r-', label='False Negative Correction Rate')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Rate')
        plt.title('Effect of Confidence Threshold on Error Rates')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_impact.png'), dpi=300)
        plt.close()

def apply_threshold_to_map(confidence_data, map_shape, conf_th, map_metadata):
    """Apply threshold to generate a map"""
    height, width = map_shape
    simulated_map = np.full((height, width), 255, dtype=np.uint8)  # Default to free
    
    origin_x = map_metadata['origin'][0]
    origin_y = map_metadata['origin'][1]
    resolution = map_metadata['resolution']
    
    # Apply confidence threshold to each cell
    for (cell_x, cell_y), data in confidence_data.items():
        # Skip cells outside map bounds
        if not (0 <= cell_y < height and 0 <= cell_x < width):
            continue
        
        # Apply confidence threshold
        if data['occupied_count'] >= conf_th:
            simulated_map[cell_y, cell_x] = 0  # Set as obstacle
        elif data['free_count'] >= conf_th:
            simulated_map[cell_y, cell_x] = 255  # Set as free
    
    return simulated_map

def calculate_map_metrics(ground_truth, simulated_map):
    """Calculate precision, recall and other metrics for a map"""
    # For ROS2 maps: 0=black=occupied, 255=white=free
    gt_obstacles = ground_truth == 0
    map_obstacles = simulated_map == 0
    
    # Calculate confusion matrix
    true_positives = np.logical_and(gt_obstacles, map_obstacles)
    false_negatives = np.logical_and(gt_obstacles, ~map_obstacles)
    false_positives = np.logical_and(~gt_obstacles, map_obstacles)
    true_negatives = np.logical_and(~gt_obstacles, ~map_obstacles)
    
    # Calculate metrics
    tp_count = np.sum(true_positives)
    fn_count = np.sum(false_negatives)
    fp_count = np.sum(false_positives)
    tn_count = np.sum(true_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'tp': tp_count,
        'fn': fn_count,
        'fp': fp_count,
        'tn': tn_count
    }

def grid_search_optimal_parameters(confidence_data, ground_truth, map_metadata, output_dir):
    """Run a grid search to find optimal parameters"""
    # Define parameter ranges to search
    confidence_thresholds = range(1, 16)  # 1 to 15
    verification_thresholds = range(1, 11)  # 1 to 10
    
    # Measures to optimize
    best_score = -1
    best_params = None
    results = []
    
    log("Starting grid search...")
    
    # Grid search
    for conf_th in confidence_thresholds:
        # For a simplified version, we just analyze how different confidence thresholds 
        # affect map performance. Verification threshold is primarily about stability
        # and isn't directly comparable in a single-step analysis.
        
        # Apply threshold to generate a simulated map
        simulated_map = apply_threshold_to_map(confidence_data, ground_truth.shape, 
                                             conf_th, map_metadata)
        
        # Calculate metrics
        metrics = calculate_map_metrics(ground_truth, simulated_map)
        
        # Use F1 score as overall metric (harmonic mean of precision and recall)
        f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        results.append({
            'confidence_threshold': conf_th,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': f1_score
        })
        
        if f1_score > best_score:
            best_score = f1_score
            best_params = conf_th
                
        log(f"Tested conf_th={conf_th}: F1={f1_score:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
    
    # Visualize grid search results
    visualize_thresholds_impact(results, output_dir)
    
    # Return the best parameters
    return best_params, results

def visualize_thresholds_impact(results, output_dir):
    """Visualize impact of different threshold values on map quality"""
    conf_thresholds = [r['confidence_threshold'] for r in results]
    precision_values = [r['precision'] for r in results]
    recall_values = [r['recall'] for r in results]
    f1_values = [r['f1_score'] for r in results]
    
    plt.figure(figsize=(10, 8))
    plt.plot(conf_thresholds, precision_values, 'b-', label='Precision', marker='o')
    plt.plot(conf_thresholds, recall_values, 'r-', label='Recall', marker='s')
    plt.plot(conf_thresholds, f1_values, 'g-', label='F1 Score', marker='^')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Confidence Threshold on Map Quality')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_metrics.png'), dpi=300)
    plt.close()
    
    # Find optimal threshold (highest F1 score)
    best_idx = np.argmax(f1_values)
    best_threshold = conf_thresholds[best_idx]
    best_f1 = f1_values[best_idx]
    
    # Create a detailed visualization of the optimal threshold
    plt.figure(figsize=(10, 8))
    plt.plot(conf_thresholds, f1_values, 'g-', marker='^')
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.text(best_threshold + 0.5, best_f1, f'Optimal: {best_threshold}', 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('Optimal Confidence Threshold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_threshold.png'), dpi=300)
    plt.close()
    
    # Save results to file
    with open(os.path.join(output_dir, 'threshold_analysis.txt'), 'w') as f:
        f.write("=== CONFIDENCE THRESHOLD ANALYSIS ===\n\n")
        f.write("Optimal confidence threshold: " + str(best_threshold) + "\n")
        f.write("F1 Score at optimal threshold: " + str(best_f1) + "\n\n")
        f.write("Threshold, Precision, Recall, F1 Score\n")
        for i, thresh in enumerate(conf_thresholds):
            f.write(f"{thresh}, {precision_values[i]:.4f}, {recall_values[i]:.4f}, {f1_values[i]:.4f}\n")

def run_parameter_tuning():
    """Run the parameter tuning analysis"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load maps
    log(f"Loading ground truth map from {GROUND_TRUTH_MAP}", force=True)
    ground_truth = load_pgm(GROUND_TRUTH_MAP, apply_transforms=TRANSFORM_GROUND_TRUTH)
    
    log(f"Loading current map from {CURRENT_MAP}", force=True)
    current_map = load_pgm(CURRENT_MAP, apply_transforms=TRANSFORM_CURRENT_MAP)
    
    # Load map metadata
    log(f"Loading map metadata from {MAP_YAML}", force=True)
    map_metadata = load_map_metadata(MAP_YAML)
    if not map_metadata:
        log("Error: Could not load map metadata. Exiting.", force=True)
        exit(1)
    
    # Extract confidence values from database
    log(f"Extracting confidence values from database at {DB_PATH}", force=True)
    confidence_data = extract_confidence_values(DB_PATH)
    log(f"Found {len(confidence_data)} cells with confidence data", force=True)
    
    # Identify misclassified cells
    log("Identifying misclassified cells and analyzing confidence distributions...", force=True)
    misclassified_cells = identify_misclassified_cells(ground_truth, current_map, confidence_data, map_metadata)
    
    # Calculate statistics
    fp_count = len(misclassified_cells['false_positives'])
    fn_count = len(misclassified_cells['false_negatives'])
    tp_count = len(misclassified_cells['true_positives'])
    tn_count = len(misclassified_cells['true_negatives'])
    
    log(f"Map analysis complete:", force=True)
    log(f"True Positives: {tp_count} cells", force=True)
    log(f"False Positives: {fp_count} cells", force=True)
    log(f"False Negatives: {fn_count} cells", force=True)
    log(f"True Negatives: {tn_count} cells", force=True)
    
    # Analyze confidence distributions
    analysis_results = analyze_confidence_distributions(misclassified_cells)
    
    # Visualize results
    log("Visualizing confidence distributions...", force=True)
    visualize_confidence_distributions(analysis_results, misclassified_cells, OUTPUT_DIR)
    
    # Suggest thresholds
    log("Suggesting optimal thresholds based on analysis...", force=True)
    suggested_thresholds = suggest_thresholds(analysis_results, misclassified_cells)
    
    log("Suggested thresholds:", force=True)
    for purpose, value in suggested_thresholds.items():
        log(f"  {purpose}: {value}", force=True)
    
    # Write recommendations to a file
    with open(os.path.join(OUTPUT_DIR, 'threshold_recommendations.txt'), 'w') as f:
        f.write("=== THRESHOLD RECOMMENDATIONS ===\n\n")
        f.write("Based on the analysis of confidence values in misclassified cells, the following thresholds are recommended:\n\n")
        
        for purpose, value in suggested_thresholds.items():
            f.write(f"{purpose}: {value}\n")
        
        f.write("\nRecommendation summary:\n")
        
        if 'balanced' in suggested_thresholds:
            f.write(f"For a balanced approach, we recommend:\n")
            f.write(f"- confidence_threshold: {suggested_thresholds['balanced']}\n")
            f.write(f"- verification_threshold: {suggested_thresholds['verification_threshold']}\n\n")
        
        f.write("These recommendations aim to balance the tradeoff between reducing false positives and false negatives.\n")
        f.write("Adjust these values based on whether you prioritize map completeness (lower threshold) or accuracy (higher threshold).\n")
    
    # Run grid search if requested
    if RUN_GRID_SEARCH:
        log("Starting grid search for optimal parameters...", force=True)
        best_conf, grid_results = grid_search_optimal_parameters(
            confidence_data, ground_truth, map_metadata, OUTPUT_DIR)
        
        log(f"Grid search recommendation:", force=True)
        log(f"  confidence_threshold: {best_conf}", force=True)
        # For verification threshold, a reasonable rule of thumb:
        suggested_ver = max(3, best_conf // 2)
        log(f"  verification_threshold: {suggested_ver}", force=True)
        
        # Update the recommendations file with grid search results
        with open(os.path.join(OUTPUT_DIR, 'threshold_recommendations.txt'), 'a') as f:
            f.write("\n\n=== GRID SEARCH RECOMMENDATION ===\n\n")
            f.write(f"Based on exhaustive grid search, the optimal thresholds are:\n")
            f.write(f"- confidence_threshold: {best_conf}\n")
            f.write(f"- verification_threshold: {suggested_ver}\n")
    
    log(f"Analysis complete. Results saved to {OUTPUT_DIR}", force=True)
    
    # Return the suggested thresholds
    return suggested_thresholds

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    log("Starting parameter tuning script...", force=True)
    try:
        thresholds = run_parameter_tuning()
        
        # Display recommended parameters
        log("\nRECOMMENDED PARAMETERS:", force=True)
        if 'balanced' in thresholds:
            log(f"confidence_threshold: {thresholds['balanced']}", force=True)
            log(f"verification_threshold: {thresholds['verification_threshold']}", force=True)
        else:
            if 'reduce_false_positives' in thresholds:
                log(f"To reduce false positives: confidence_threshold={thresholds['reduce_false_positives']}", force=True)
            if 'reduce_false_negatives' in thresholds:
                log(f"To reduce false negatives: confidence_threshold={thresholds['reduce_false_negatives']}", force=True)
        
        log(f"\nCheck {OUTPUT_DIR} for detailed analysis and visualizations.", force=True)
        
    except Exception as e:
        log(f"Error during parameter tuning: {str(e)}", force=True)
        import traceback
        traceback.print_exc()