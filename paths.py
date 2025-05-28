import json
import heapq
import random
import math
import numpy as np

def load_graph(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    nodes = {node['id']: (node['x'], node['y']) for node in data['nodes']}
    edges = {node['id']: [] for node in data['nodes']}
    
    for edge in data.get('edges', []):
        edges[edge['source']].append(edge['target'])
        edges[edge['target']].append(edge['source'])
    
    return nodes, edges

def calculate_path_length(graph, path):
    """Calculate the total length of a path based on Euclidean distance."""
    if len(path) < 2:
        return 0
        
    total_length = 0
    for i in range(len(path) - 1):
        node1 = graph[path[i]]
        node2 = graph[path[i+1]]
        distance = math.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)
        total_length += distance
    
    return total_length

def truncate_path(graph, path, max_length):
    """Truncate a path to not exceed the maximum length."""
    if len(path) < 2:
        return path, 0
        
    truncated_path = [path[0]]
    current_length = 0
    
    for i in range(1, len(path)):
        node1 = graph[path[i-1]]
        node2 = graph[path[i]]
        segment_length = math.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)
        
        # If adding this segment would exceed max_length, stop here
        if current_length + segment_length > max_length:
            break
            
        truncated_path.append(path[i])
        current_length += segment_length
    
    return truncated_path, current_length

def heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def a_star_search(graph, edges, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(graph[start], graph[goal])

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in edges.get(current, []):
            tentative_g_score = g_score[current] + heuristic(graph[current], graph[neighbor])
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(graph[neighbor], graph[goal])
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def find_distant_node(graph, start_node, percentage=0.8):
    """Find a node that's distant from the starting node (in the top X% of distances)."""
    start_pos = graph[start_node]
    
    # Calculate distances to all other nodes
    distances = []
    for node_id, pos in graph.items():
        if node_id != start_node:
            dist = heuristic(start_pos, pos)
            distances.append((dist, node_id))
    
    # Sort by distance
    distances.sort(reverse=True)
    
    # Take one from the top X% distant nodes
    cutoff = int(len(distances) * percentage)
    if cutoff == 0:
        cutoff = 1
    
    # Return a random one from the distant nodes
    selected_index = random.randint(0, min(cutoff, len(distances)-1))
    return distances[selected_index][1]

def create_multi_segment_path(graph, edges, min_length, max_length, max_segments=8):
    """Create a multi-segment path with length between min and max."""
    nodes = list(graph.keys())
    
    # Start with a random node
    current_node = random.choice(nodes)
    full_path = [current_node]
    current_length = 0
    
    for _ in range(max_segments):
        # Find a distant node for the next segment
        next_node = find_distant_node(graph, current_node)
        
        # Find path to this node
        segment_path = a_star_search(graph, edges, current_node, next_node)
        
        if segment_path and len(segment_path) > 1:
            # Check if adding this segment would exceed max_length
            temp_path = full_path.copy()
            temp_path.extend(segment_path[1:])
            temp_length = calculate_path_length(graph, temp_path)
            
            if temp_length > max_length:
                # If we already meet min_length, we can stop here
                if current_length >= min_length:
                    break
                    
                # Otherwise, truncate the segment to fit max_length
                truncated_path, _ = truncate_path(graph, temp_path, max_length)
                full_path = truncated_path
                current_length = calculate_path_length(graph, full_path)
                break
            
            # Add the segment
            full_path.extend(segment_path[1:])
            
            # Update current node and recalculate length
            current_node = next_node
            current_length = temp_length
            
            # If we've reached the minimum length, we can stop
            if current_length >= min_length:
                break
    
    return full_path, current_length

def create_figure_eight(graph, edges, min_length, max_length):
    """Create a figure-eight path by finding two loops that share a node."""
    nodes = list(graph.keys())
    
    # Start with a random node
    center_node = random.choice(nodes)
    
    # Find two other nodes to create loops with
    distant_nodes = []
    for _ in range(2):
        distant_node = find_distant_node(graph, center_node, percentage=0.5)
        distant_nodes.append(distant_node)
    
    # Create two loops
    first_half = a_star_search(graph, edges, center_node, distant_nodes[0])
    second_half = a_star_search(graph, edges, center_node, distant_nodes[1])
    
    if not first_half or not second_half:
        return None, 0
    
    # Combine the loops into a figure-eight
    # The center node appears at the beginning of both paths
    figure_eight = first_half + second_half[1:]
    
    # Add the center node again to close the loop
    figure_eight.append(center_node)
    
    # Check if the path is within length constraints
    path_length = calculate_path_length(graph, figure_eight)
    
    if path_length > max_length:
        # Truncate if necessary
        figure_eight, path_length = truncate_path(graph, figure_eight, max_length)
    
    if path_length < min_length:
        return None, 0  # Path is too short
    
    return figure_eight, path_length

def create_spiral_path(graph, edges, center_node, min_length, max_length, max_iterations=5):
    """Create a spiral-like path that winds outward from a center point."""
    nodes = list(graph.keys())
    
    if center_node is None:
        center_node = random.choice(nodes)
        
    full_path = [center_node]
    current_length = 0
    
    # Try to create a spiral by moving to increasingly distant nodes
    for i in range(max_iterations):
        # Find progressively more distant nodes
        distance_percentage = 0.2 + (i * 0.15)  # 0.2, 0.35, 0.5, 0.65, 0.8
        next_node = find_distant_node(graph, center_node, percentage=min(distance_percentage, 0.9))
        
        # Find path to this node
        segment = a_star_search(graph, edges, full_path[-1], next_node)
        
        if segment and len(segment) > 1:
            # Check if adding this segment would exceed max_length
            temp_path = full_path.copy()
            temp_path.extend(segment[1:])
            temp_length = calculate_path_length(graph, temp_path)
            
            if temp_length > max_length:
                # If we already meet min_length, we can stop here
                if current_length >= min_length:
                    break
                    
                # Otherwise, truncate the segment to fit max_length
                truncated_path, _ = truncate_path(graph, temp_path, max_length)
                full_path = truncated_path
                current_length = calculate_path_length(graph, full_path)
                break
            
            # Add the segment (excluding first node to avoid duplication)
            full_path.extend(segment[1:])
            current_length = temp_length
            
            if current_length >= min_length:
                break
    
    # If we're still below minimum length, try to circle back to center
    if current_length < min_length and len(full_path) > 1:
        closing_segment = a_star_search(graph, edges, full_path[-1], center_node)
        if closing_segment and len(closing_segment) > 1:
            # Check max length first
            temp_path = full_path.copy()
            temp_path.extend(closing_segment[1:])
            temp_length = calculate_path_length(graph, temp_path)
            
            if temp_length <= max_length:
                full_path = temp_path
                current_length = temp_length
            else:
                # Truncate to max length
                truncated_path, current_length = truncate_path(graph, temp_path, max_length)
                full_path = truncated_path
    
    return full_path, current_length

def generate_long_paths(graph, edges, num_paths=400, min_length=100.0, max_length=200.0):
    """Generate paths with lengths between min_length and max_length."""
    nodes = list(graph.keys())
    paths = []
    
    # Different path strategies and their weights
    strategies = [
        ("multi_segment", 0.35),
        ("figure_eight", 0.25),
        ("spiral", 0.2),
        ("circular", 0.1),
        ("back_and_forth", 0.1)
    ]
    
    for i in range(num_paths):
        # Choose a strategy based on weights
        strategy = random.choices([s[0] for s in strategies], 
                                 weights=[s[1] for s in strategies], 
                                 k=1)[0]
        
        path = None
        length = 0
        
        # Try to generate a path with the selected strategy
        if strategy == "multi_segment":
            path, length = create_multi_segment_path(graph, edges, min_length, max_length)
            
        elif strategy == "figure_eight":
            path, length = create_figure_eight(graph, edges, min_length, max_length)
            if not path:  # Fallback if figure-eight fails
                path, length = create_multi_segment_path(graph, edges, min_length, max_length)
                
        elif strategy == "spiral":
            # Choose a central node for the spiral
            center_node = random.choice(nodes)
            path, length = create_spiral_path(graph, edges, center_node, min_length, max_length)
            
        elif strategy == "circular":
            # Create a path that returns to its starting point
            start_node = random.choice(nodes)
            
            # Find a sequence of distant nodes
            distant_node = find_distant_node(graph, start_node, percentage=0.7)
            
            # Create a path that goes from start to distant and back
            outward = a_star_search(graph, edges, start_node, distant_node)
            if outward and len(outward) > 1:
                # For the return, we'll find a different path if possible
                return_path = a_star_search(graph, edges, distant_node, start_node)
                
                if return_path and len(return_path) > 1:
                    full_path = outward + return_path[1:]
                    path_length = calculate_path_length(graph, full_path)
                    
                    # Check max length
                    if path_length <= max_length:
                        path = full_path
                        length = path_length
                    elif path_length > max_length and path_length >= min_length:
                        # Truncate to max length
                        path, length = truncate_path(graph, full_path, max_length)
            
            # Fallback
            if not path or length < min_length:
                path, length = create_multi_segment_path(graph, edges, min_length, max_length)
                
        elif strategy == "back_and_forth":
            # Create a path that goes back and forth multiple times
            start_node = random.choice(nodes)
            
            # Find a distant node
            distant_node = find_distant_node(graph, start_node)
            
            # Create the initial path
            initial_path = a_star_search(graph, edges, start_node, distant_node)
            
            if initial_path and len(initial_path) > 1:
                # Create multiple back-and-forth segments
                full_path = initial_path.copy()
                single_segment_length = calculate_path_length(graph, initial_path)
                
                # Calculate how many segments we need
                segments_needed = math.ceil(min_length / single_segment_length)
                
                # Estimate if we'll exceed max_length
                estimated_length = single_segment_length * segments_needed
                if estimated_length > max_length:
                    # Reduce segments to fit max_length
                    segments_needed = math.floor(max_length / single_segment_length)
                
                # Add back-and-forth segments
                current_length = calculate_path_length(graph, full_path)
                for _ in range(segments_needed - 1):
                    # Reverse the path (excluding the endpoints to avoid duplicates)
                    reverse_segment = initial_path[-2:0:-1]
                    
                    # Check if adding will exceed max_length
                    temp_path = full_path.copy()
                    temp_path.extend(reverse_segment)
                    temp_length = calculate_path_length(graph, temp_path)
                    
                    if temp_length > max_length:
                        break
                        
                    full_path = temp_path
                    current_length = temp_length
                    
                    # Add the forward path again (excluding the first node)
                    temp_path = full_path.copy()
                    temp_path.extend(initial_path[1:])
                    temp_length = calculate_path_length(graph, temp_path)
                    
                    if temp_length > max_length:
                        break
                        
                    full_path = temp_path
                    current_length = temp_length
                
                path = full_path
                length = current_length
            
            # Fallback
            if not path or length < min_length:
                path, length = create_multi_segment_path(graph, edges, min_length, max_length)
        
        # If we still couldn't generate a path of sufficient length, try multi-segment with higher iteration count
        if not path or length < min_length * 0.8:  # Accept paths at least 80% of desired length
            path, length = create_multi_segment_path(graph, edges, min_length, max_length, max_segments=15)
        
        # Final check to ensure path is within constraints
        if path and len(path) > 1:
            final_length = calculate_path_length(graph, path)
            
            # Handle edge cases where path is outside constraints
            if final_length > max_length:
                path, final_length = truncate_path(graph, path, max_length)
            
            if final_length >= min_length * 0.8:  # Accept paths at least 80% of min_length
                paths.append({
                    'start': path[0],
                    'goal': path[-1],
                    'path': path,
                    'path_type': strategy,
                    'length': final_length
                })
                
            if i % 10 == 0:
                print(f"Generated {i+1}/{num_paths} paths, latest: {strategy} with length {final_length:.2f}")
    
    return paths

def save_paths_to_json(paths, output_filename):
    with open(output_filename, 'w') as f:
        json.dump(paths, f, indent=2)

if __name__ == "__main__":
    # Load the graph
    graph, edges = load_graph("config/humans_waypoint_graph.json")
    
    # Generate paths with the enhanced algorithm
    min_path_length = 100.0   # Minimum path length
    max_path_length = 500.0  # Maximum path length
    
    paths = generate_long_paths(
        graph, 
        edges, 
        num_paths=1000,
        min_length=min_path_length,
        max_length=max_path_length
    )
    
    # Save the paths to JSON
    save_paths_to_json(paths, "config/paths.json")
    
    # Print statistics
    path_types = {}
    path_lengths = []
    
    for path in paths:
        path_type = path.get('path_type', 'unknown')
        path_types[path_type] = path_types.get(path_type, 0) + 1
        path_lengths.append(path.get('length', 0))
    
    print(f"\nGenerated {len(paths)} paths:")
    for path_type, count in path_types.items():
        print(f"- {path_type}: {count} paths")
    
    if path_lengths:
        avg_length = sum(path_lengths) / len(path_lengths)
        min_length = min(path_lengths)
        max_length = max(path_lengths)
        
        # Count paths that meet length constraints
        paths_in_range = sum(1 for length in path_lengths if min_path_length <= length <= max_path_length)
        percentage = (paths_in_range / len(paths)) * 100
        
        print(f"\nPath length statistics:")
        print(f"- Minimum: {min_length:.2f}")
        print(f"- Average: {avg_length:.2f}")
        print(f"- Maximum: {max_length:.2f}")
        print(f"- Paths within range ({min_path_length}-{max_path_length}): {paths_in_range}/{len(paths)} ({percentage:.1f}%)")