#!/usr/bin/env python3

import re
import time
import csv
import os
import psutil
import signal
import sys
import subprocess
import datetime

# Configuration
output_file = "/home/patrickdalager/ros2_ws_master/install/master_project2/share/master_project2/logs/gazebo_rtf_log.csv"
sample_rate = 10.0  # Hz
monitor_memory = True
max_rtf = 100.0
log_to_console = True
stats_window = 100  # Number of samples to keep for rolling statistics
direct_mode = True  # Run gz stats directly in Python instead of using the bash script

# Initialize data storage
rtf_values = []
cpu_values = []
memory_values = []
gz_process = None
start_time = None

def signal_handler(sig, frame):
    """Handle Ctrl+C to exit gracefully"""
    print("\nExiting...")
    if gz_process:
        print("Terminating gz stats process...")
        gz_process.terminate()
    sys.exit(0)

def calculate_statistics(values):
    """Calculate basic statistics for a list of values"""
    if not values:
        return None
        
    stats = {
        'mean': sum(values) / len(values),
        'max': max(values),
        'min': min(values),
        'count': len(values)
    }
    return stats

def main():
    """Main function to monitor Gazebo RTF"""
    global gz_process, start_time
    
    # Set the start time for the relative time column
    start_time = time.time()
    
    print("\n===== Gazebo RTF Log Reader =====")
    print(f"Saving data to: {output_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print("Press Ctrl+C to exit")
    print("========================================\n")
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if monitor_memory:
            writer.writerow(['timestamp', 'time', 'rtf', 'cpu_percent', 'memory_percent'])
        else:
            writer.writerow(['timestamp', 'time', 'rtf', 'cpu_percent'])
    
    # Start gz stats directly in Python if in direct mode
    if direct_mode:
        print("Starting gz stats process directly...")
        gz_process = subprocess.Popen(
            ["gz", "stats"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
    
    # Main monitoring loop
    cycle = 0
    sleep_time = 1.0 / sample_rate
    
    try:
        while True:
            cycle += 1
            cycle_start_time = time.time()
            
            rtf = None
            
            if direct_mode:
                # Read directly from the gz stats process output
                if gz_process.poll() is not None:
                    print("gz stats process has terminated. Exiting...")
                    break
                
                line = gz_process.stdout.readline().strip()
                if line:
                    if log_to_console:
                        print(f"Read line: {line}")
                    
                    # Parse the RTF value
                    match = re.search(r"Factor\[([0-9.]+)\]", line)
                    if match:
                        rtf = float(match.group(1))
                        # Apply reasonable bounds to RTF value
                        rtf = max(0.01, min(rtf, max_rtf))
                        if log_to_console:
                            print(f"Current RTF: {rtf:.2f}")
            
            if rtf is not None:
                rtf_values.append(rtf)
                
                # Keep only the most recent values for statistics
                if len(rtf_values) > stats_window:
                    rtf_values.pop(0)
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.01)  # Quick check
            cpu_values.append(cpu_percent)
            if len(cpu_values) > stats_window:
                cpu_values.pop(0)
            
            # Get memory usage if enabled
            memory_percent = None
            if monitor_memory:
                memory_percent = psutil.virtual_memory().percent
                memory_values.append(memory_percent)
                if len(memory_values) > stats_window:
                    memory_values.pop(0)
            
            # Save to file if we have RTF values
            if rtf is not None:
                # Generate timestamp and elapsed time
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_time = time.time() - start_time
                
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if monitor_memory:
                        writer.writerow([timestamp, elapsed_time, rtf, cpu_percent, memory_percent])
                    else:
                        writer.writerow([timestamp, elapsed_time, rtf, cpu_percent])
                
                # Print status every 10 cycles
                if cycle % 10 == 0:
                    print(f"\nCycle {cycle}:")
                    if len(rtf_values) > 0:
                        rtf_stats = calculate_statistics(rtf_values)
                        print(f"RTF: current={rtf_values[-1]:.2f}, mean={rtf_stats['mean']:.2f}, min={rtf_stats['min']:.2f}, max={rtf_stats['max']:.2f}")
                    
                    if len(cpu_values) > 0:
                        cpu_stats = calculate_statistics(cpu_values)
                        print(f"CPU: current={cpu_percent:.1f}%, mean={cpu_stats['mean']:.1f}%, min={cpu_stats['min']:.1f}%, max={cpu_stats['max']:.1f}%")
                    
                    if monitor_memory and len(memory_values) > 0:
                        memory_stats = calculate_statistics(memory_values)
                        print(f"Memory: current={memory_percent:.1f}%, mean={memory_stats['mean']:.1f}%, min={memory_stats['min']:.1f}%, max={memory_stats['max']:.1f}%")
            
            # Sleep to maintain sample rate
            elapsed = time.time() - cycle_start_time
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)
            else:
                if cycle % 20 == 0:  # Only show this warning occasionally
                    print(f"Warning: Processing took {elapsed:.3f}s, longer than the desired {sleep_time:.3f}s cycle time")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        if gz_process:
            print("Terminating gz stats process...")
            gz_process.terminate()
    
    print("Monitor stopped")

if __name__ == "__main__":
    main()