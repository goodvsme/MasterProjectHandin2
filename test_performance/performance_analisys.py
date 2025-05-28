#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

# Configure these parameters as needed
CSV_FILE_PATH = "/home/patrickdalager/ros2_ws_master/src/master_project2/test_performance/gazebo_rtf_log_1.csv"
Y_SCALE_FACTOR = 1.0  # Change this value to scale the RTF y-axis labels

def analyze_performance_data(csv_file):
    """
    Analyze and visualize the performance data collected by the gazebo_performance_monitor
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    # Get the directory containing the CSV file
    folder = os.path.dirname(csv_file)
    
    # Create results folder if it doesn't exist
    results_folder = os.path.join(folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Check which columns are available
    has_memory = 'memory_percent' in df.columns
    
    # Print basic statistics
    print("\n===== SUMMARY STATISTICS =====")
    
    # RTF statistics
    rtf_stats = {
        'mean': df['rtf'].mean(),
        'median': df['rtf'].median(),
        'min': df['rtf'].min(),
        'max': df['rtf'].max(),
        'std_dev': df['rtf'].std(),
        'variance': df['rtf'].var()
    }
    
    print("\nReal-Time Factor (RTF) Statistics:")
    for stat, value in rtf_stats.items():
        print(f"  {stat.capitalize()}: {value:.4f}")
    
    # CPU statistics
    cpu_stats = {
        'mean': df['cpu_percent'].mean(),
        'median': df['cpu_percent'].median(),
        'min': df['cpu_percent'].min(),
        'max': df['cpu_percent'].max(),
        'std_dev': df['cpu_percent'].std(),
        'variance': df['cpu_percent'].var()
    }
    
    print("\nCPU Usage Statistics:")
    for stat, value in cpu_stats.items():
        print(f"  {stat.capitalize()}: {value:.4f}%")
    
    # Memory statistics (if available)
    if has_memory:
        memory_stats = {
            'mean': df['memory_percent'].mean(),
            'median': df['memory_percent'].median(),
            'min': df['memory_percent'].min(),
            'max': df['memory_percent'].max(),
            'std_dev': df['memory_percent'].std(),
            'variance': df['memory_percent'].var()
        }
        
        print("\nMemory Usage Statistics:")
        for stat, value in memory_stats.items():
            print(f"  {stat.capitalize()}: {value:.4f}%")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Set up the figure with appropriate number of subplots
    num_plots = 2 + (1 if has_memory else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    
    # RTF plot
    axes[0].plot(df['time'], df['rtf'], 'b-')
    axes[0].set_title('Real-Time Factor over Time')
    
    # Set the y-axis label based on the scaling
    if Y_SCALE_FACTOR == 1.0:
        axes[0].set_ylabel('RTF (World set to 0.5 RTF)')
    else:
        axes[0].set_ylabel(f'RTF × {Y_SCALE_FACTOR:.2f}')
    
    axes[0].grid(True)
    
    # Create a custom formatter for the y-axis tick labels with ALWAYS 2 decimal places
    def scale_rtf_labels(y, pos):
        return f'{y * Y_SCALE_FACTOR:.2f}'
    
    # Apply the custom formatter to the RTF plot y-axis
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(scale_rtf_labels))
    
    # Add a horizontal line at RTF = 1.0
    axes[0].axhline(y=1.0, color='r', linestyle='--')
    
    # Add the text label aligned to the right side of the plot
    axes[0].text(df['time'].iloc[-1], 1.05, f'World setting (RTF=0.5)', color='r', 
                 horizontalalignment='right')
    
    # CPU Usage plot
    axes[1].plot(df['time'], df['cpu_percent'], 'g-')
    axes[1].set_title('CPU Usage over Time')
    axes[1].set_ylabel('CPU (%)')
    axes[1].grid(True)
    
    # Memory Usage plot (if available)
    if has_memory:
        axes[2].plot(df['time'], df['memory_percent'], 'm-')
        axes[2].set_title('Memory Usage over Time')
        axes[2].set_ylabel('Memory (%)')
        axes[2].grid(True)
        axes[2].set_xlabel('Time (s)')
    else:
        axes[1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save the plot to results folder
    base_filename = os.path.basename(csv_file)
    output_plot = os.path.join(results_folder, os.path.splitext(base_filename)[0] + '_analysis.png')
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Show the plot
    plt.show()
    
    # Save summary to a file in results folder
    output_summary = os.path.join(results_folder, os.path.splitext(base_filename)[0] + '_summary.txt')
    with open(output_summary, 'w') as f:
        f.write("===== GAZEBO PERFORMANCE SUMMARY =====\n\n")
        
        f.write("Real-Time Factor (RTF) Statistics:\n")
        for stat, value in rtf_stats.items():
            f.write(f"  {stat.capitalize()}: {value:.4f}\n")
        
        f.write("\nCPU Usage Statistics:\n")
        for stat, value in cpu_stats.items():
            f.write(f"  {stat.capitalize()}: {value:.4f}%\n")
        
        if has_memory:
            f.write("\nMemory Usage Statistics:\n")
            for stat, value in memory_stats.items():
                f.write(f"  {stat.capitalize()}: {value:.4f}%\n")
    
    print(f"Summary saved to {output_summary}")

if __name__ == "__main__":
    # Simply run the analysis with the hardcoded CSV file path
    analyze_performance_data(CSV_FILE_PATH)