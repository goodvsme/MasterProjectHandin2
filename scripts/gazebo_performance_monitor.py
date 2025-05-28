#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import psutil
import numpy as np
import time
import csv
import os
import subprocess
import re
from std_msgs.msg import Float32

class GazeboPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('gazebo_performance_monitor')
        
        self.declare_parameter('log_directory', 'logs')
        self.declare_parameter('publish_stats', False)
        self.declare_parameter('stats_window', 5)
        self.declare_parameter('sample_rate', 1.0)
        self.declare_parameter('monitor_memory', True)
        self.declare_parameter('log_to_console', False)
        
        self.log_directory = self.get_parameter('log_directory').value
        self.publish_stats = self.get_parameter('publish_stats').value
        self.stats_window = self.get_parameter('stats_window').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.monitor_memory = self.get_parameter('monitor_memory').value
        self.log_to_console = self.get_parameter('log_to_console').value
        
        from ament_index_python.packages import get_package_share_directory
        package_dir = get_package_share_directory('master_project2')
        self.log_path = os.path.join(package_dir, self.log_directory)
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_file = os.path.join(self.log_path, f"gazebo_performance_{timestamp}.csv")
        
        self.get_logger().info(f"Log directory set to: {self.log_path}")
        self.get_logger().info(f"Output file will be: {self.output_file}")
        
        self.rtf_values = []
        self.cpu_values = []
        self.memory_percent_values = []
        
        if self.publish_stats:
            self.rtf_stats_pub = {
                'mean': self.create_publisher(Float32, 'rtf/mean', 10),
                'max': self.create_publisher(Float32, 'rtf/max', 10),
                'min': self.create_publisher(Float32, 'rtf/min', 10),
                'median': self.create_publisher(Float32, 'rtf/median', 10),
                'std_dev': self.create_publisher(Float32, 'rtf/std_dev', 10),
                'variance': self.create_publisher(Float32, 'rtf/variance', 10)
            }
            self.cpu_stats_pub = {
                'mean': self.create_publisher(Float32, 'cpu/mean', 10),
                'max': self.create_publisher(Float32, 'cpu/max', 10),
                'min': self.create_publisher(Float32, 'cpu/min', 10),
                'median': self.create_publisher(Float32, 'cpu/median', 10),
                'std_dev': self.create_publisher(Float32, 'cpu/std_dev', 10),
                'variance': self.create_publisher(Float32, 'cpu/variance', 10)
            }
            if self.monitor_memory:
                self.memory_stats_pub = {
                    'mean': self.create_publisher(Float32, 'memory/mean', 10),
                    'max': self.create_publisher(Float32, 'memory/max', 10),
                    'min': self.create_publisher(Float32, 'memory/min', 10),
                    'median': self.create_publisher(Float32, 'memory/median', 10),
                    'std_dev': self.create_publisher(Float32, 'memory/std_dev', 10),
                    'variance': self.create_publisher(Float32, 'memory/variance', 10)
                }
        
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            if self.monitor_memory:
                writer.writerow(['time', 'rtf', 'cpu_percent', 'memory_percent'])
            else:
                writer.writerow(['time', 'rtf', 'cpu_percent'])
        
        timer_period = 1.0 / self.sample_rate
        self.timer = self.create_timer(timer_period, self.monitor_callback)
        
        self.get_logger().info(f"Gazebo Performance Monitor started. Saving data to {self.output_file}")
        
        try:
            subprocess.run(["gz", "stats", "-p"], capture_output=True, text=True, check=False)
            self.get_logger().info("Gazebo gz stats command is available")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            self.get_logger().error(f"Gazebo gz stats command not working correctly: {str(e)}")
            self.get_logger().error("Make sure Gazebo is running correctly")
    
    def get_rtf_from_gz_stats(self):
        try:
            if self.log_to_console:
                self.get_logger().info("Running 'gz stats -p' command...")
            
            result = subprocess.run(
                ["gz", "stats", "-p"],
                capture_output=True,
                text=True,
                timeout=1.0
            )
            
            self.get_logger().info(f"gz stats return code: {result.returncode}")
            self.get_logger().info(f"gz stats stdout: '{result.stdout}'")
            if result.stderr:
                self.get_logger().info(f"gz stats stderr: '{result.stderr}'")
            
            match = re.search(r"Factor\[([0-9.]+)\]", result.stdout)
            if match:
                rtf = float(match.group(1))
                self.get_logger().info(f"Successfully extracted RTF value: {rtf}")
                return rtf
            else:
                self.get_logger().warn("Could not parse RTF value from gz stats output")
                return None
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            self.get_logger().error(f"Error getting RTF from gz stats: {str(e)}")
            return None
    
    def monitor_callback(self):
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            rtf = self.get_rtf_from_gz_stats()
            
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 10 == 0:
                self.get_logger().info(f"Monitor cycle {self.debug_counter}: RTF = {rtf}, Have values: {len(self.rtf_values)}")
            
            if rtf is not None:
                rtf = max(0.01, min(rtf, 100.0))
                self.rtf_values.append(rtf)
                
                if len(self.rtf_values) > self.stats_window:
                    self.rtf_values.pop(0)
                    
                self.get_logger().info(f"Current RTF: {rtf:.2f}")
            else:
                self.get_logger().warn("Failed to get RTF from gz stats")
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_values.append(cpu_percent)
            
            if len(self.cpu_values) > self.stats_window:
                self.cpu_values.pop(0)
                
            memory_percent = None
            if self.monitor_memory:
                memory_percent = psutil.virtual_memory().percent
                self.memory_percent_values.append(memory_percent)
                
                if len(self.memory_percent_values) > self.stats_window:
                    self.memory_percent_values.pop(0)
            
            if self.rtf_values:
                try:
                    self.get_logger().info(f"Writing to file: {self.output_file}")
                    
                    with open(self.output_file, 'a') as f:
                        writer = csv.writer(f)
                        if self.monitor_memory:
                            writer.writerow([
                                current_time, 
                                self.rtf_values[-1], 
                                cpu_percent, 
                                memory_percent if memory_percent is not None else 0
                            ])
                        else:
                            writer.writerow([
                                current_time, 
                                self.rtf_values[-1], 
                                cpu_percent
                            ])
                        
                    self.get_logger().info(f"Successfully wrote data to file")
                except Exception as e:
                    self.get_logger().error(f"Error writing to file: {str(e)}")
            else:
                self.get_logger().warn("No RTF values available, skipping file write")
            
            self.calculate_and_publish_stats()
            
        except Exception as e:
            self.get_logger().error(f"Error in monitor_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def get_memory_usage(self):
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used,
            'free': mem.free
        }
    
    def calculate_and_publish_stats(self):
        if not self.publish_stats:
            return
            
        if not self.rtf_values or not self.cpu_values:
            return
        
        rtf_array = np.array(self.rtf_values)
        rtf_stats = {
            'mean': np.mean(rtf_array),
            'max': np.max(rtf_array),
            'min': np.min(rtf_array),
            'median': np.median(rtf_array),
            'std_dev': np.std(rtf_array),
            'variance': np.var(rtf_array)
        }
        
        cpu_array = np.array(self.cpu_values)
        cpu_stats = {
            'mean': np.mean(cpu_array),
            'max': np.max(cpu_array),
            'min': np.min(cpu_array),
            'median': np.median(cpu_array),
            'std_dev': np.std(cpu_array),
            'variance': np.var(cpu_array)
        }
        
        if self.monitor_memory and self.memory_percent_values:
            memory_array = np.array(self.memory_percent_values)
            memory_stats = {
                'mean': np.mean(memory_array),
                'max': np.max(memory_array),
                'min': np.min(memory_array),
                'median': np.median(memory_array),
                'std_dev': np.std(memory_array),
                'variance': np.var(memory_array)
            }
        
        if self.log_to_console:
            self.get_logger().info(
                f"RTF - Mean: {rtf_stats['mean']:.2f}, Max: {rtf_stats['max']:.2f}, "
                f"Min: {rtf_stats['min']:.2f}, Median: {rtf_stats['median']:.2f}, "
                f"StdDev: {rtf_stats['std_dev']:.2f}"
            )
            
            self.get_logger().info(
                f"CPU - Mean: {cpu_stats['mean']:.2f}%, Max: {cpu_stats['max']:.2f}%, "
                f"Min: {cpu_stats['min']:.2f}%, Median: {cpu_stats['median']:.2f}%, "
                f"StdDev: {cpu_stats['std_dev']:.2f}%"
            )
            
            if self.monitor_memory and self.memory_percent_values:
                self.get_logger().info(
                    f"Memory - Mean: {memory_stats['mean']:.2f}%, Max: {memory_stats['max']:.2f}%, "
                    f"Min: {memory_stats['min']:.2f}%, Median: {memory_stats['median']:.2f}%, "
                    f"StdDev: {memory_stats['std_dev']:.2f}%"
                )
        
        if self.publish_stats:
            for stat_name, value in rtf_stats.items():
                msg = Float32()
                msg.data = float(value)
                self.rtf_stats_pub[stat_name].publish(msg)
            
            for stat_name, value in cpu_stats.items():
                msg = Float32()
                msg.data = float(value)
                self.cpu_stats_pub[stat_name].publish(msg)
                
            if self.monitor_memory and self.memory_percent_values:
                for stat_name, value in memory_stats.items():
                    msg = Float32()
                    msg.data = float(value)
                    self.memory_stats_pub[stat_name].publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    print("\n===== Gazebo Performance Monitor =====")
    print("DEBUG MODE: Enabling verbose logging")
    print("This monitor uses the 'gz stats -p' command to get RTF information.")
    print("Ensure Gazebo is running for this to work correctly.")
    print("============================================\n")
    
    monitor = GazeboPerformanceMonitor()
    monitor.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
    
    print("Testing gz stats command...")
    try:
        result = subprocess.run(["gz", "stats", "-p"], capture_output=True, text=True, check=False)
        print(f"Test result: {result.returncode}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
            
        if "Factor" in result.stdout:
            print("SUCCESS: gz stats is working correctly!")
        else:
            print("WARNING: gz stats ran but didn't return expected format")
    except Exception as e:
        print(f"ERROR running gz stats: {str(e)}")
    
    try:
        print("Starting node spin...")
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("Cleaning up...")
        monitor.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete")

if __name__ == '__main__':
    main()