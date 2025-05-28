#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import time

class MissionCompletionHandler(Node):

    def __init__(self):
        super().__init__('mission_completion_handler')
        
        self.status_sub = self.create_subscription(
            String,
            'path_execution_status',
            self.status_callback,
            10
        )
        
        self.mission_sub = self.create_subscription(
            String,
            'mission_complete',
            self.mission_complete_callback,
            10
        )
        
        self.get_logger().info('Mission completion handler started')
        
    def status_callback(self, msg):

        status_text = msg.data.lower()
        
        if "path execution completed" in status_text or "completed successfully" in status_text:
            self.get_logger().info('Final goal reached. Mission completed successfully!')
            self.exit_after_delay(2.0)
            
        elif "failed" in status_text or "navigation cancelled" in status_text:
            if "at waypoint" in status_text and "failed at waypoint" not in status_text:
                return
                
            self.get_logger().error('Mission failed. Initiating shutdown.')
            self.exit_after_delay(2.0)
    
    def mission_complete_callback(self, msg):

        self.get_logger().info(f'Received mission complete notification: {msg.data}')
        self.exit_after_delay(2.0)
    
    def exit_after_delay(self, delay_seconds):

        self.get_logger().info(f'Mission complete - exiting in {delay_seconds} seconds')
        time.sleep(delay_seconds)
        sys.exit(0)  

def main(args=None):
    rclpy.init(args=args)
    
    node = MissionCompletionHandler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()