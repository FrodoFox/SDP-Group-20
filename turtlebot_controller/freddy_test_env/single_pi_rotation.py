import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped 
import math
import time
import sys
from GolfBallDetection import GolfBallTracker

class BallFollowerNode(Node):
    def __init__(self, tracker):
        super().__init__('ball_follower_node')
        
        # Initialising an instance of the tracker directly inside the turtlebot node
        self.tracker = tracker
        
        # Creating a publisher for turtlebot rotation
        self.rotation_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        # Camera constants (required to recompute movements)
        self.FOCAL_LENGTH = 1675 

        # Value to simply indicate offset (positive if pointing to the left of ball and negative for if it's to the right of the ball)
        self.CAMERA_OFFSET_DEG = 0.0 

    def run_logic(self):
        
        # Simply initialising the message to the hardware and logging the systems start
        msg = TwistStamped()
        self.get_logger().info("Single-Pi Vision System Active. Tracking ball...")

        # While the ROS2 system is running (and happy)
        while rclpy.ok():
            
            # Simply grabbing latest location
            location = self.tracker.get_latest_location()

            # If a location was being tracked or has been being tracked over the last 15 frames
            if location:
                x_mm, y_mm, z_mm = location
                
                # Converting the recieved x into an angle offset
                angle_rad = math.atan2(x_mm, self.FOCAL_LENGTH)
                error_deg = math.degrees(angle_rad) + self.CAMERA_OFFSET_DEG

                # Checking the error of the detected ball is within 5 degrees of where the tutlebot is poitning
                if abs(error_deg) < 5.0:
                    self.get_logger().info(f"LOCKED: {error_deg:.1f}°")
                    msg.twist.angular.z = 0.0

                # Turning the robot (right if error is positive and left if the error is negative)
                else:
                    speed = 0.35 if abs(error_deg) > 15 else 0.2                # Changing speed depending on distance from the middle of the camera
                    msg.twist.angular.z = -speed if error_deg > 0 else speed    # Turning the robot (z for some reason is the rotation)
                    
                # Building and publishing the ROS2 message to be interpreted by the turtlebot
                msg.header.stamp = self.get_clock().now().to_msg()  # Attaches a timestamp argument to the metadata to mitigate against network lag issues
                msg.header.frame_id = 'base_link'                   # base_link meaning the reference point of where the turtlebot currently is
                self.rotation_publisher.publish(msg)                # The actual message to rotate the robot towards the ball
                
                # Readable output for debugging
                sys.stdout.write(f'\rTarget: {error_deg:>7.2f}° | Z-Dist: {z_mm:>7.2f}mm')
                sys.stdout.flush()

            else:
                # If the tracker returns None as a ball isn't being tracked then it stops the motors immediately
                stop_msg = TwistStamped()
                stop_msg.header.stamp = self.get_clock().now().to_msg()
                self.rotation_publisher.publish(stop_msg)               # By default when given no other parameters the message initialises to 0 (stop)

                # Terminal output to indicate ball is being searched for (and not currently detected)
                sys.stdout.write('\rSearching for golf ball...                     ')
                sys.stdout.flush()

            # Running the loop every 50ms to try and not make it judder with the movement of the camera and turtle
            time.sleep(0.05)

def main():
    # Initialising ROS2
    rclpy.init()

    # Starting the vision tracker (runs on a thread by default so no need to thread it again)
    tracker = GolfBallTracker(show_window=True)
    print("Initializing Camera...")
    tracker.start()
    
    # Create the node and giving it access to the tracker
    node = BallFollowerNode(tracker)
    
    # Starting the node as well as handling keyboard interrupts and shutting everything down properly on completion
    try:
        node.run_logic()
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
    finally:
        stop_msg = TwistStamped()
        node.rotation_publisher.publish(stop_msg)
        tracker.stop()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()