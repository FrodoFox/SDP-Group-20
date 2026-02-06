import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from turtlebot_controller.golf_ball_tracker import GolfBallTracker
import threading
import time

class WandererWithCamera(Node):
    def __init__(self):
        super().__init__('wanderer_with_camera')

        # Publisher to move the robot
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # Start the ball tracker (from the Pi camera code)
        self.tracker = GolfBallTracker(show_window=False)
        threading.Thread(target=self.tracker.start, daemon=True).start()

        # Timer to check ball location and move robot
        self.timer = self.create_timer(0.1, self.update_motion)
    def update_motion(self):
        msg = TwistStamped()
        location = self.tracker.get_latest_location()

        if location:
            x_off, y_off, z = location
            if z <= 300:  # stop when within 30 cm
                msg.twist.linear.x = 0.0
                msg.twist.angular.z = 0.0
            else:
                msg.twist.linear.x = min(0.2, 0.001*(z-300))
                msg.twist.angular.z = max(min(-0.0025*x_off, 1.0), -1.0)
        else:
            # If no ball detected, just spin slowly to search
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.5

        self.cmd_pub.publish(msg)
def main():
    rclpy.init()
    node = WandererWithCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tracker.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
