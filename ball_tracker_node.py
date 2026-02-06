import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from turtlebot_controller.golf_ball_tracker import GolfBallTracker
import threading
import time

class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_node')

        # Publisher for ball coordinates
        self.ball_pub = self.create_publisher(PointStamped, 'ball_location', 10)

        # Start the tracker in a thread
        self.tracker = GolfBallTracker(show_window=False)
        threading.Thread(target=self.tracker.start, daemon=True).start()

        # Timer to periodically publish the location
        self.timer = self.create_timer(0.1, self.publish_location)

    def publish_location(self):
        loc = self.tracker.get_latest_location()
        if loc:
            x_off, y_off, z = loc
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = x_off / 1000.0  # mm → meters
            msg.point.y = y_off / 1000.0
            msg.point.z = z / 1000.0
            self.ball_pub.publish(msg)

def main():
    rclpy.init()
    node = BallTrackerNode()
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
