import rclpy
from rclpy.node import Node
# Changed from TwistStamped to Twist
from geometry_msgs.msg import Twist 

class DriveRobot(Node):
    def __init__(self):
        super().__init__('drive_robot')
        # Changed message type to Twist
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.send_velocity)
        self.get_logger().info('Robot driver node has started!')

    def send_velocity(self):
        msg = Twist()
        # Twist does not have a header, so we set linear/angular directly
        msg.linear.x = 0.1  # forward speed (m/s)
        msg.angular.z = 0.0 # turning speed (rad/s)
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = DriveRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Stop the robot before shutting down
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
        node.get_logger().info('Stopping robot...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()