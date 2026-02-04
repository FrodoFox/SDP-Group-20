import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class DriveRobot(Node):
	def __init__(self):
		super().__init__('drive_robot')
		self.publisher_ = self.create_publisher(TwistStamped, 'cmd_vel', 10)
		self.timer = self.create_timer(0.1, self.send_velocity)

	def send_velocity(self):
		msg = TwistStamped()
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.twist.linear.x = 0.1 # forward speed
		msg.twist.angular.z = 0.0 # turning speed
		self.publisher_.publish(msg)

def main():
	rclpy.init()
	node = DriveRobot()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
