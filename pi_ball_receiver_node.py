import socket
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005

class BallReceiver(Node):
    def __init__(self):
        super().__init__('ball_receiver')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.1, self.update)

        self.latest_ball = None

    def update(self):
        # Receive UDP messages
        try:
            data, _ = self.sock.recvfrom(1024)
            self.latest_ball = json.loads(data)
        except BlockingIOError:
            pass  # No data this tick

        if self.latest_ball:
            x = self.latest_ball['x']
            z = self.latest_ball['z']

            msg = Twist()

            # Simple proportional control
            # Turn toward the ball
            msg.angular.z = -0.0025 * x  # Adjust multiplier if too slow/fast

            # Move forward if ball is far enough
            if z > 300:  # Stop distance ~30cm
                msg.linear.x = min(0.2, 0.001*(z - 300))
            else:
                msg.linear.x = 0.0

            self.pub.publish(msg)
        else:
            # Stop if no ball detected
            self.pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = BallReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

