import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import numpy as np

class DetectionColor(Node):
    def __init__(self):
        super().__init__('detection_color_node')
        self.bridge = CvBridge()
        self.color_id = 0
        self.rgb_subscription = self.create_subscription(
            Image,
            '/rgb',
            self.img_convert,
            10
        )
        self._pub = self.create_publisher(Int32, '/color_id', 10)

    def img_convert(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        # 색상 범위 설정
        mask_blue = cv2.inRange(img_hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
        mask_green = cv2.inRange(img_hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

        blue_pixel_count = cv2.countNonZero(mask_blue)
        green_pixel_count = cv2.countNonZero(mask_green)

        if blue_pixel_count > green_pixel_count:
            self.color_id = 1
        elif green_pixel_count > blue_pixel_count:
            self.color_id = 2
        else:
            self.color_id = 0 # 아무것도 감지 안 됨

        # 색상이 확정된 경우에만 publish
        if self.color_id != 0:
            self._pub.publish(Int32(data=self.color_id))
            self.get_logger().info(f"색상 확정, publish: {self.color_id}")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = DetectionColor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()