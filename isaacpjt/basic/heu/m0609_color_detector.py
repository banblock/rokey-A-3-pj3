#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.subscription = self.create_subscription(Image, '/rgb', self.listener_callback, 10)
        self.publisher = self.create_publisher(Int32, '/color_id', 10)
        self.bridge = CvBridge()
        self.get_logger().info('Color detector node started')

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}')
            return

        color_id = self.detect_color(cv_image)
        id_msg = Int32()
        id_msg.data = color_id
        self.publisher.publish(id_msg)

    def detect_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        blue_mask = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
        green_mask = cv2.inRange(hsv, (35, 60, 60), (85, 255, 255))

        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        blue_pixels = int(np.count_nonzero(blue_mask))
        green_pixels = int(np.count_nonzero(green_mask))
        total_pixels = blue_pixels + green_pixels

        if total_pixels < 500:
            return 0

        if blue_pixels > green_pixels and blue_pixels / max(total_pixels, 1) > 0.4:
            return 1
        if green_pixels > blue_pixels and green_pixels / max(total_pixels, 1) > 0.4:
            return 2
        return 0


def main():
    rclpy.init()
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
