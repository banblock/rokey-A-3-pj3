#!/usr/bin/env python3
"""
카메라 1, 2번 토픽 구독해서 PNG로 저장.
한 장씩만 받고 종료.

실행:
  python3 capture_cameras.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime


class CameraCapture(Node):

    def __init__(self):
        super().__init__('camera_capture')
        self.bridge = CvBridge()
        self.saved = {'cam1': False, 'cam2': False, 'cam3': False}

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image, '/d455_1/color/image_raw',
            lambda msg: self.save_image(msg, 'cam1'), image_qos,
        )
        self.create_subscription(
            Image, '/d455_2/color/image_raw',
            lambda msg: self.save_image(msg, 'cam2'), image_qos,
        )
        self.create_subscription(
            Image, '/d455_3/color/image_raw',
            lambda msg: self.save_image(msg, 'cam3'), image_qos,
        )
        self.get_logger().info('Waiting for camera images...')

    def save_image(self, msg, cam_name):
        if self.saved[cam_name]:
            return

        try:
            if cam_name == 'cam3':
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                path = f'/home/rokey/dataset_bottom/{cam_name}_{current_time}_captured.png'
                cv2.imwrite(path, cv_image)
                self.saved[cam_name] = True
                self.get_logger().info(f'{cam_name} saved: {path}')
            else:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                path = f'/home/rokey/dataset2/{cam_name}_{current_time}_captured.png'
                cv2.imwrite(path, cv_image)
                self.saved[cam_name] = True
                self.get_logger().info(f'{cam_name} saved: {path}')
        except Exception as e:
            self.get_logger().error(f'{cam_name} failed: {e}')

        if all(self.saved.values()):
            self.get_logger().info('Both cameras captured. Done.')
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = CameraCapture()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()