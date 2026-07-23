import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ShoeDetector(Node):
    def __init__(self) -> None:
        super().__init__("shoe_detector")

        self.subscription = self.create_subscription(
            LaserScan,
            "/tim781/scan",
            self.scan_callback,
            10,
        )

        self.last_log_time = self.get_clock().now()

        self.get_logger().info(
            "TIM781 index checker started. "
            "신발이 없을 때와 지나갈 때의 index/distance를 확인하세요."
        )

    def scan_callback(self, msg: LaserScan) -> None:
        valid_points = []

        for index, distance in enumerate(msg.ranges):
            distance = float(distance)

            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            if distance < msg.range_min:
                continue

            if distance > msg.range_max:
                continue

            valid_points.append((index, distance))

        if not valid_points:
            return

        nearest_index, nearest_distance = min(
            valid_points,
            key=lambda point: point[1],
        )

        angle_radian = (
            msg.angle_min
            + nearest_index * msg.angle_increment
        )

        angle_degree = math.degrees(angle_radian)

        # 너무 빠르게 출력되지 않도록 0.2초마다 한 번 로그 출력
        current_time = self.get_clock().now()
        elapsed = (
            current_time - self.last_log_time
        ).nanoseconds / 1_000_000_000.0

        if elapsed >= 0.2:
            self.get_logger().info(
                f"nearest_index={nearest_index}, "
                f"distance={nearest_distance:.3f} m, "
                f"angle={angle_degree:.1f} deg, "
                f"total_ranges={len(msg.ranges)}"
            )

            self.last_log_time = current_time


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ShoeDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()