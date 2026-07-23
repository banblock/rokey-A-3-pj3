import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


START_SPEED = 1.0
SLOW_SPEED = 0.3
REVERSE_SPEED = -0.5
STOP_SPEED = 0.0


class ConveyorController(Node):
    """Isaac Sim 컨베이어 속도를 제어하는 ROS2 노드."""

    def __init__(self) -> None:
        super().__init__("conveyor_controller")

        self.publisher_ = self.create_publisher(
            Float32,
            "/conveyor/velocity",
            10,
        )

        self.current_speed = STOP_SPEED

        self.get_logger().info("Conveyor Controller 시작")

    def set_speed(self, speed: float) -> None:
        message = Float32()
        message.data = float(speed)

        self.publisher_.publish(message)

        # 발행한 메시지가 ROS2 통신에 반영될 시간을 잠깐 확보
        rclpy.spin_once(self, timeout_sec=0.01)

        self.current_speed = float(speed)

        self.get_logger().info(
            f"/conveyor/velocity 발행: {self.current_speed}"
        )

    def run_auto_sequence(self) -> None:

        self.get_logger().info("자동 동작 시작")

        self.set_speed(START_SPEED)
        self.get_logger().info("컨베이어 시작")

        time.sleep(1.55)

        self.set_speed(STOP_SPEED)
        self.get_logger().info("1.55초 경과: 컨베이어 정지")

        time.sleep(20.0)

        self.set_speed(START_SPEED)
        self.get_logger().info("4초 경과: 컨베이어 다시 시작")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConveyorController()

    try:
        # 노드 실행 직후 자동 시퀀스 실행
        node.run_auto_sequence()

    except KeyboardInterrupt:
        node.set_speed(STOP_SPEED)
        print("\n컨베이어를 정지하고 종료합니다.")

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()