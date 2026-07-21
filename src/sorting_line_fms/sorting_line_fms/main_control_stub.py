#!/usr/bin/env python3
"""메인 컨트롤 노드가 아직 없을 때, FMS의 ShoesList 서비스를 대신 호출해주는 임시 스텁.

일정 주기마다 무작위 종류(A~D) + 무작위 길이 5개짜리 배치를 만들어 /fms/shoes_list로
보낸다. 실제 메인 컨트롤 노드가 생기면 이 파일은 삭제(launch 파일에서도 빼면 됨).
"""

import itertools
import random

import rclpy
from rclpy.node import Node

from sorting_line_interfaces.srv import ShoesList

AUTO_TRIGGER_INTERVAL_SEC = 3.0
# 소/중/대 경계(SIZE_THRESHOLDS_MM=(255,275))에 걸치는 실측 사이즈 그대로 사용
SHOE_LENGTH_CHOICES_MM = [240, 260, 280]
SHOE_TYPE_LABELS = ["A", "B", "C", "D"]
SHOES_NUM_LIST = [0, 0]
# 매 호출마다 iter()로 새로 만들면 항상 처음부터 다시 시작해버려서 순회가 안 된다
# — 모듈 로드 시 한 번만 만든 무한 순환 이터레이터를 계속 재사용해야 한다.
_SHOES_NUM_CYCLE = itertools.cycle(SHOES_NUM_LIST)

class MainControlStub(Node):

    def __init__(self):
        super().__init__("main_control_stub")
        self.client = self.create_client(ShoesList, "/fms/shoes_list")
        self.create_timer(AUTO_TRIGGER_INTERVAL_SEC, self._send_random_batch)
        self.get_logger().info(
            f"메인 컨트롤 스텁 시작 — {AUTO_TRIGGER_INTERVAL_SEC:.0f}초마다 무작위 배치를 "
            f"/fms/shoes_list로 전송"
        )

    def _send_random_batch(self):
        if not self.client.service_is_ready():
            self.get_logger().warn("/fms/shoes_list 서비스가 아직 준비되지 않아 이번 주기는 건너뜀")
            return
        request = ShoesList.Request()
        request.shoes_num = next(_SHOES_NUM_CYCLE)
        # request.shoes_num = random.randint(0, len(SHOE_TYPE_LABELS) - 1)
        request.shoes_length = [random.choice(SHOE_LENGTH_CHOICES_MM) for _ in range(5)]
        self.client.call_async(request)
        self.get_logger().info(
            f"[전송] 종류={SHOE_TYPE_LABELS[request.shoes_num]} 길이={request.shoes_length}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MainControlStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
