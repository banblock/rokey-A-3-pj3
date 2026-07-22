#!/usr/bin/env python3
"""메인 컨트롤 노드가 아직 없을 때, FMS의 ShoesList 서비스를 대신 호출해주는 임시 스텁.

일정 주기마다 무작위 종류(A~D) + 무작위 길이 5개짜리 배치를 만들어 /fms/shoes_list로
보낸다. 실제 메인 컨트롤 노드가 생기면 이 파일은 삭제(launch 파일에서도 빼면 됨).
"""

import random

import rclpy
from rclpy.node import Node

from recycle_interfaces.srv import ShoesList

AUTO_TRIGGER_INTERVAL_SEC = 3.0
# 소/중/대 경계(SIZE_THRESHOLDS_MM=(255,275))에 걸치는 실측 사이즈 그대로 사용
SHOE_LENGTH_CHOICES_MM = [240, 260, 280]
SHOE_TYPE_LABELS = ["A", "B", "C", "D"]
SHOES_NUM_LIST = [0, 0]  # 이 목록 길이(2개)만큼만 보내고 끝낸다 — 무한 반복 아님

class MainControlStub(Node):

    def __init__(self):
        super().__init__("main_control_stub")
        self.client = self.create_client(ShoesList, "/fms/shoes_list")
        # itertools.cycle()을 쓰면 리스트를 끝없이 반복해서 "몇 번만 보내고
        # 끝"이 안 된다 — SHOES_NUM_LIST 길이(2개)만큼만 next()가 되고 그 뒤론
        # StopIteration이 나는 1회성 iter()를 그대로 쓴다.
        self._shoes_num_iter = iter(SHOES_NUM_LIST)
        self._timer = self.create_timer(AUTO_TRIGGER_INTERVAL_SEC, self._send_random_batch)
        self.get_logger().info(
            f"메인 컨트롤 스텁 시작 — {AUTO_TRIGGER_INTERVAL_SEC:.0f}초마다 "
            f"{SHOES_NUM_LIST} 순서대로 총 {len(SHOES_NUM_LIST)}번만 /fms/shoes_list로 전송"
        )

    def _send_random_batch(self):
        if not self.client.service_is_ready():
            self.get_logger().warn("/fms/shoes_list 서비스가 아직 준비되지 않아 이번 주기는 건너뜀")
            return  # 아직 iterator를 소모하지 않았으니 다음 tick에 다시 시도된다
        try:
            shoes_num = next(self._shoes_num_iter)
        except StopIteration:
            self.get_logger().info(f"{len(SHOES_NUM_LIST)}번 전송 완료 — 더 이상 보내지 않음")
            self._timer.cancel()  # 타이머를 꺼서 콜백이 더 이상 호출되지 않게 한다
            return
        request = ShoesList.Request()
        request.shoes_num = shoes_num
        request.shoes_length = [280, 280, 260, 260, 240]
        # request.shoes_num = random.randint(0, len(SHOE_TYPE_LABELS) - 1)
        # request.shoes_length = [random.choice(SHOE_LENGTH_CHOICES_MM) for _ in range(5)]
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
