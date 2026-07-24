#!/usr/bin/env python3
"""메인 컨트롤 노드가 아직 없을 때, FMS의 PickupList 토픽을 대신 발행해주는 임시 스텁.

일정 주기마다 무작위 종류(A~D) + 무작위 길이 5개짜리 배치를 만들어 /control/pickup로
보낸다. 실제 메인 컨트롤 노드가 생기면 이 파일은 삭제(launch 파일에서도 빼면 됨).
"""

import random

import rclpy
from rclpy.node import Node

from recycle_interfaces.msg import PickupList

AUTO_TRIGGER_INTERVAL_SEC = 3.0
# 소/중/대 경계(SIZE_THRESHOLDS_MM=(255,275))에 걸치는 실측 사이즈 그대로 사용
SHOE_LENGTH_CHOICES_MM = [240, 260, 280]
SHOE_TYPE_LABELS = ["A", "B", "C", "D"]
SHOES_NUM_LIST = [0]  # 이 목록 길이(2개)만큼만 보내고 끝낸다 — 무한 반복 아님

class MainControlStub(Node):

    def __init__(self):
        super().__init__("main_control_stub")
        self.pub = self.create_publisher(PickupList, "/control/pickup", 10)
        # itertools.cycle()을 쓰면 리스트를 끝없이 반복해서 "몇 번만 보내고
        # 끝"이 안 된다 — SHOES_NUM_LIST 길이(2개)만큼만 next()가 되고 그 뒤론
        # StopIteration이 나는 1회성 iter()를 그대로 쓴다.
        self._shoes_num_iter = iter(SHOES_NUM_LIST)
        self._timer = self.create_timer(AUTO_TRIGGER_INTERVAL_SEC, self._send_random_batch)
        self.get_logger().info(
            f"메인 컨트롤 스텁 시작 — {AUTO_TRIGGER_INTERVAL_SEC:.0f}초마다 "
            f"{SHOES_NUM_LIST} 순서대로 총 {len(SHOES_NUM_LIST)}번만 /control/pickup로 전송"
        )

    def _send_random_batch(self):
        # 토픽이라 서비스처럼 "준비됐는지" 확인·응답 대기가 없다 — 대신 구독자가
        # 아직 하나도 안 붙었으면(FMS가 아직 안 떴거나 디스커버리 중) 발행해도
        # 그냥 유실되니, 있을 때까지 iterator를 소모하지 않고 기다린다.
        if self.pub.get_subscription_count() == 0:
            self.get_logger().warn("/control/pickup 구독자가 아직 없어 이번 주기는 건너뜀")
            return
        try:
            shoes_num = next(self._shoes_num_iter)
        except StopIteration:
            self.get_logger().info(f"{len(SHOES_NUM_LIST)}번 전송 완료 — 더 이상 보내지 않음")
            self._timer.cancel()  # 타이머를 꺼서 콜백이 더 이상 호출되지 않게 한다
            return
        msg = PickupList()
        msg.place = shoes_num
        msg.shoes = [280]
        # msg.place = random.randint(0, len(SHOE_TYPE_LABELS) - 1)
        # msg.shoes = [random.choice(SHOE_LENGTH_CHOICES_MM) for _ in range(5)]
        self.pub.publish(msg)
        self.get_logger().info(
            f"[전송] 종류={SHOE_TYPE_LABELS[msg.place]} 길이={list(msg.shoes)}"
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
