#!/usr/bin/env python3
"""비전 노드가 아직 없을 때 쓰는 임시 대체 노드 — 실제 비전 파이프라인이 생기면 이 파일은 지울 것.

- /vision/classify_shoes (std_srvs/Trigger) 서비스를 호스팅한다.
  fms_node.py가 기대하는 형식대로, 실제 데이터(종류/길이)는 response.message에
  JSON 문자열로 실어보낸다: {"shoe_type": 0~3, "shoe_length_mm": [int]*5}
- /vision/shoe_ready(std_msgs/Empty)를 기본적으로 AUTO_TRIGGER_INTERVAL_SEC마다
  자동 발행한다. 수동으로 원하는 타이밍에만 트리거하고 싶으면 아래 상수를
  None으로 바꾸고, 다른 터미널에서 이렇게 직접 쏘면 된다:

    ros2 topic pub /vision/shoe_ready std_msgs/msg/Empty "{}" -1
"""

import itertools
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

AUTO_TRIGGER_INTERVAL_SEC = 15.0  # None으로 바꾸면 자동 발행 끄고 수동 트리거만 씀

# 종류 A(0)→B(1)→C(2)→D(3)를 순서대로 돌리면서, 길이는 소/중/대 경계를 걸치도록 섞음
# (fms_node.SIZE_THRESHOLDS_MM = (255, 275) 기준: <255=소, 255~275=중, >275=대)
TEST_BATCHES = [
    (0, [240, 260, 280, 250, 270]),  # A: 소/중/대/소/중
    (1, [230, 265, 300, 255, 275]),  # B
    (2, [245, 260, 290, 260, 260]),  # C
    (3, [250, 270, 285, 240, 265]),  # D
]


class VisionStub(Node):
    def __init__(self):
        super().__init__("vision_stub")
        self.create_service(Trigger, "/vision/classify_shoes", self._on_classify_request)
        self.ready_pub = self.create_publisher(Empty, "/vision/shoe_ready", 10)
        self._batches = itertools.cycle(TEST_BATCHES)

        if AUTO_TRIGGER_INTERVAL_SEC is not None:
            self.create_timer(AUTO_TRIGGER_INTERVAL_SEC, self._publish_ready)
            self.get_logger().info(
                f"비전 스텁 시작 — {AUTO_TRIGGER_INTERVAL_SEC:.0f}초마다 자동으로 신발 배치 발생"
            )
        else:
            self.get_logger().info(
                "비전 스텁 시작 — 자동 발행 꺼짐. 수동 트리거:\n"
                '  ros2 topic pub /vision/shoe_ready std_msgs/msg/Empty "{}" -1'
            )

    def _on_classify_request(self, request, response):
        shoe_type, lengths = next(self._batches)
        response.success = True
        response.message = json.dumps({"shoe_type": shoe_type, "shoe_length_mm": lengths})
        self.get_logger().info(f"[스텁] 분류 응답: type={shoe_type} lengths={lengths}")
        return response

    def _publish_ready(self):
        self.get_logger().info("[스텁] shoe_ready 트리거 발행")
        self.ready_pub.publish(Empty())


def main():
    rclpy.init()
    node = VisionStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
