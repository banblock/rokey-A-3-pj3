#!/usr/bin/env python3
"""AGV 마커(AprilTag) 기반 위치 보정 노드.

각 로봇의 하향 카메라(1_conveyor_sorter_env.py의 spawn_marker_camera가 붙인
/<robot_id>/marker_cam/image_raw)에서 마커 ID만 검출한다 — 6DOF pose 추정
(카메라 내참수 + estimatePoseSingleMarkers)까지는 안 한다. 마커가 붙은
지점(PICKUP_X, RackX_280 등)은 애초에 로봇이 "정확히 멈춰야 하는" 자리라서,
"이 마커가 보인다 = 지금 이 노드의 알려진 좌표에 있다"로 충분히 정확하게
스냅(snap) 보정할 수 있다 — 실물 SLAM/AMCL과 달리 연속적인 자기 위치 추정이
목적이 아니라 "누적 드리프트를 주기적으로 리셋"하는 AGV 스타일 보정이기
때문이다(project_real_deployment_roadmap 메모리의 2026-07-23 논의 참고).

두 컴퓨터 구조(시뮬레이션+YOLO / FMS)를 전제로, 이 노드는 시뮬레이션 컴퓨터
쪽에서 띄우는 걸 권장한다 — 그래야 원본 카메라 영상이 네트워크를 안 넘어가고,
가벼운 보정값(JSON, robot_id+x+y+node_id)만 /fms/pose_correction으로 FMS
컴퓨터의 fleet_driver.py에 전달된다.
"""
import json
import sys

from ament_index_python.packages import get_package_share_directory

_SHARE_DIR = get_package_share_directory("sorting_line_fms")
if _SHARE_DIR not in sys.path:
    sys.path.insert(0, _SHARE_DIR)

import importlib

import cv2.aruco as aruco
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from marker_config import MARKER_DICT_NAME, build_marker_maps  # noqa: E402

# 같은 마커 앞에 몇 초씩 머무르는 동안(픽업 대기, 배치 대기 등) 매 프레임마다
# 보정을 쏘면 /fms/pose_correction만 불필요하게 바빠진다 — 로봇 하나당, 같은
# 마커에 대해서는 이 간격 이상 지나야 다시 보정을 보낸다.
CORRECTION_COOLDOWN_SEC = 1.0


class AgvMarkerLocalizer(Node):

    def __init__(self):
        super().__init__("agv_marker_localizer")
        self.declare_parameter("config_module", "fleet_config")
        config_module_name = self.get_parameter("config_module").get_parameter_value().string_value
        config = importlib.import_module(config_module_name)
        self.NODE_GRAPH = config.NODE_GRAPH
        self.ROBOT_HOME_NODE = config.ROBOT_HOME_NODE

        _, self._node_id_by_marker_id = build_marker_maps(self.NODE_GRAPH)
        self._dictionary = aruco.getPredefinedDictionary(getattr(aruco, MARKER_DICT_NAME))
        self._detector_params = aruco.DetectorParameters()
        self._bridge = CvBridge()
        self._last_correction_at = {}  # (robot_id, node_id) -> 마지막 발행 시각

        self.correction_pub = self.create_publisher(String, "/fms/pose_correction", 10)

        for robot_id in self.ROBOT_HOME_NODE:
            self.create_subscription(
                Image, f"/{robot_id}/marker_cam/image_raw",
                lambda msg, rid=robot_id: self._on_image(rid, msg), 10,
            )

        self.get_logger().info(
            f"AGV 마커 위치 보정 노드 시작 — 로봇 {len(self.ROBOT_HOME_NODE)}대, "
            f"마커 {len(self._node_id_by_marker_id)}개 등록"
        )

    def _on_image(self, robot_id, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        corners, ids, _ = aruco.detectMarkers(frame, self._dictionary, parameters=self._detector_params)
        if ids is None:
            return

        # 한 프레임에 여러 마커가 잡혀도(카메라 화각이 넓거나 옆 슬롯 마커까지
        # 걸리는 경우) 로봇 1대가 지금 있어야 할 자리는 하나뿐이므로 첫 번째
        # 인식분만 쓴다 — corners는 실제로 안 쓰므로 여기선 무시.
        del corners
        node_id = None
        for marker_id in ids.flatten():
            node_id = self._node_id_by_marker_id.get(int(marker_id))
            if node_id is not None:
                break
        if node_id is None:
            return  # 등록 안 된(다른 용도의) 마커거나 오검출 — 무시

        now = self.get_clock().now()
        last_at = self._last_correction_at.get((robot_id, node_id))
        if last_at is not None and (now - last_at) < Duration(seconds=CORRECTION_COOLDOWN_SEC):
            return
        self._last_correction_at[(robot_id, node_id)] = now

        position = self.NODE_GRAPH[node_id]["position"]
        out = String()
        out.data = json.dumps({
            "robot_id": robot_id,
            "node_id": node_id,
            "x": position[0],
            "y": position[1],
        })
        self.correction_pub.publish(out)
        self.get_logger().info(f"[{robot_id}] 마커 인식 — {node_id}({position[0]:.3f}, {position[1]:.3f})로 보정")


def main(args=None):
    rclpy.init(args=args)
    node = AgvMarkerLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
