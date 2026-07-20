#!/usr/bin/env python3
"""Fleet Driver — FMS와 Isaac Sim(순수 물리 세계) 사이의 중간 관리자.

3단 구조:
  FMS(관제탑, fms_node.py) → Fleet Driver(이 노드) → Isaac Sim(순수 ROS2 브리지)

Isaac Sim은 로봇 제어 로직을 전혀 갖지 않는다 — 각 Nova Carter의 ROS2 브리지
(디퍼렌셜 드라이브 + 오도메트리)만 켜둔 채로, 외부에서 오는 /<robot_id>/cmd_vel
값대로 바퀴를 굴리고 /<robot_id>/odom을 내뱉기만 한다. 이 노드는 순수 rclpy로만
동작하고 Isaac Sim에 대한 의존성이 전혀 없다 — 시뮬레이션 없이도 단독 실행/테스트 가능.

하는 일:
  1. FMS의 /fms/commands(다음 노드 좌표)를 robot_id별로 받아 목표로 저장한다.
  2. 각 로봇의 /<robot_id>/odom을 구독해 실시간 위치/자세(x, y, yaw)를 추적한다.
  3. go-to-goal 제어(각도 오차가 크면 제자리 회전 → 정렬되면 직진)로
     /<robot_id>/cmd_vel(Twist)을 계산해 내보낸다.
  4. 목표 반경 안에 들어오면 정지시키고 FMS에 /amr/status arrived를 보고한다.

주의: NODE_GRAPH의 랙 슬롯 Z값(0.3~2.1m)은 "선반 높이"이지 지상 로봇이 실제로
올라갈 좌표가 아니다 — Nova Carter는 지상 주행체라 내비게이션은 XY 평면만 쓰고
Z는 무시한다(신발을 몇 번째 선반에 놓을지는 이 노드가 관여하지 않는 별도 문제).
"""

import json
import math
import sys

# fleet_config.py는 이 ROS 패키지 밖(isaacpjt/sorting_line/)에 있는 순수 데이터
# 모듈이다 — fms_node.py와 동일한 이유로 패키지 안으로 옮기지 않았다 (Isaac Sim
# 스크립트도 rclpy 없이 그대로 가져다 쓴다). setup.py의 data_files로 같이 설치해두고
# ament_index로 경로를 찾는다 — 설치 방식(심링크/일반)과 무관하게 항상 동작한다.
from ament_index_python.packages import get_package_share_directory

_SHARE_DIR = get_package_share_directory("sorting_line_fms")
if _SHARE_DIR not in sys.path:
    sys.path.insert(0, _SHARE_DIR)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from fleet_config import ROBOT_HOME_NODE  # noqa: E402

ARRIVE_RADIUS_M = 0.15
ANGLE_TOLERANCE_RAD = 0.25   # 이 안이면 회전 없이 바로 직진
MAX_LINEAR_MPS = 0.6
MAX_ANGULAR_RPS = 1.2
K_LINEAR = 0.8
K_ANGULAR = 1.5
CONTROL_PERIOD_SEC = 0.05    # 20Hz


def _yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class FleetDriver(Node):

    def __init__(self):
        super().__init__("fleet_driver")
        self.robots = {}  # robot_id -> {x, y, yaw, has_odom, target_xy, target_node, cmd_pub}

        self.status_pub = self.create_publisher(String, "/amr/status", 10)
        self.create_subscription(String, "/fms/commands", self._on_command, 10)

        for robot_id in ROBOT_HOME_NODE:
            self._ensure_robot(robot_id)
            self._publish_status(robot_id, "idle", "BOOT")

        self.create_timer(CONTROL_PERIOD_SEC, self._control_tick)
        self.get_logger().info(f"Fleet Driver 시작 — {len(self.robots)}대 담당 (cmd_vel/odom 기반)")

    def _ensure_robot(self, robot_id):
        if robot_id in self.robots:
            return
        self.robots[robot_id] = {
            "x": 0.0, "y": 0.0, "yaw": 0.0, "has_odom": False,
            "target_xy": None, "target_node": None,
            "cmd_pub": self.create_publisher(Twist, f"/{robot_id}/cmd_vel", 10),
        }
        # default-arg로 robot_id를 바인딩 — 그냥 클로저로 캡처하면 루프 후반 값으로
        # 전부 덮어써지는 전형적인 late-binding 버그가 남
        self.create_subscription(
            Odometry, f"/{robot_id}/odom",
            lambda msg, rid=robot_id: self._on_odom(rid, msg), 10,
        )

    def _on_odom(self, robot_id, msg):
        robot = self.robots[robot_id]
        pos = msg.pose.pose.position
        robot["x"], robot["y"] = pos.x, pos.y
        robot["yaw"] = _yaw_from_quaternion(msg.pose.pose.orientation)
        robot["has_odom"] = True

    def _on_command(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]
        self._ensure_robot(robot_id)
        x, y, _z = data["position"]  # Z(선반 높이)는 지상 주행 목표에는 쓰지 않음
        self.robots[robot_id]["target_xy"] = (x, y)
        self.robots[robot_id]["target_node"] = data["node_id"]

    def _control_tick(self):
        for robot_id, robot in self.robots.items():
            target = robot["target_xy"]
            if target is None or not robot["has_odom"]:
                continue

            dx = target[0] - robot["x"]
            dy = target[1] - robot["y"]
            distance = math.hypot(dx, dy)

            if distance <= ARRIVE_RADIUS_M:
                arrived_node = robot["target_node"]
                robot["target_xy"] = None
                robot["target_node"] = None
                self._stop(robot_id)
                self._publish_status(robot_id, "arrived", arrived_node)
                continue

            heading_error = _normalize_angle(math.atan2(dy, dx) - robot["yaw"])

            twist = Twist()
            if abs(heading_error) > ANGLE_TOLERANCE_RAD:
                twist.linear.x = 0.0  # 큰 각도 오차는 제자리 회전으로 먼저 정렬
            else:
                twist.linear.x = min(K_LINEAR * distance, MAX_LINEAR_MPS)
            twist.angular.z = max(-MAX_ANGULAR_RPS, min(MAX_ANGULAR_RPS, K_ANGULAR * heading_error))
            robot["cmd_pub"].publish(twist)

    def _stop(self, robot_id):
        self.robots[robot_id]["cmd_pub"].publish(Twist())

    def _publish_status(self, robot_id, state, node_id):
        msg = String()
        msg.data = json.dumps({"robot_id": robot_id, "state": state, "node_id": node_id})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FleetDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
