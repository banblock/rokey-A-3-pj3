#!/usr/bin/env python3
"""edge_conflict(구간 충돌) 회피 로직 단독 테스트용 FMS.

crossing_test_config.py의 X자 교차 그래프에서 로봇 2대를 영원히 왕복시킨다.
fms_node.py의 신발 분류/픽업/랙 배치 같은 업무 로직은 전혀 없고, 그 파일의
node_locks(노드 뮤텍스) + _find_edge_conflict(구간 기하 교차 판정) + 이동
워치독 3가지만 그대로 가져왔다 — 두 로봇의 경로가 공유 노드 없이 순수
기하학적으로만 겹치는 상황(node_locks만으로는 절대 못 막는 상황)에서 실제로
충돌 없이 서로 양보하는지 확인하기 위한 용도.

fleet_driver.py는 수정 없이 그대로 재사용한다 — /fms/commands, /amr/status
메시지 형식이 fms_node.py와 완전히 동일해서, config_module 파라미터만 이
그래프로 바꿔주면 fleet_driver.py는 자기가 상대하는 그래프가 운영용인지
테스트용인지 몰라도 된다.
"""

import json
import random
import sys

from ament_index_python.packages import get_package_share_directory

_SHARE_DIR = get_package_share_directory("sorting_line_fms")
if _SHARE_DIR not in sys.path:
    sys.path.insert(0, _SHARE_DIR)

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String

from crossing_test_config import NODE_GRAPH, ROBOT_HOME_NODE  # noqa: E402

MOVE_TIMEOUT_SEC = 30.0
# 매 왕복마다 끝점에서 머무는 시간을 무작위로 줘서, 두 로봇이 교차점 근처에서
# 마주치는 타이밍이 매번 달라지게 한다 — 고정 시간이면 항상 같은 상대적 위상으로만
# 움직여서 "가끔 동시에 지나가는" 상황을 우연에 기대지 않고도 재현할 수 있다.
DWELL_MIN_SEC = 0.5
DWELL_MAX_SEC = 4.0
BLOCK_LOG_INTERVAL_SEC = 2.0  # 대기 로그를 이 간격으로만 반복(매 tick마다 찍으면 노이즈)


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4):
    """fms_node.py의 동명 함수와 동일한 CCW 기반 선분-교차 판정."""
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


class CrossingTestFMS(Node):

    def __init__(self):
        super().__init__("crossing_test_fms")
        self.node_locks = {node_id: None for node_id in NODE_GRAPH}
        self.robots = {}
        self._blocked_since = {}  # robot_id -> 마지막 대기 로그를 찍은 시각

        self.command_pub = self.create_publisher(String, "/fms/commands", 10)
        self.create_subscription(String, "/amr/status", self._on_robot_status, 10)

        for robot_id, home_node in ROBOT_HOME_NODE.items():
            self.robots[robot_id] = {
                "current_node": home_node,
                # 이 그래프는 로봇마다 이웃이 정확히 하나뿐이라(왕복 상대편 끝),
                # "다음 목표"가 항상 유일하게 정해진다.
                "target_node": NODE_GRAPH[home_node]["neighbors"][0],
                "state": "waiting_next_hop",
                "move_started_at": None,
                "move_target": None,
                "dwell_until": None,
            }
            self.node_locks[home_node] = robot_id

        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"Crossing 테스트 FMS 시작 — 로봇 {len(self.robots)}대, "
            f"공유 노드 없이 대각선만 교차하는 그래프에서 영원히 왕복"
        )

    def _on_robot_status(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]
        if robot_id not in self.robots:
            return
        if data["state"] != "arrived":
            return

        robot = self.robots[robot_id]
        arrived_node = data["node_id"]
        prev_node = robot["current_node"]
        if self.node_locks.get(prev_node) == robot_id:
            self.node_locks[prev_node] = None
        self.node_locks[arrived_node] = robot_id
        robot["current_node"] = arrived_node
        robot["move_started_at"] = None
        robot["move_target"] = None

        # 끝점(HOME 또는 FAR)에 도착 — 무작위 시간만큼 머문 뒤 반대쪽으로 재출발.
        robot["target_node"] = NODE_GRAPH[arrived_node]["neighbors"][0]
        robot["dwell_until"] = self.get_clock().now() + Duration(
            seconds=random.uniform(DWELL_MIN_SEC, DWELL_MAX_SEC)
        )
        robot["state"] = "dwelling"
        self.get_logger().info(
            f"[{robot_id}] {arrived_node} 도착 — 잠시 대기 후 {robot['target_node']}(으)로 재출발"
        )

    def _tick(self):
        now = self.get_clock().now()

        for robot_id, robot in self.robots.items():
            if robot["state"] == "moving" and robot["move_started_at"] is not None:
                elapsed = (now - robot["move_started_at"]).nanoseconds / 1e9
                if elapsed > MOVE_TIMEOUT_SEC:
                    self.get_logger().error(f"[{robot_id}] 이동이 {MOVE_TIMEOUT_SEC:.0f}초 넘게 응답 없음 → 재시도")
                    if self.node_locks.get(robot["move_target"]) == robot_id:
                        self.node_locks[robot["move_target"]] = None
                    robot["state"] = "waiting_next_hop"
                    robot["move_started_at"] = None
                    robot["move_target"] = None

            if robot["state"] == "dwelling" and robot["dwell_until"] is not None and now >= robot["dwell_until"]:
                robot["dwell_until"] = None
                robot["state"] = "waiting_next_hop"

        for robot_id in self.robots:
            self._try_move(robot_id, now)

    def _try_move(self, robot_id, now):
        robot = self.robots[robot_id]
        if robot["state"] != "waiting_next_hop":
            return

        next_node = robot["target_node"]

        holder_id = self.node_locks[next_node]
        if holder_id is not None:
            self._note_block(robot_id, next_node, holder_id, now, "node_lock")
            return

        # 이 그래프의 핵심 — 두 로봇이 공유하는 노드는 하나도 없으니, 위의
        # node_locks 체크는 이 시나리오에서 절대 걸릴 일이 없다. 실제로 로봇을
        # 멈춰 세우는 건 오직 이 구간(edge) 기하 교차 판정뿐이다.
        conflicting_id = self._find_edge_conflict(robot_id, robot["current_node"], next_node)
        if conflicting_id is not None:
            self._note_block(robot_id, next_node, conflicting_id, now, "edge_conflict")
            return

        self.node_locks[next_node] = robot_id
        robot["state"] = "moving"
        robot["move_started_at"] = now
        robot["move_target"] = next_node
        self._send_move_command(robot_id, next_node)
        self._blocked_since.pop(robot_id, None)

    def _find_edge_conflict(self, robot_id, from_node, to_node):
        p1 = NODE_GRAPH[from_node]["position"][:2]
        p2 = NODE_GRAPH[to_node]["position"][:2]
        for other_id, other in self.robots.items():
            if other_id == robot_id or other["state"] != "moving":
                continue
            p3 = NODE_GRAPH[other["current_node"]]["position"][:2]
            p4 = NODE_GRAPH[other["move_target"]]["position"][:2]
            if _segments_intersect(p1, p2, p3, p4):
                return other_id
        return None

    def _note_block(self, robot_id, next_node, holder_id, now, reason):
        last_logged = self._blocked_since.get(robot_id)
        if last_logged is not None:
            elapsed = (now - last_logged).nanoseconds / 1e9
            if elapsed < BLOCK_LOG_INTERVAL_SEC:
                return
        self._blocked_since[robot_id] = now
        reason_kr = "구간 교차(edge_conflict)" if reason == "edge_conflict" else "노드 점유"
        self.get_logger().warn(f"[{robot_id}] {next_node} 진입 대기 중 — {reason_kr}, {holder_id}에게 막힘")

    def _send_move_command(self, robot_id, node_id):
        position = list(NODE_GRAPH[node_id]["position"])
        msg = String()
        msg.data = json.dumps({
            "robot_id": robot_id,
            "action": "move_to_node",
            "node_id": node_id,
            "position": position,
            # 이 그래프는 세분화 칸이 없어(간선이 이미 짧음) 매 홉이 항상 마지막 홉이다.
            "is_final_hop": True,
        })
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrossingTestFMS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
