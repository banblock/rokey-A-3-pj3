#!/usr/bin/env python3
"""Fleet Management System 예시 구현 — 노드 예약제(폐색 구간/Block System) 방식.

- Nav2 없이: 로봇에게는 "다음 노드로 이동" 명령만 순차적으로 보낸다.
- 노드 그래프를 미리 정의해두고, 각 노드에 락(mutex)을 건다.
- 로봇이 다음 노드로 가고 싶어도 그 노드가 잠겨있으면 대기시킨다.
- 실제 주행(등속 보간 등)은 로봇 쪽(시뮬레이션 스크립트)에서 수행하고,
  FMS는 "누가 어느 노드로 가도 되는지"만 결정한다.

토픽 프로토콜 (전부 JSON 문자열, std_msgs/String):
  구독:
    /vision/classification  { "shoe_id": int, "type": "A|B|C|D", "size": "소|중|대" }
    /amr/status              { "robot_id": str, "state": "idle|moving|arrived", "node_id": str }
  발행:
    /fms/commands             { "robot_id": str, "action": "move_to_node", "node_id": str, "position": [x,y,z] }
"""

import heapq
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 노드 그래프 정의 (씬 좌표와 대응)                              ║
# ╚══════════════════════════════════════════════════════════════╝
# 노드 하나 = {"position": (x, y, z), "neighbors": [node_id, ...]}
# PICKUP → HUB(교차로) → 각 랙 진입 노드 → 랙 안의 크기별 슬롯 노드
NODE_GRAPH = {
    "PICKUP":       {"position": (1.7, 0.0, 0.794), "neighbors": ["HUB"]},
    "HUB":          {"position": (1.0, 2.0, 0.0),   "neighbors": ["PICKUP", "RackA_IN", "RackB_IN", "RackC_IN", "RackD_IN", "REJECT", "BUFFER1", "BUFFER2"]},

    "RackA_IN":     {"position": (-3.75, 3.4, 0.0), "neighbors": ["HUB", "RackA_소", "RackA_중", "RackA_대"]},
    "RackA_소":      {"position": (-3.75, 4.3, 0.3), "neighbors": ["RackA_IN"]},
    "RackA_중":      {"position": (-3.75, 4.3, 1.2), "neighbors": ["RackA_IN"]},
    "RackA_대":      {"position": (-3.75, 4.3, 2.1), "neighbors": ["RackA_IN"]},

    "RackB_IN":     {"position": (-1.25, 3.4, 0.0), "neighbors": ["HUB", "RackB_소", "RackB_중", "RackB_대"]},
    "RackB_소":      {"position": (-1.25, 4.3, 0.3), "neighbors": ["RackB_IN"]},
    "RackB_중":      {"position": (-1.25, 4.3, 1.2), "neighbors": ["RackB_IN"]},
    "RackB_대":      {"position": (-1.25, 4.3, 2.1), "neighbors": ["RackB_IN"]},

    "RackC_IN":     {"position": (1.25, 3.4, 0.0),  "neighbors": ["HUB", "RackC_소", "RackC_중", "RackC_대"]},
    "RackC_소":      {"position": (1.25, 4.3, 0.3),  "neighbors": ["RackC_IN"]},
    "RackC_중":      {"position": (1.25, 4.3, 1.2),  "neighbors": ["RackC_IN"]},
    "RackC_대":      {"position": (1.25, 4.3, 2.1),  "neighbors": ["RackC_IN"]},

    "RackD_IN":     {"position": (3.75, 3.4, 0.0),  "neighbors": ["HUB", "RackD_소", "RackD_중", "RackD_대"]},
    "RackD_소":      {"position": (3.75, 4.3, 0.3),  "neighbors": ["RackD_IN"]},
    "RackD_중":      {"position": (3.75, 4.3, 1.2),  "neighbors": ["RackD_IN"]},
    "RackD_대":      {"position": (3.75, 4.3, 2.1),  "neighbors": ["RackD_IN"]},

    "REJECT":       {"position": (0.0, 5.7, 0.0),   "neighbors": ["HUB"]},
    "BUFFER1":      {"position": (-1.5, 5.7, 0.0),  "neighbors": ["HUB"]},
    "BUFFER2":      {"position": (1.5, 5.7, 0.0),   "neighbors": ["HUB"]},
}

# 그래프는 양방향이라고 가정하고 인접 리스트를 자동으로 대칭 보정
for _node_id, _node in NODE_GRAPH.items():
    for _nb in _node["neighbors"]:
        if _node_id not in NODE_GRAPH[_nb]["neighbors"]:
            NODE_GRAPH[_nb]["neighbors"].append(_node_id)


def _distance(a, b):
    ax, ay, az = NODE_GRAPH[a]["position"]
    bx, by, bz = NODE_GRAPH[b]["position"]
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


def shortest_path(start, goal):
    """다익스트라 최단 경로 — 노드 그래프가 작아서 단순 구현으로 충분."""
    dist = {start: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for nb in NODE_GRAPH[node]["neighbors"]:
            nd = d + _distance(node, nb)
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(heap, (nd, nb))

    if goal not in dist:
        return None  # 경로 없음

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# 신발 종류/크기 → 목적지 슬롯 노드 매핑
TARGET_SLOT_NODE = {
    (t, s): f"Rack{t}_{s}"
    for t in ["A", "B", "C", "D"]
    for s in ["소", "중", "대"]
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. FMS 노드                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
class FleetManagementSystem(Node):

    def __init__(self):
        super().__init__("fleet_management_system")

        # 로봇 레지스트리: {robot_id: {"state", "current_node", "path", "path_idx", "task"}}
        self.robots = {}

        # 노드 락: {node_id: robot_id 또는 None}
        self.node_locks = {node_id: None for node_id in NODE_GRAPH}

        # 아직 로봇이 배정되지 않은 태스크 큐
        self.task_queue = []

        self.command_pub = self.create_publisher(String, "/fms/commands", 10)
        self.create_subscription(String, "/vision/classification", self._on_classification, 10)
        self.create_subscription(String, "/amr/status", self._on_robot_status, 10)

        # 0.5초마다 태스크 배정 + 대기 중인 로봇의 다음 홉 진행 시도
        self.create_timer(0.5, self._dispatch_tick)

        self.get_logger().info("FMS 시작 — 노드 예약제(block system) 기반")

    # ── 로봇 등록 (실제로는 로봇이 처음 status를 보낼 때 자동 등록되게 해도 됨) ──
    def register_robot(self, robot_id, home_node="PICKUP"):
        self.robots[robot_id] = {
            "state": "idle",
            "current_node": home_node,
            "path": [],
            "path_idx": 0,
            "task": None,
        }
        self.node_locks[home_node] = robot_id

    # ── 비전 분류 결과 수신 → 태스크 큐에 추가 ──
    def _on_classification(self, msg):
        data = json.loads(msg.data)
        shoe_type, shoe_size = data["type"], data["size"]
        target_node = TARGET_SLOT_NODE.get((shoe_type, shoe_size))
        if target_node is None:
            self.get_logger().warn(f"알 수 없는 종류/크기: {shoe_type}/{shoe_size}")
            return
        self.task_queue.append({"shoe_id": data["shoe_id"], "target_node": target_node})
        self.get_logger().info(f"[태스크 등록] shoe_id={data['shoe_id']} → {target_node}")

    # ── 로봇 상태 업데이트 수신 ──
    def _on_robot_status(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]
        if robot_id not in self.robots:
            self.register_robot(robot_id)

        robot = self.robots[robot_id]

        if data["state"] == "arrived":
            arrived_node = data["node_id"]
            prev_node = robot["current_node"]

            # 이전 노드 락 해제, 새 노드는 도착했으니 이 로봇이 계속 점유
            if self.node_locks.get(prev_node) == robot_id:
                self.node_locks[prev_node] = None
            self.node_locks[arrived_node] = robot_id
            robot["current_node"] = arrived_node
            robot["path_idx"] += 1

            if robot["path_idx"] >= len(robot["path"]) - 1:
                # 목적지(경로의 마지막 노드) 도착 완료
                robot["state"] = "idle"
                robot["task"] = None
                robot["path"] = []
                robot["path_idx"] = 0
                self.get_logger().info(f"[{robot_id}] 목적지 도착, idle 전환")
            else:
                robot["state"] = "waiting_next_hop"

    # ── 주기적으로: 새 태스크 배정 + 다음 홉 진행 가능 여부 확인 ──
    def _dispatch_tick(self):
        # 1) idle 로봇에게 태스크 배정
        for robot_id, robot in self.robots.items():
            if robot["state"] == "idle" and self.task_queue:
                task = self.task_queue.pop(0)
                path = shortest_path(robot["current_node"], task["target_node"])
                if path is None:
                    self.get_logger().error(f"경로 없음: {robot['current_node']} → {task['target_node']}")
                    continue
                robot["task"] = task
                robot["path"] = path
                robot["path_idx"] = 0
                robot["state"] = "waiting_next_hop"
                self.get_logger().info(f"[{robot_id}] 태스크 배정: {' → '.join(path)}")

        # 2) 다음 홉 대기 중인 로봇들 — 다음 노드가 비어있으면 락 걸고 명령 전송
        for robot_id, robot in self.robots.items():
            if robot["state"] != "waiting_next_hop":
                continue
            next_idx = robot["path_idx"] + 1
            if next_idx >= len(robot["path"]):
                continue
            next_node = robot["path"][next_idx]

            if self.node_locks[next_node] is None:
                self.node_locks[next_node] = robot_id  # 예약(이동 시작 시점에 미리 잠금)
                robot["state"] = "moving"
                self._send_move_command(robot_id, next_node)
            # else: 잠겨있으면 이번 tick은 그냥 대기 (다음 tick에 재시도)

    def _send_move_command(self, robot_id, node_id):
        position = list(NODE_GRAPH[node_id]["position"])
        msg = String()
        msg.data = json.dumps({
            "robot_id": robot_id,
            "action": "move_to_node",
            "node_id": node_id,
            "position": position,
        })
        self.command_pub.publish(msg)
        self.get_logger().info(f"[{robot_id}] → {node_id} 이동 명령 전송")


def main(args=None):
    rclpy.init(args=args)
    node = FleetManagementSystem()
    # 데모용으로 로봇 1대 미리 등록 (실제로는 /amr/status 첫 수신 시 자동 등록됨)
    node.register_robot("amr_1", home_node="PICKUP")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
