#!/usr/bin/env python3
"""Fleet Management System — 8대 전용(종류별 2대 전담) 다중 로봇 버젼

- 신발 종류(A/B/C/D) 하나당 로봇 2대를 전담 배정한다. 다른 종류의 태스크는
  받지 않으므로 랙 간 교차 이동이 줄어 혼잡도가 낮아진다.
- 로봇마다 "홈 슬롯(WAIT_N)"을 하나씩 전용으로 가진다. 예전처럼 전체 로봇이
  PICKUP_WAIT 노드 하나를 공유하면 두 번째로 복귀하는 로봇부터 영원히
  막히므로(노드=뮤텍스는 동시 점유자가 1명), 로봇 수만큼 대기 슬롯을 만든다.
- 노드 이동 명령을 보낸 뒤 일정 시간 안에 "arrived"가 안 오면(내비게이션
  실패, 메시지 유실 등) 락을 풀고 재시도시키는 워치독을 둔다 — 그렇지 않으면
  로봇 1대가 멈추는 순간 그 노드를 지나야 하는 나머지가 전부 연쇄 정지한다.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

from m0609_interfaces.srv import ClassifyShoes

from fleet_config import (
    NODE_GRAPH,
    ROBOT_HOME_NODE,
    ROBOT_SHOE_TYPE,
    ROBOTS_PER_TYPE,
    SHOE_TYPES,
    TARGET_SLOT_NODE,
    shortest_path,
)

MOVE_TIMEOUT_SEC = 15.0  # 이동 명령 후 이만큼 지나도 arrived가 없으면 멈춘 것으로 간주
DEADLOCK_ALERT_SEC = 5.0  # 다른 로봇이 다음 노드를 점유해 이만큼 계속 못 넘어가면 교착으로 보고

# 신발 길이(mm) → 크기 등급 경계값. 실측 후 조정 예정(현재는 자리표시용 플레이스홀더).
SIZE_THRESHOLDS_MM = (255, 275)  # length < 255 → 소 | 255 ~ 275 → 중 | length > 275 → 대


def _bucket_size(length_mm):
    low, high = SIZE_THRESHOLDS_MM
    if length_mm < low:
        return "소"
    if length_mm <= high:
        return "중"
    return "대"


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. FMS 노드 로직                                               ║
# ╚══════════════════════════════════════════════════════════════╝
class FleetManagementSystem(Node):

    def __init__(self):
        super().__init__("fleet_management_system")
        self.robots = {}
        self.node_locks = {node_id: None for node_id in NODE_GRAPH}
        # 종류별로 독립된 큐 — 담당 로봇에게만 배정되므로 큐도 종류별로 나눈다.
        self.task_queues = {shoe_type: [] for shoe_type in SHOE_TYPES}
        self._rr_offset = 0  # 다음 홉 배정 시 순회 시작점을 매 tick 회전(기아 방지)
        self._next_shoe_id = 0
        self._blocked_since = {}  # robot_id -> {"since","next_node","holder","reported"} — 동선 겹침 추적

        self.command_pub = self.create_publisher(String, "/fms/commands", 10)
        # 담당 종류의 로봇이 PICKUP에 도착해 신발을 받을 준비가 됐음을 시뮬레이션
        # 구동 파일에 알리는 신호 — 실제 신발 프림을 옮기는 동작은 그 파일이 담당한다.
        self.pickup_ready_pub = self.create_publisher(String, "/fms/pickup_ready", 10)
        # 로봇 동선이 겹쳐 한쪽이 오래 대기하게 되면(교착 상태) 알리는 토픽 —
        # 지금은 구독자가 없어도, 나중에 만들 상위 대시보드 노드가 그대로 구독하면 됨.
        self.deadlock_pub = self.create_publisher(String, "/fms/deadlock_alert", 10)
        self.create_subscription(String, "/amr/status", self._on_robot_status, 10)

        # 비전 쪽에서 "신발 5켤레 배치 분류 끝났다" 신호(트리거)를 주면,
        # 그 순간 실제 분류 결과(종류 1개 + 길이 5개)를 srv로 가져온다.
        self.classify_client = self.create_client(ClassifyShoes, "/vision/classify_shoes")
        self.create_subscription(Empty, "/vision/shoe_ready", self._on_shoe_ready, 10)

        self.create_timer(0.5, self._dispatch_tick)
        self.get_logger().info(
            f"FMS 시작 — 로봇 {len(ROBOT_SHOE_TYPE)}대, 종류별 {ROBOTS_PER_TYPE}대 전담 모드"
        )

    def register_robot(self, robot_id):
        home_node = ROBOT_HOME_NODE.get(robot_id)
        shoe_type = ROBOT_SHOE_TYPE.get(robot_id)
        if home_node is None or shoe_type is None:
            self.get_logger().error(
                f"[{robot_id}] 등록 거부: ROBOT_SHOE_TYPE에 정의되지 않은 robot_id (8대 전용 구성)"
            )
            return False
        if self.node_locks.get(home_node) not in (None, robot_id):
            # 두 로봇이 같은 home_node를 보고할 일은 없지만(로봇마다 슬롯이 다름),
            # 방어적으로 남겨둔다 — 예전처럼 조건 없이 락을 덮어쓰지 않는다.
            self.get_logger().error(
                f"[{robot_id}] 등록 거부: 홈 슬롯 {home_node}이(가) 이미 "
                f"{self.node_locks[home_node]}에 의해 점유됨"
            )
            return False

        self.robots[robot_id] = {
            "state": "idle",
            "current_node": home_node,
            "home_node": home_node,
            "shoe_type": shoe_type,
            "path": [],
            "path_idx": 0,
            "task": None,
            "move_started_at": None,
            "move_target": None,
        }
        self.node_locks[home_node] = robot_id
        self.get_logger().info(f"[{robot_id}] 등록 완료 (담당 종류={shoe_type}, 홈 슬롯={home_node})")
        return True

    def _on_shoe_ready(self, msg):
        if not self.classify_client.service_is_ready():
            self.get_logger().warn("분류 서비스(/vision/classify_shoes)가 아직 준비되지 않아 요청을 건너뜀")
            return
        future = self.classify_client.call_async(ClassifyShoes.Request())
        future.add_done_callback(self._on_classify_response)

    def _on_classify_response(self, future):
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001 — 서비스 호출 실패 자체를 로그로 남기기 위함
            self.get_logger().error(f"분류 서비스 호출 실패: {exc}")
            return

        if not response.success:
            self.get_logger().warn("분류 서비스 응답: 새 배치 없음(success=False)")
            return

        if not (0 <= response.shoe_type < len(SHOE_TYPES)):
            self.get_logger().error(f"알 수 없는 shoe_type 인덱스: {response.shoe_type}")
            return
        shoe_type = SHOE_TYPES[response.shoe_type]

        for length_mm in response.shoe_length_mm:
            shoe_size = _bucket_size(length_mm)
            target_node = TARGET_SLOT_NODE[(shoe_type, shoe_size)]
            self._next_shoe_id += 1
            self.task_queues[shoe_type].append({"shoe_id": self._next_shoe_id, "target_node": target_node})
            self.get_logger().info(
                f"[신규 태스크] shoe_id={self._next_shoe_id} 길이={length_mm}mm({shoe_size}) "
                f"→ {target_node} (담당 종류={shoe_type})"
            )

    def _on_robot_status(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]

        if robot_id not in self.robots:
            if not self.register_robot(robot_id):
                return

        robot = self.robots[robot_id]

        if data["state"] == "arrived":
            arrived_node = data["node_id"]
            prev_node = robot["current_node"]

            # 이전 노드 잠금 해제, 새 노드 잠금
            if self.node_locks.get(prev_node) == robot_id:
                self.node_locks[prev_node] = None
            self.node_locks[arrived_node] = robot_id

            robot["current_node"] = arrived_node
            robot["path_idx"] += 1
            robot["move_started_at"] = None
            robot["move_target"] = None

            # PICKUP은 모든 태스크 경로가 반드시 거쳐가는 지점이라, 살아있는
            # 태스크를 든 채 여기 도착했다는 건 "이 로봇이 받아야 할 신발 앞에
            # 섰다"는 뜻이다 — 이 순간 시뮬레이션 쪽에 신발 전달 준비 완료를 알린다.
            if arrived_node == "PICKUP" and robot["task"] is not None:
                self._publish_pickup_ready(robot_id, robot)

            if robot["path_idx"] >= len(robot["path"]) - 1:
                # 목적지 도착 완료
                robot["state"] = "idle"
                robot["task"] = None
                robot["path"] = []
                robot["path_idx"] = 0

                # 자기 전용 홈 슬롯이 아닌 곳(랙, PICKUP 등)에 멈췄다면 즉시 복귀
                if arrived_node != robot["home_node"]:
                    self.get_logger().info(f"[{robot_id}] 작업 완료. 홈 슬롯({robot['home_node']})으로 복귀합니다.")
                    return_path = shortest_path(arrived_node, robot["home_node"])
                    if return_path:
                        robot["path"] = return_path
                        robot["path_idx"] = 0
                        robot["state"] = "waiting_next_hop"
                    else:
                        self.get_logger().error(
                            f"[{robot_id}] 홈 슬롯 복귀 경로 없음: {arrived_node} → {robot['home_node']}"
                        )
            else:
                robot["state"] = "waiting_next_hop"

    def _dispatch_tick(self):
        now = self.get_clock().now()

        # 0) 이동 워치독 — 명령 보낸 뒤 응답 없이 너무 오래 걸리면 락을 풀고 재시도
        for robot_id, robot in self.robots.items():
            if robot["state"] != "moving" or robot["move_started_at"] is None:
                continue
            elapsed = (now - robot["move_started_at"]).nanoseconds / 1e9
            if elapsed > MOVE_TIMEOUT_SEC:
                stuck_node = robot["move_target"]
                self.get_logger().error(
                    f"[{robot_id}] {stuck_node} 이동이 {MOVE_TIMEOUT_SEC:.0f}초 넘게 응답 없음 → 락 해제 후 재시도"
                )
                if self.node_locks.get(stuck_node) == robot_id:
                    self.node_locks[stuck_node] = None
                robot["state"] = "waiting_next_hop"
                robot["move_started_at"] = None
                robot["move_target"] = None

        # 1) 태스크 배정 — 종류가 일치하는 idle 로봇 중, 목적지까지 더 가까운 로봇 우선
        for shoe_type, queue in self.task_queues.items():
            while queue:
                candidates = [
                    (robot_id, robot)
                    for robot_id, robot in self.robots.items()
                    if robot["state"] == "idle" and robot["shoe_type"] == shoe_type
                ]
                if not candidates:
                    break

                task = queue[0]
                best_robot_id, best_path = None, None
                for robot_id, robot in candidates:
                    path = shortest_path(robot["current_node"], task["target_node"])
                    if path is None:
                        continue
                    if best_path is None or len(path) < len(best_path):
                        best_robot_id, best_path = robot_id, path

                if best_robot_id is None:
                    self.get_logger().error(f"경로 불가: {task['target_node']} (담당 종류={shoe_type})")
                    break

                queue.pop(0)
                robot = self.robots[best_robot_id]
                robot["task"] = task
                robot["path"] = best_path
                robot["path_idx"] = 0
                robot["state"] = "waiting_next_hop"
                self.get_logger().info(f"[{best_robot_id}] 태스크 시작: {' → '.join(best_path)}")

        # 2) 다음 홉 이동 시도 (노드 예약제/Mutex 핵심 로직)
        #    매 tick마다 순회 시작 로봇을 한 칸씩 돌려서, 특정 로봇이 계속 우선권을
        #    독점해 나머지가 굶는 상황을 방지한다.
        robot_ids = list(self.robots.keys())
        if robot_ids:
            self._rr_offset %= len(robot_ids)
            ordered_ids = robot_ids[self._rr_offset:] + robot_ids[:self._rr_offset]
            self._rr_offset += 1
        else:
            ordered_ids = []

        for robot_id in ordered_ids:
            robot = self.robots[robot_id]
            if robot["state"] != "waiting_next_hop":
                continue

            next_idx = robot["path_idx"] + 1
            if next_idx >= len(robot["path"]):
                continue

            next_node = robot["path"][next_idx]
            holder_id = self.node_locks[next_node]

            # 다음 노드가 비어있다면(None) 점유 후 이동 명령 전송!
            if holder_id is None:
                self.node_locks[next_node] = robot_id  # 락 걸기
                robot["state"] = "moving"
                robot["move_started_at"] = now
                robot["move_target"] = next_node
                self._send_move_command(robot_id, next_node)
                self._clear_deadlock_alert(robot_id, now)
            else:
                # 다른 로봇이 점유 중이라 못 넘어감 — 얼마나 오래 막혀있는지 추적하다가
                # DEADLOCK_ALERT_SEC를 넘기면 한 번만 알림을 보낸다.
                self._track_deadlock_alert(robot_id, next_node, holder_id, now)

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

    def _publish_pickup_ready(self, robot_id, robot):
        task = robot["task"]
        msg = String()
        msg.data = json.dumps({
            "robot_id": robot_id,
            "shoe_type": robot["shoe_type"],
            "shoe_id": task["shoe_id"],       # FMS 내부 카운터 — 로깅/추적용
            "target_node": task["target_node"],
        })
        self.pickup_ready_pub.publish(msg)
        self.get_logger().info(
            f"[{robot_id}] PICKUP 도착 — 신발 수령 준비 완료 (shoe_type={robot['shoe_type']}, "
            f"shoe_id={task['shoe_id']})"
        )

    def _track_deadlock_alert(self, robot_id, next_node, holder_id, now):
        entry = self._blocked_since.get(robot_id)
        if entry is None:
            self._blocked_since[robot_id] = {
                "since": now, "next_node": next_node, "holder": holder_id, "reported": False,
            }
            return

        # 막고 있는 대상이 바뀌었을 수도 있으니 최신 정보로 갱신하되, 시작 시각은 유지
        entry["next_node"] = next_node
        entry["holder"] = holder_id
        if entry["reported"]:
            return

        elapsed_sec = (now - entry["since"]).nanoseconds / 1e9
        if elapsed_sec >= DEADLOCK_ALERT_SEC:
            entry["reported"] = True
            self._publish_deadlock_alert("raised", robot_id, next_node, holder_id, elapsed_sec)

    def _clear_deadlock_alert(self, robot_id, now):
        entry = self._blocked_since.pop(robot_id, None)
        if entry is not None and entry["reported"]:
            elapsed_sec = (now - entry["since"]).nanoseconds / 1e9
            self._publish_deadlock_alert(
                "cleared", robot_id, entry["next_node"], entry["holder"], elapsed_sec
            )

    def _publish_deadlock_alert(self, event, robot_id, next_node, holder_id, elapsed_sec):
        msg = String()
        msg.data = json.dumps({
            "event": event,  # "raised" | "cleared"
            "robot_id": robot_id,
            "blocked_at_node": next_node,
            "blocked_by_robot": holder_id,
            "blocked_seconds": round(elapsed_sec, 1),
        })
        self.deadlock_pub.publish(msg)
        if event == "raised":
            self.get_logger().warn(
                f"[교착 감지] {robot_id} — {next_node} 진입 대기 중, {holder_id}가 점유 "
                f"({elapsed_sec:.1f}초째 정체)"
            )
        else:
            self.get_logger().info(f"[교착 해소] {robot_id} — {elapsed_sec:.1f}초 만에 재개")


def main(args=None):
    rclpy.init(args=args)
    node = FleetManagementSystem()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
