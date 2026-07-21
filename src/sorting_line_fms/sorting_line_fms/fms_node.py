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
import sys

# fleet_config.py는 이 ROS 패키지 밖(isaacpjt/sorting_line/)에 있는 순수 데이터
# 모듈이다 — Isaac Sim 스크립트도 rclpy 없이 그대로 가져다 쓰므로 일부러 패키지
# 안으로 옮기지 않았다. setup.py의 data_files로 share/sorting_line_fms/ 밑에
# 같이 설치해두고, ament_index로 그 경로를 찾는다 — __file__ 기준 상대경로
# 계산과 달리 --symlink-install 여부와 무관하게 항상 정확하게 동작한다.
from ament_index_python.packages import get_package_share_directory

_SHARE_DIR = get_package_share_directory("sorting_line_fms")
if _SHARE_DIR not in sys.path:
    sys.path.insert(0, _SHARE_DIR)

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String

from sorting_line_interfaces.srv import AmrState, ShoesList

from fleet_config import (
    NODE_GRAPH,
    PICKUP_NODE,
    PICKUP_WAIT_NODE,
    ROBOT_HOME_NODE,
    ROBOT_SHOE_TYPE,
    ROBOTS_PER_TYPE,
    SHOE_TYPES,
    TARGET_SLOT_NODE,
    shortest_path,
)

MOVE_TIMEOUT_SEC = 15.0  # 이동 명령 후 이만큼 지나도 arrived가 없으면 멈춘 것으로 간주
DEADLOCK_ALERT_SEC = 5.0  # 다른 로봇이 다음 노드를 점유해 이만큼 계속 못 넘어가면 교착으로 보고
SHOE_PLACEMENT_DWELL_SEC = 3.0  # 랙 목표 지점 도착 후 신발을 내려놓는 가상 대기시간(임시값, 조절 가능)

# 메인 컨트롤 노드와 맞춰야 하는 서비스 이름 — 실제 이름이 다르면 여기만 바꾸면 된다.
SHOES_LIST_SERVICE_NAME = "/fms/shoes_list"
AMR_STATE_SERVICE_NAME = "/main_control/amr_state"

# AmrState.srv의 state 문자열 값 — 실제 메인 컨트롤 노드가 기대하는 값으로 조정 가능.
AMR_STATE_COMPLETE = "완료"
AMR_STATE_DEADLOCK = "교착"

# 신발 길이(mm) → 크기 등급 경계값. 실측 후 조정 예정(현재는 자리표시용 플레이스홀더).
SIZE_THRESHOLDS_MM = (255, 275)  # length < 255 → 소 | 255 ~ 275 → 중 | length > 275 → 대


def _bucket_size(length_mm):
    low, high = SIZE_THRESHOLDS_MM
    if length_mm < low:
        return "소"
    if length_mm <= high:
        return "중"
    return "대"


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4):
    """두 선분(p1→p2, p3→p4)이 (x, y) 평면상에서 교차하는지 판정한다.

    노드 락(뮤텍스)은 "다음 노드"라는 점(point)만 예약할 뿐, 그 점까지 가는
    직선 구간(edge) 자체는 아무도 보호하지 않는다 — 두 로봇이 서로 다른(둘 다
    비어있는) 노드로 향하더라도, 그 사이 경유지-경유지 구간이 기하학적으로
    교차하면 정확히 그 교차 지점에서 부딪힐 수 있다. 이 함수는 그 위험을
    감지하기 위한 순수 기하 판정(CCW 기반 원안-교차 테스트)이다.
    """
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


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
        # 종류별로 "아직 랙에 안착 안 한(대기 중+진행 중) 신발 수" — 배치가 언제
        # 완전히 끝나는지 판정하는 용도. ShoesList.srv로 배치가 들어올 때 늘고,
        # 로봇이 배치 대기(dwelling)를 마칠 때마다 하나씩 줄어든다. 0이 되는
        # 순간이 "이 종류의 배치가 전부 끝났다"는 뜻이다.
        self._pending_batch_count = {shoe_type: 0 for shoe_type in SHOE_TYPES}

        self.command_pub = self.create_publisher(String, "/fms/commands", 10)
        # 담당 종류의 로봇이 PICKUP에 도착해 신발을 받을 준비가 됐음을 시뮬레이션
        # 구동 파일에 알리는 신호 — 실제 신발 프림을 옮기는 동작은 그 파일이 담당한다.
        self.pickup_ready_pub = self.create_publisher(String, "/fms/pickup_ready", 10)
        # 로봇 동선이 겹쳐 한쪽이 오래 대기하게 되면(교착 상태) 알리는 토픽 —
        # 지금은 구독자가 없어도, 나중에 만들 상위 대시보드 노드가 그대로 구독하면 됨.
        self.deadlock_pub = self.create_publisher(String, "/fms/deadlock_alert", 10)
        self.create_subscription(String, "/amr/status", self._on_robot_status, 10)

        # 메인 컨트롤 노드 → FMS: 신발 종류 하나에 대한 배치(길이 리스트) 작업 지시.
        # 예전에는 FMS가 비전 서비스를 직접 트리거·호출했지만, 이제는 메인 컨트롤
        # 노드가 비전 분류 결과를 정리해서 이 서비스로 바로 넘겨준다 — FMS는 더
        # 이상 비전 쪽과 직접 통신하지 않는다.
        self.create_service(ShoesList, SHOES_LIST_SERVICE_NAME, self._on_shoes_list)
        # FMS → 메인 컨트롤 노드: 배치 작업 완료 / 교착 상태 발생 알림 (응답 불필요).
        self.amr_state_client = self.create_client(AmrState, AMR_STATE_SERVICE_NAME)

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
            "dwell_until": None,
            "batch_complete_target": None,  # 배치 완료로 PICKUP/PICKUP_WAIT로 향하는 중이면 그 목표
        }
        self.node_locks[home_node] = robot_id
        self.get_logger().info(f"[{robot_id}] 등록 완료 (담당 종류={shoe_type}, 홈 슬롯={home_node})")
        return True

    def _on_shoes_list(self, request, response):
        if not (0 <= request.shoes_num < len(SHOE_TYPES)):
            self.get_logger().error(f"알 수 없는 shoes_num: {request.shoes_num}")
            return response
        shoe_type = SHOE_TYPES[request.shoes_num]

        for length_mm in request.shoes_length:
            shoe_size = _bucket_size(length_mm)
            target_node = TARGET_SLOT_NODE[(shoe_type, shoe_size)]
            self._next_shoe_id += 1
            self.task_queues[shoe_type].append({"shoe_id": self._next_shoe_id, "target_node": target_node})
            self.get_logger().info(
                f"[신규 태스크] shoe_id={self._next_shoe_id} 길이={length_mm}mm({shoe_size}) "
                f"→ {target_node} (담당 종류={shoe_type})"
            )
        self._pending_batch_count[shoe_type] += len(request.shoes_length)
        return response

    def _on_robot_status(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]
        now = self.get_clock().now()

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
            if arrived_node == PICKUP_NODE.get(robot["shoe_type"]) and robot["task"] is not None:
                self._publish_pickup_ready(robot_id, robot)

            if robot["path_idx"] >= len(robot["path"]) - 1:
                # 목적지 도착 완료
                if robot["task"] is not None:
                    # 랙의 목표 지점에 도착 — 실제로 옮기지는 않지만, 신발을 내려놓는
                    # 가상의 시간만큼 여기 머무른다. 이 동안 이 노드(및 근접 통로의
                    # 경우 "대" 체크포인트를 이미 지나왔다면 그 지점)는 계속 락이
                    # 걸려있어, 같은 랙으로 오는 다른 로봇이 근접/우회를 판단할 때
                    # 반영된다.
                    robot["state"] = "dwelling"
                    robot["dwell_until"] = now + Duration(seconds=SHOE_PLACEMENT_DWELL_SEC)
                    robot["task"] = None
                    robot["path"] = []
                    robot["path_idx"] = 0
                else:
                    # 복귀 완료 — 홈 슬롯이든, 배치가 다 끝나 PICKUP/PICKUP_WAIT로
                    # 돌아온 것이든 여기서 처리된다.
                    robot["state"] = "idle"
                    robot["path"] = []
                    robot["path_idx"] = 0
                    if robot["batch_complete_target"] is not None and arrived_node == robot["batch_complete_target"]:
                        robot["batch_complete_target"] = None
                        self.get_logger().info(
                            f"[{robot_id}] 배치 작업 완료 — {arrived_node}에서 다음 배치 대기"
                        )
                        self._report_amr_state(AMR_STATE_COMPLETE)
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

        # 0.5) 배치 대기시간(dwelling) 처리 — 신발을 내려놓는 가상 시간이 지나면 복귀 시작.
        #      그 시간 동안은 도착 노드의 락이 계속 걸려있어, 같은 랙으로 오는 다른
        #      로봇의 근접/우회 판단(_pick_rack_target)에 자연스럽게 반영된다.
        for robot_id, robot in self.robots.items():
            if robot["state"] != "dwelling" or robot["dwell_until"] is None:
                continue
            if now < robot["dwell_until"]:
                continue
            robot["dwell_until"] = None
            arrived_node = robot["current_node"]  # 배치 대기는 항상 랙의 목표 지점에서 시작한다
            shoe_type = robot["shoe_type"]
            self._pending_batch_count[shoe_type] = max(0, self._pending_batch_count[shoe_type] - 1)
            batch_done = self._pending_batch_count[shoe_type] == 0

            # 이 종류의 배치가 아직 안 끝났으면 홈 슬롯으로, 다 끝났으면 PICKUP(또는
            # 이미 점유돼 있으면 PICKUP_WAIT)으로 복귀한다 — 다음 배치를 바로 받을 수
            # 있게 픽업 근처에서 대기.
            target = self._pick_pickup_target(shoe_type) if batch_done else robot["home_node"]
            robot["batch_complete_target"] = target if batch_done else None
            self.get_logger().info(f"[{robot_id}] 배치 대기 종료. {target}(으)로 이동합니다.")
            return_path = shortest_path(arrived_node, target)
            if return_path:
                robot["path"] = return_path
                robot["path_idx"] = 0
                robot["state"] = "waiting_next_hop"
            else:
                self.get_logger().error(f"[{robot_id}] 복귀 경로 없음: {arrived_node} → {target}")
                robot["state"] = "idle"

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
                # 근접 통로의 첫 번째 저장소(대)를 아직 아무도 지나지 않은 상태(=락이
                # 비어있음)면 근접, 누가 아직 거기 있으면(이동 중이거나 배치 대기 중)
                # 우회로 배정한다.
                actual_target = self._pick_rack_target(task["target_node"])
                best_robot_id, best_path = None, None
                for robot_id, robot in candidates:
                    path = shortest_path(robot["current_node"], actual_target)
                    if path is None:
                        continue
                    if best_path is None or len(path) < len(best_path):
                        best_robot_id, best_path = robot_id, path

                if best_robot_id is None:
                    self.get_logger().error(f"경로 불가: {actual_target} (담당 종류={shoe_type})")
                    break

                queue.pop(0)
                robot = self.robots[best_robot_id]
                task["target_node"] = actual_target
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

            if holder_id is not None:
                # 다른 로봇이 다음 노드를 점유 중이라 못 넘어감 — 얼마나 오래 막혀있는지
                # 추적하다가 DEADLOCK_ALERT_SEC를 넘기면 한 번만 알림을 보낸다.
                self._track_deadlock_alert(robot_id, next_node, holder_id, now, reason="node_lock")
                continue

            # 다음 노드는 비어있어도, 거기까지 가는 구간(edge) 자체가 지금 이동 중인
            # 다른 로봇의 구간과 기하학적으로 교차할 수 있다 — 노드 락은 점만 보호하고
            # 구간은 안 보호하기 때문. 교차 위험이 있으면 이번 tick은 보류하고 다음
            # tick에 다시 시도한다(그동안 상대가 지나가면 저절로 풀린다).
            conflicting_id = self._find_edge_conflict(robot_id, robot["current_node"], next_node)
            if conflicting_id is not None:
                self._track_deadlock_alert(robot_id, next_node, conflicting_id, now, reason="edge_conflict")
                continue

            # 다음 노드가 비어있고 구간 충돌 위험도 없으면 점유 후 이동 명령 전송!
            self.node_locks[next_node] = robot_id  # 락 걸기
            robot["state"] = "moving"
            robot["move_started_at"] = now
            robot["move_target"] = next_node
            self._send_move_command(robot_id, next_node)
            self._clear_deadlock_alert(robot_id, now)

    def _pick_rack_target(self, canonical_target):
        """canonical_target(예: RackA_소)에 대해 근접/우회 중 실제로 쓸 노드를 고른다.

        근접 통로의 첫 번째 저장소(대)가 비어있으면(아직 아무도 안 왔거나, 이미 다
        지나갔으면) 근접 그대로 쓴다. 누군가 아직 그 자리를 지나지 않았다면(이동
        중이거나 배치 대기 중이라 락이 걸려있음) 우회 통로의 같은 사이즈 노드를
        대신 돌려준다. 실제 목표가 무엇이든(소/중/대) 판단 기준은 항상 "대"
        하나다 — 대→중→소 순서로만 진입하기 때문에 대가 뚫려있어야 그 뒤도 갈 수 있다.
        """
        rack_prefix = canonical_target.split("_")[0]  # "RackA_소" → "RackA"
        near_first_checkpoint = f"{rack_prefix}_대"
        if self.node_locks.get(near_first_checkpoint) is None:
            return canonical_target
        return f"{canonical_target}_우회"

    def _pick_pickup_target(self, shoe_type):
        """배치를 다 끝낸 로봇이 향할 곳 — 자기 종류 PICKUP_X가 비어있으면
        PICKUP_X, 이미 같은 종류 파트너 로봇이 있으면 그 바로 뒤의
        PICKUP_WAIT_X에서 대기한다."""
        pickup_node = PICKUP_NODE[shoe_type]
        if self.node_locks.get(pickup_node) is None:
            return pickup_node
        return PICKUP_WAIT_NODE[shoe_type]

    def _report_amr_state(self, state):
        if not self.amr_state_client.service_is_ready():
            self.get_logger().warn(f"AmrState 서비스가 아직 준비되지 않아 상태 보고를 건너뜀 (state={state})")
            return
        request = AmrState.Request()
        request.state = state
        self.amr_state_client.call_async(request)  # 응답이 없는 알림이라 콜백은 필요 없음

    def _find_edge_conflict(self, robot_id, from_node, to_node):
        """from_node→to_node 구간이 현재 이동 중인 다른 로봇의 구간과 교차하는지 확인.

        교차하는 첫 번째 상대 robot_id를 돌려주고, 없으면 None. 시간까지 정밀하게
        시뮬레이션하지는 않는다 — "같은 순간 이동 중"이라는 조건만으로 보수적으로
        판단한다(약간 과도하게 대기시킬 수는 있어도, 놓쳐서 실제로 부딪히는 것보다
        안전한 쪽을 택한다).
        """
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

    def _track_deadlock_alert(self, robot_id, next_node, holder_id, now, reason):
        entry = self._blocked_since.get(robot_id)
        if entry is None:
            self._blocked_since[robot_id] = {
                "since": now, "next_node": next_node, "holder": holder_id,
                "reason": reason, "reported": False,
            }
            return

        # 막고 있는 대상/사유가 바뀌었을 수도 있으니 최신 정보로 갱신하되, 시작 시각은 유지
        entry["next_node"] = next_node
        entry["holder"] = holder_id
        entry["reason"] = reason
        if entry["reported"]:
            return

        elapsed_sec = (now - entry["since"]).nanoseconds / 1e9
        if elapsed_sec >= DEADLOCK_ALERT_SEC:
            entry["reported"] = True
            self._publish_deadlock_alert("raised", robot_id, next_node, holder_id, reason, elapsed_sec)

    def _clear_deadlock_alert(self, robot_id, now):
        entry = self._blocked_since.pop(robot_id, None)
        if entry is not None and entry["reported"]:
            elapsed_sec = (now - entry["since"]).nanoseconds / 1e9
            self._publish_deadlock_alert(
                "cleared", robot_id, entry["next_node"], entry["holder"], entry["reason"], elapsed_sec
            )

    def _publish_deadlock_alert(self, event, robot_id, next_node, holder_id, reason, elapsed_sec):
        msg = String()
        msg.data = json.dumps({
            "event": event,  # "raised" | "cleared"
            "robot_id": robot_id,
            "blocked_at_node": next_node,
            "blocked_by_robot": holder_id,
            "reason": reason,  # "node_lock"(다음 노드를 점유당함) | "edge_conflict"(구간 교차 위험)
            "blocked_seconds": round(elapsed_sec, 1),
        })
        self.deadlock_pub.publish(msg)
        if event == "raised":
            if reason == "edge_conflict":
                self.get_logger().warn(
                    f"[교착 감지] {robot_id} — {next_node} 진입 구간이 {holder_id}의 이동 구간과 "
                    f"겹칠 위험 있어 대기 ({elapsed_sec:.1f}초째 정체)"
                )
            else:
                self.get_logger().warn(
                    f"[교착 감지] {robot_id} — {next_node} 진입 대기 중, {holder_id}가 점유 "
                    f"({elapsed_sec:.1f}초째 정체)"
                )
            self._report_amr_state(AMR_STATE_DEADLOCK)
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
