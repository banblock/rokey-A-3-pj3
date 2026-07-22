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
from std_srvs.srv import Trigger

from recycle_interfaces.srv import AmrState, ShoesList

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

MOVE_TIMEOUT_SEC = 60.0  # 이동 명령 후 이만큼 지나도 arrived가 없으면 멈춘 것으로 간주.
# 15초였던 값을 60초로 올렸다 — 그래프를 2.5배 스케일한 뒤로 세분화 안 된
# 일부 간선(특히 RackX_OUT → WAIT_N 복귀 구간)이 최대 22m(0.6m/s로 36.9초)까지
# 길어져서, 정상 이동 중인데도 15초 만에 워치독이 "멈췄다"고 오판해 락을
# 강제로 풀고 재전송하는 문제가 있었다. 최악 이동 시간(약 37초)에 회전 시간
# 등 여유를 더해 60초로 재보정.
DEADLOCK_ALERT_SEC = 5.0  # 다른 로봇이 다음 노드를 점유해 이만큼 계속 못 넘어가면 일단 "감지"로 보고
DEADLOCK_UNRECOVERED_SEC = 25.0  # 감지 이후에도 이만큼 더 못 풀리면 "복구 실패"로 보고 위치별 카운트를 올림
DEADLOCK_NOTIFY_THRESHOLD = 3  # 같은 위치(node) 또는 같은 로봇에서 "복구 실패"가 이 횟수 이상 반복돼야 메인 컨트롤 노드에 알림
SHOE_PLACEMENT_DWELL_SEC = 3.0  # 랙 목표 지점 도착 후 신발을 내려놓는 가상 대기시간(임시값, 조절 가능)

# AmrState.srv의 code 값: 0 = 완료, 1 = 교착(같은 로봇이 반복, DEADLOCK_NOTIFY_THRESHOLD회 이상),
# 2 = 교착(같은 위치가 반복, DEADLOCK_NOTIFY_THRESHOLD회 이상) — 둘 다 해당하면 각각 따로 보고된다.

# 신발 길이(mm) → 크기 등급 경계값. 실측 후 조정 예정(현재는 자리표시용 플레이스홀더).
# 등급 라벨 자체를 실제 길이(mm) 문자열로 쓴다(280=대, 260=중, 240=소) — 노드
# 이름(RackA_280 등)도 전부 이 값과 동일하게 맞춰뒀다.
SIZE_THRESHOLDS_MM = (255, 275)  # length < 255 → 240 | 255 ~ 275 → 260 | length > 275 → 280

# 선반 번호(0~11) — A280,A260,A240,B280,B260,B240,...,D280,D260,D240 순서로 고정 배정.
# /fms/amr_carrying이 보내는 선반 번호가 바로 이 인덱스다. 근접/detour는 물리적으로
# 같은 선반이라 detour 노드명("RackA_240_detour")도 접미사를 떼면 같은 인덱스를 가리킨다.
_RACK_SIZES_IN_ORDER = ["280", "260", "240"]
SHELF_INDEX = {
    f"Rack{t}_{s}": i
    for i, (t, s) in enumerate(
        (t, s) for t in SHOE_TYPES for s in _RACK_SIZES_IN_ORDER
    )
}


def _canonical_rack_node(node_id):
    """detour 노드명("RackA_240_detour")도 근접 노드명("RackA_240")과 같은 선반을 가리키므로,
    선반 번호를 찾기 전에 접미사를 떼어 정규화한다."""
    return node_id[: -len("_detour")] if node_id.endswith("_detour") else node_id


def _bucket_size(length_mm):
    low, high = SIZE_THRESHOLDS_MM
    if length_mm < low:
        return "240"
    if length_mm <= high:
        return "260"
    return "280"


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
        # "복구 실패" 확정 횟수를 위치 기준/로봇 기준으로 각각 따로 누적한다 — 같은
        # 자리가 반복 병목인지(코드 2), 특정 로봇이 반복적으로 걸리는지(코드 1)를
        # 구분해서 메인 컨트롤 노드에 알리기 위함.
        self._deadlock_count_by_node = {}  # node_id -> 그 위치 누적 "복구 실패" 횟수
        self._deadlock_count_by_robot = {}  # robot_id -> 그 로봇의 누적 "복구 실패" 횟수
        # 종류별로 "아직 랙에 안착 안 한(대기 중+진행 중) 신발 수" — 배치가 언제
        # 완전히 끝나는지 판정하는 용도. ShoesList.srv로 배치가 들어올 때 늘고,
        # 로봇이 배치 대기(dwelling)를 마칠 때마다 하나씩 줄어든다. 0이 되는
        # 순간이 "이 종류의 배치가 전부 끝났다"는 뜻이다.
        self._pending_batch_count = {shoe_type: 0 for shoe_type in SHOE_TYPES}
        # AMR이 PICKUP에 도착할 때마다 그 신발의 shoe_id를 순서대로 쌓아두는 큐 —
        # /fms/amr_ready(Trigger)가 호출될 때마다 하나씩 꺼내 응답한다. Trigger는
        # 요청에 아무 필드가 없어서 "어떤 로봇이냐"를 요청 쪽에서 지정할 수 없으므로,
        # 도착한 순서대로 하나씩 내어주는 FIFO 큐로 처리한다.
        self._ready_shoe_queue = []
        # /fms/amr_carrying(Trigger)이 꺼내갈 큐 — AMR이 저장소(280/260/240)에
        # 도착할 때마다 쌓인다. 선반별 누적 박스 수도 같이 추적한다(정규화된
        # 노드명 기준 — 근접/detour는 같은 선반이므로 카운트를 공유해야 한다).
        self._carrying_queue = []
        self._shelf_box_count = {node: 0 for node in SHELF_INDEX}

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
        self.create_service(ShoesList, "/fms/shoes_list", self._on_shoes_list)
        # FMS → 메인 컨트롤 노드: 배치 작업 완료 / 교착 상태 발생 알림 (응답 불필요).
        self.amr_state_client = self.create_client(AmrState, "/main_control/amr_state")
        # 시뮬레이션 쪽에서 "AMR이 PICKUP에 도착했다 → 어떤 신발을 올려야 하는지"를
        # 물어보는 서비스. std_srvs/Trigger라 요청엔 필드가 없고 응답도 success(bool)/
        # message(string)뿐이라, 신발 번호(shoe_id)는 message에 문자열로 담아 보낸다.
        self.create_service(Trigger, "/fms/amr_ready", self._on_amr_ready)
        # 시뮬레이션 쪽에서 "AMR이 저장소(280/260/240)에 도착해 신발을 내려놓았다"를
        # 물어보는 서비스 — /fms/amr_ready와 동일하게 Trigger + message에 JSON.
        self.create_service(Trigger, "/fms/amr_carrying", self._on_amr_carrying)

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
                    # 걸려있어, 같은 랙으로 오는 다른 로봇이 근접/detour를 판단할 때
                    # 반영된다.
                    self._publish_amr_carrying(robot_id, arrived_node)
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
                        self._report_amr_state(robot_id, 0, f"{arrived_node} 배치 완료")
            else:
                robot["state"] = "waiting_next_hop"
                # 다음 tick(최대 0.5초 뒤)까지 안 기다리고 도착한 즉시 다음 홉을
                # 시도한다 — 중간 경유지에서 로봇이 잠깐 멈칫하던 것의 상당 부분이
                # 실제로는 "도착 → 다음 명령까지의 대기 시간(tick 주기)"였다.
                self._try_advance_hop(robot_id, now)

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
        #      로봇의 근접/detour 판단(_pick_rack_target)에 자연스럽게 반영된다.
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

            # 배치가 끝났든 안 끝났든, 복귀 목적지는 항상 PICKUP(또는 이미
            # 점유돼 있으면 PICKUP_WAIT)이다 — 로봇이 쉴 때 가는 곳은 결국
            # 이 둘뿐이라 별도의 "홈 슬롯" 개념 없이 그때그때 점유 상태만 보고
            # 정한다. batch_done일 때만 도착 시 완료 보고(AMR_STATE_COMPLETE)를
            # 하도록 표시해둔다.
            target = self._pick_pickup_target(shoe_type)
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

        # 1) 태스크 배정 — 종류가 일치하는 idle 로봇 중 번호가 앞선 로봇을 우선
        # 배정한다. 거리 비교 대신 번호 순으로 하는 이유: 같은 종류를 담당하는
        # 로봇들이 대개 같은 허브/랙 구조를 거쳐가서 경로의 실제 목적지 도달
        # 가능 여부는 다 똑같고, 거리 차이는 로봇 홈 슬롯을 앞뒤로 벌려둔
        # 배치상의 부산물일 뿐 실제로 최적화할 대상이 아니다. 번호 순 방식은
        # 코드가 단순하고, 담당 로봇 수가 2대보다 늘어나도 그대로 확장된다.
        # 로봇 등록 순서(=self.robots 삽입 순서)는 DDS 디스커버리 타이밍에 따라
        # 달라질 수 있어 보장이 없으므로, 후보를 로봇 번호로 명시적으로 정렬한다.
        for shoe_type, queue in self.task_queues.items():
            while queue:
                candidates = sorted(
                    (
                        (robot_id, robot)
                        for robot_id, robot in self.robots.items()
                        if robot["state"] == "idle" and robot["shoe_type"] == shoe_type
                    ),
                    key=lambda item: int(item[0].split("_")[1]),
                )
                if not candidates:
                    break

                task = queue[0]
                # 근접 통로의 첫 번째 저장소(280)를 아직 아무도 지나지 않은 상태(=락이
                # 비어있음)면 근접, 누가 아직 거기 있으면(이동 중이거나 배치 대기 중)
                # detour로 배정한다.
                actual_target = self._pick_rack_target(task["target_node"])
                best_robot_id, best_path = None, None
                for robot_id, robot in candidates:
                    path = shortest_path(robot["current_node"], actual_target)
                    if path is not None:
                        best_robot_id, best_path = robot_id, path
                        break  # 번호가 가장 앞선 idle 로봇을 그대로 채택

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
            self._try_advance_hop(robot_id, now)

    def _try_advance_hop(self, robot_id, now):
        """robot_id를 한 홉 전진시킨다 — 다음 노드가 비어있고 구간 충돌 위험도
        없으면 점유 후 이동 명령을 보낸다. 주기 tick(_dispatch_tick)과, 로봇이
        막 도착해 다음 목표가 필요해진 순간(반응형, _on_robot_status) 양쪽에서
        호출된다 — 도착 즉시 시도해야 "다음 tick까지 대기(최대 0.5초)"로 인한
        중간 경유지에서의 순간 정지를 없앨 수 있다.
        """
        robot = self.robots[robot_id]
        if robot["state"] != "waiting_next_hop":
            return

        next_idx = robot["path_idx"] + 1
        if next_idx >= len(robot["path"]):
            return

        next_node = robot["path"][next_idx]

        # 근접 통로의 "280"(대) 체크포인트로 들어가려는 참인데 이미 다른 로봇이
        # 거기 있다면(태스크 배정 시점엔 비어있어서 근접으로 나왔지만, 그 사이
        # 파트너가 먼저 들어온 경우) — 배정 당시 판단은 이미 낡았으니 무작정
        # 기다리지 말고 지금(HUB 등에서) detour로 즉시 재경로한다. _pick_rack_target은
        # 배정 시점 스냅샷 판단이라 이동 중 상황 변화를 못 따라가는 게 근본 원인.
        if (
            next_node.startswith("Rack") and next_node.endswith("_280")
            and self.node_locks.get(next_node) not in (None, robot_id)
            and robot["task"] is not None
            and not robot["task"]["target_node"].endswith("_detour")
        ):
            detour_target = f"{robot['task']['target_node']}_detour"
            detour_path = shortest_path(robot["current_node"], detour_target)
            if detour_path is not None:
                self.get_logger().info(
                    f"[{robot_id}] {next_node} 점유 중 → detour로 재경로 "
                    f"({robot['task']['target_node']} → {detour_target})"
                )
                robot["task"]["target_node"] = detour_target
                robot["path"] = detour_path
                robot["path_idx"] = 0
                next_idx = 1
                next_node = robot["path"][next_idx]

        holder_id = self.node_locks[next_node]

        if holder_id is not None:
            # 다른 로봇이 다음 노드를 점유 중이라 못 넘어감 — 얼마나 오래 막혀있는지
            # 추적하다가 DEADLOCK_ALERT_SEC를 넘기면 한 번만 알림을 보낸다.
            self._track_deadlock_alert(robot_id, next_node, holder_id, now, reason="node_lock")
            return

        # 다음 노드는 비어있어도, 거기까지 가는 구간(edge) 자체가 지금 이동 중인
        # 다른 로봇의 구간과 기하학적으로 교차할 수 있다 — 노드 락은 점만 보호하고
        # 구간은 안 보호하기 때문. 교차 위험이 있으면 이번 시도는 보류하고 다음
        # tick에 다시 시도한다(그동안 상대가 지나가면 저절로 풀린다).
        conflicting_id = self._find_edge_conflict(robot_id, robot["current_node"], next_node)
        if conflicting_id is not None:
            self._track_deadlock_alert(robot_id, next_node, conflicting_id, now, reason="edge_conflict")
            return

        # 다음 노드가 비어있고 구간 충돌 위험도 없으면 점유 후 이동 명령 전송!
        self.node_locks[next_node] = robot_id  # 락 걸기
        robot["state"] = "moving"
        robot["move_started_at"] = now
        robot["move_target"] = next_node
        # "경로의 마지막 노드일 때만 정지"로 하면, 최종 목표가 RackA_240일 때
        # RackA_280/RackA_260처럼 세분화 칸이 아닌 진짜 노드까지 감속 없이
        # 통과해버린다(실제로 이 버그로 랙 280/260 위치에서 안 멈추는 문제가
        # 있었다). 정지 여부는 "세분화로 생긴 가짜 칸인지"로 판단해야 한다 —
        # _subdivide_edge가 만드는 칸 이름은 항상 "__"(이중 언더스코어)를
        # 포함하고, 실제 노드 이름(PICKUP_A, HUB_A, RackA_240_detour 등)은 전부
        # 단일 언더스코어만 쓰므로 이 표시로 구분할 수 있다.
        is_final_hop = "__" not in next_node
        self._send_move_command(robot_id, next_node, is_final_hop)
        self._clear_deadlock_alert(robot_id, now)

    def _pick_rack_target(self, canonical_target):
        """canonical_target(예: RackA_240)에 대해 근접/detour 중 실제로 쓸 노드를 고른다.

        근접 통로의 첫 번째 저장소(280)가 비어있으면(아직 아무도 안 왔거나, 이미 다
        지나갔으면) 근접 그대로 쓴다. 누군가 아직 그 자리를 지나지 않았다면(이동
        중이거나 배치 대기 중이라 락이 걸려있음) detour 통로의 같은 사이즈 노드를
        대신 돌려준다. 실제 목표가 무엇이든(240/260/280) 판단 기준은 항상 "280"
        하나다 — 280→260→240 순서로만 진입하기 때문에 280이 뚫려있어야 그 뒤도 갈 수 있다.
        """
        rack_prefix = canonical_target.split("_")[0]  # "RackA_240" → "RackA"
        near_first_checkpoint = f"{rack_prefix}_280"
        if self.node_locks.get(near_first_checkpoint) is None:
            return canonical_target
        return f"{canonical_target}_detour"

    def _pick_pickup_target(self, shoe_type):
        """배치를 다 끝낸 로봇이 향할 곳 — 자기 종류 PICKUP_X가 비어있으면
        PICKUP_X, 이미 같은 종류 파트너 로봇이 있으면 그 바로 뒤의
        PICKUP_WAIT_X에서 대기한다."""
        pickup_node = PICKUP_NODE[shoe_type]
        if self.node_locks.get(pickup_node) is None:
            return pickup_node
        return PICKUP_WAIT_NODE[shoe_type]

    def _report_amr_state(self, robot_id, code, desc=""):
        if not self.amr_state_client.service_is_ready():
            self.get_logger().warn(f"AmrState 서비스가 아직 준비되지 않아 상태 보고를 건너뜀 (code={code})")
            return
        request = AmrState.Request()
        # robot_id는 "amr_3" 같은 내부 문자열 식별자(등록 순서와 무관하게 fleet_config의
        # ROBOT_SHOE_TYPE에 고정된 이름)라서, 메인 컨트롤 노드가 기대하는 int32 amr_id로
        # 보내려면 뒤의 숫자만 뽑아내야 한다.
        request.amr_id = int(robot_id.split("_")[1])
        request.code = code
        request.state = desc  # 사람이 읽을 문제 상황 설명(선택)
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

    def _send_move_command(self, robot_id, node_id, is_final_hop):
        position = list(NODE_GRAPH[node_id]["position"])
        msg = String()
        msg.data = json.dumps({
            "robot_id": robot_id,
            "action": "move_to_node",
            "node_id": node_id,
            "position": position,
            # 경로의 마지막 홉인지 — fleet_driver가 도착 시 완전히 정지할지,
            # 아니면 중간 경유지라 다음 명령이 올 때까지 그대로 지나갈지 결정하는 데 씀.
            "is_final_hop": is_final_hop,
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
        # /fms/amr_ready(Trigger)가 꺼내갈 큐 — 여기서 말하는 "신발 번호"는
        # ShoesList.srv의 shoes_num과 같은 종류 인덱스(A=0,B=1,C=2,D=3)이지
        # task 내부의 순번(shoe_id)이 아니다. 도착한 AMR의 실제 번호(robot_id
        # "amr_3" → 3, _report_amr_state와 동일한 방식)도 같이 묶어 보낸다.
        self._ready_shoe_queue.append({
            "amr_id": int(robot_id.split("_")[1]),
            "shoes_num": SHOE_TYPES.index(robot["shoe_type"]),
        })
        self.get_logger().info(
            f"[{robot_id}] PICKUP 도착 — 신발 수령 준비 완료 (shoe_type={robot['shoe_type']}, "
            f"shoe_id={task['shoe_id']})"
        )

    def _on_amr_ready(self, request, response):
        if self._ready_shoe_queue:
            ready = self._ready_shoe_queue.pop(0)
            response.success = True
            response.message = json.dumps(ready)
        else:
            response.success = False
            response.message = ""
        return response

    def _publish_amr_carrying(self, robot_id, arrived_node):
        canonical_node = _canonical_rack_node(arrived_node)
        shelf_num = SHELF_INDEX[canonical_node]
        self._shelf_box_count[canonical_node] += 1
        self._carrying_queue.append({
            "amr_id": int(robot_id.split("_")[1]),
            "shelf_num": shelf_num,
            "box_count": self._shelf_box_count[canonical_node],
        })
        self.get_logger().info(
            f"[{robot_id}] {arrived_node}(선반 {shelf_num}) 도착 — 누적 박스 수 "
            f"{self._shelf_box_count[canonical_node]}"
        )

    def _on_amr_carrying(self, request, response):
        if self._carrying_queue:
            carrying = self._carrying_queue.pop(0)
            response.success = True
            response.message = json.dumps(carrying)
        else:
            response.success = False
            response.message = ""
        return response

    def _track_deadlock_alert(self, robot_id, next_node, holder_id, now, reason):
        entry = self._blocked_since.get(robot_id)
        if entry is None:
            self._blocked_since[robot_id] = {
                "since": now, "next_node": next_node, "holder": holder_id,
                "reason": reason, "reported": False, "unrecovered_reported": False,
            }
            return

        # 막고 있는 대상/사유가 바뀌었을 수도 있으니 최신 정보로 갱신하되, 시작 시각은 유지
        entry["next_node"] = next_node
        entry["holder"] = holder_id
        entry["reason"] = reason

        elapsed_sec = (now - entry["since"]).nanoseconds / 1e9

        if not entry["reported"] and elapsed_sec >= DEADLOCK_ALERT_SEC:
            entry["reported"] = True
            self._publish_deadlock_alert("raised", robot_id, next_node, holder_id, reason, elapsed_sec)

        # "감지(raised)"만으로는 아직 진짜 교착인지 알 수 없다 — 잠깐 정체됐다가 곧
        # 스스로 풀리는 경우가 대부분이라, 그보다 훨씬 긴 시간(DEADLOCK_UNRECOVERED_SEC)이
        # 지나도록 여전히 같은 자리에서 못 벗어나야만 "복구 실패"로 확정하고, 그때
        # 비로소 이 위치의 누적 카운트를 올린다 — 곧 풀릴 정체까지 카운트에 넣으면
        # 실제로는 문제없는 위치도 금방 메인 컨트롤 알림 임계치를 넘겨버린다.
        if not entry["unrecovered_reported"] and elapsed_sec >= DEADLOCK_UNRECOVERED_SEC:
            entry["unrecovered_reported"] = True
            # 위치 기준/로봇 기준을 각각 독립적으로 누적한다 — 같은 자리가 반복
            # 병목인지, 특정 로봇이 반복적으로 걸리는지는 서로 다른 원인이라 따로
            # 세고 따로 알린다.
            node_deadlock_count = self._deadlock_count_by_node.get(next_node, 0) + 1
            self._deadlock_count_by_node[next_node] = node_deadlock_count
            robot_deadlock_count = self._deadlock_count_by_robot.get(robot_id, 0) + 1
            self._deadlock_count_by_robot[robot_id] = robot_deadlock_count
            self._publish_deadlock_alert(
                "unrecovered", robot_id, next_node, holder_id, reason, elapsed_sec,
                node_deadlock_count, robot_deadlock_count,
            )

    def _clear_deadlock_alert(self, robot_id, now):
        entry = self._blocked_since.pop(robot_id, None)
        if entry is not None and entry["reported"]:
            elapsed_sec = (now - entry["since"]).nanoseconds / 1e9
            self._publish_deadlock_alert(
                "cleared", robot_id, entry["next_node"], entry["holder"], entry["reason"], elapsed_sec
            )

    def _publish_deadlock_alert(
        self, event, robot_id, next_node, holder_id, reason, elapsed_sec,
        node_deadlock_count=None, robot_deadlock_count=None,
    ):
        msg = String()
        msg.data = json.dumps({
            "event": event,  # "raised"(감지) | "unrecovered"(복구 실패 확정) | "cleared"(해소)
            "robot_id": robot_id,
            "blocked_at_node": next_node,
            "blocked_by_robot": holder_id,
            "reason": reason,  # "node_lock"(다음 노드를 점유당함) | "edge_conflict"(구간 교차 위험)
            "blocked_seconds": round(elapsed_sec, 1),
            # 아래 둘 다 "unrecovered"일 때만 값 있음 — 각각 위치 기준/로봇 기준 누적 횟수
            "node_deadlock_count": node_deadlock_count,
            "robot_deadlock_count": robot_deadlock_count,
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
        elif event == "unrecovered":
            self.get_logger().error(
                f"[교착 미복구] {robot_id} — {next_node}, {elapsed_sec:.1f}초째 미복구 "
                f"(이 위치 누적 {node_deadlock_count}회 / 이 로봇 누적 {robot_deadlock_count}회)"
            )
            # 위치 기준(코드 2)과 로봇 기준(코드 1)을 각각 독립적으로 임계치 검사한다
            # — 같은 자리가 반복 병목이면 위치 문제로, 특정 로봇이 반복 걸리면 로봇
            # 문제로 메인 컨트롤이 구분할 수 있게 서로 다른 code로 알린다. 매 tick마다
            # 알리면 노이즈가 많아 threshold 이상일 때만 보낸다.
            if robot_deadlock_count >= DEADLOCK_NOTIFY_THRESHOLD:
                self._report_amr_state(
                    robot_id, 1, f"{robot_id} 반복 교착 {robot_deadlock_count}회 (최근 위치: {next_node})"
                )
            if node_deadlock_count >= DEADLOCK_NOTIFY_THRESHOLD:
                self._report_amr_state(
                    robot_id, 2, f"{next_node} 반복 교착 {node_deadlock_count}회 (로봇: {robot_id})"
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
