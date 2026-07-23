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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from recycle_interfaces.msg import PickupList
from recycle_interfaces.srv import AmrState

from fleet_config import (
    NODE_GRAPH,
    PICKUP_NODE,
    PICKUP_WAIT_SLOTS,
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
SHOE_PLACEMENT_DWELL_SEC = 1.0  # 랙 목표 지점 도착 후 신발을 내려놓는 가상 대기시간(임시값, 조절 가능)

# AMR 준비/적재 이벤트(/fms/amr_ready, /fms/amr_carrying)용 QoS — 시뮬레이션과
# 유선으로 분리된 별도 컴퓨터에서 통신하고, 그 컴퓨터는 YOLO 추론까지 같이
# 돌리는 환경이라 "매 프레임 서비스로 물어보는" 폴링 대신 이벤트가 났을 때만
# 발행하는 토픽으로 바꿨다. RELIABLE(유실 시 재전송)은 depth만 지정해도 기본값
# 이지만, DURABILITY는 명시적으로 TRANSIENT_LOCAL로 줘야 한다 — 그래야 (1)
# 노드 디스커버리가 끝나기 전 찰나에 발행된 메시지, (2) 시뮬레이션이
# 크래시/재시작하는 동안 발행된 메시지도 재구독 시점에 그대로 받아갈 수 있다
# (기본값 VOLATILE은 이 두 경우 다 그냥 유실됨). 구독하는 쪽(시뮬레이션)도
# 반드시 같은 조합(RELIABLE + TRANSIENT_LOCAL)으로 맞춰야 정상 수신된다.
RELIABLE_EVENT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
)

MAX_SHOES_PER_TRIP = 5  # 로봇 1대가 한 번 나갈 때 실을 수 있는 최대 신발 수(실제 적재 용량)
# idle 로봇이 있어도 같은 종류 큐에 이 개수 이상 쌓이기 전에는 출발시키지 않는다
# — 원래는 idle 로봇+큐에 뭐라도 있으면 즉시(남은 만큼만이라도) 출발시켰는데,
# 그러면 신발이 3~4개만 들어와도 로봇이 바로 나가서 트립을 다 못 채우는 경우가
# 잦았다. MAX_SHOES_PER_TRIP과 값이 같아야 "항상 꽉 채운 트립만 나간다"가
# 되고, 더 작게 두면(예: 3) "3개 이상만 모이면 출발, 최대 5개까지만 싣는다"처럼
# 최소/최대를 따로 조절할 수 있다.
MIN_SHOES_TO_DISPATCH = 1
# 배치 안에서 여러 사이즈가 섞여 있으면 반드시 280→260→240 순서로 방문해야 한다
# — 랙 진입 통로가 일방통행이라 작은 사이즈부터 들르면 큰 사이즈로 되돌아갈 수 없다.
_SIZE_VISIT_ORDER = {"280": 0, "260": 1, "240": 2}

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
        # 실제 상태 메시지(특히 최초 "BOOT" idle 알림)가 DDS 디스커버리 타이밍에
        # 따라 유실될 수 있어서(launch 파일의 지연 시작도 이 위험을 줄일 뿐 완전히
        # 없애지는 못한다), 로봇이 실제로 보고해오길 기다리지 않고 config에 정의된
        # 모든 로봇을 시작하자마자 바로 등록한다 — 그래야 홈 슬롯이 처음부터
        # 확실하게 점유 상태로 잡혀서, 스폰 직후 상태를 다른 로봇이 "비어있다"고
        # 오판해 충돌하는 일이 없다(실제로 한 바퀴 돌고 온 로봇이 아직 스폰
        # 위치에 그대로 있는 다른 로봇과 충돌하는 문제가 있었다).
        for _robot_id in ROBOT_SHOE_TYPE:
            self.register_robot(_robot_id)
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
        # 완전히 끝나는지 판정하는 용도. PickupList 메시지로 배치가 들어올 때 늘고,
        # 로봇이 배치 대기(dwelling)를 마칠 때마다 하나씩 줄어든다. 0이 되는
        # 순간이 "이 종류의 배치가 전부 끝났다"는 뜻이다.
        self._pending_batch_count = {shoe_type: 0 for shoe_type in SHOE_TYPES}
        # 선반별 누적 박스 수 — /fms/amr_carrying 이벤트를 발행할 때마다 갱신한다
        # (정규화된 노드명 기준 — 근접/detour는 같은 선반이므로 카운트를 공유해야 한다).
        self._shelf_box_count = {node: 0 for node in SHELF_INDEX}

        # 예전엔 depth=10짜리 기본(VOLATILE) QoS였는데, 이러면 fleet_driver가
        # 아직 구독을 안 맺은 짧은 시간(launch에서 fleet_driver를 지연 시작할
        # 때) 사이에 나간 첫 이동 명령이 그냥 유실된다 — FMS는 "보냈다"고 믿고
        # MOVE_TIMEOUT_SEC(60초)까지 기다리다 재전송해야만 실제로 전달되는
        # 문제가 있었다(crossing_test_fms.py에서 정확히 같은 문제로 재현됨).
        # RELIABLE_EVENT_QOS(TRANSIENT_LOCAL)로 바꿔서 fleet_driver가 늦게
        # 구독해도 그 사이 발행분을 그대로 받게 한다 — 구독 쪽도 반드시 같은
        # QoS로 맞춰야 한다(fleet_driver.py 참고).
        self.command_pub = self.create_publisher(String, "/fms/commands", RELIABLE_EVENT_QOS)
        self.create_subscription(String, "/amr/status", self._on_robot_status, 10)

        # 메인 컨트롤 노드 → FMS: 신발 종류 하나에 대한 배치(길이 리스트) 작업 지시.
        # 예전에는 FMS가 비전 서비스를 직접 트리거·호출했지만, 이제는 메인 컨트롤
        # 노드가 비전 분류 결과를 정리해서 이 토픽으로 바로 넘겨준다 — FMS는 더
        # 이상 비전 쪽과 직접 통신하지 않는다. 서비스(ShoesList.srv)가 아니라
        # 토픽(PickupList.msg) 구독으로 받는다 — 메인 컨트롤이 응답을 기다릴
        # 필요 없이 계속 쌓아 보내고, FMS는 들어오는 대로 큐에 쌓기만 하면 된다.
        self.create_subscription(PickupList, "/control/pickup", self._on_pickup_list, 10)
        # FMS → 메인 컨트롤 노드: 배치 작업 완료 / 교착 상태 발생 알림 (응답 불필요).
        self.amr_state_client = self.create_client(AmrState, "/control/amr_state")
        # "AMR이 PICKUP에 도착해 어떤 신발을 실어야 하는지" / "AMR이 저장소에
        # 도착해 신발을 내려놓았는지"를 시뮬레이션에 알리는 이벤트 토픽 —
        # 원래는 시뮬레이션이 매 프레임 서비스(Trigger)로 물어보는(폴링) 구조였는데,
        # 시뮬레이션이 별도 컴퓨터(유선 연결, YOLO 추론까지 같이 돎)에서 돌 예정이라
        # 실제 이벤트가 났을 때만 발행하는 토픽으로 바꿨다 — 폴링은 매 프레임 네트워크
        # 왕복이 생겨 그 컴퓨터의 CPU/네트워크를 불필요하게 잡아먹는다. 커스텀
        # 인터페이스는 시뮬레이션 쪽에서 못 쓰므로 기존 관례대로 std_msgs/String에
        # JSON을 담아 보낸다. QoS는 RELIABLE_EVENT_QOS(위 정의) — 구독하는
        # 시뮬레이션 쪽도 반드시 같은 QoS로 맞춰야 한다.
        self.amr_ready_pub = self.create_publisher(String, "/fms/amr_ready", RELIABLE_EVENT_QOS)
        self.amr_carrying_pub = self.create_publisher(String, "/fms/amr_carrying_complete", RELIABLE_EVENT_QOS)

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
            "tasks": [],  # 이번 트립에서 아직 안 내려놓은 태스크들(최대 MAX_SHOES_PER_TRIP개) — 맨 앞이 현재 진행 중
            "move_started_at": None,
            "move_target": None,
            "dwell_until": None,
            "batch_complete_target": None,  # 배치 완료로 PICKUP/PICKUP_WAIT로 향하는 중이면 그 목표
        }
        self.node_locks[home_node] = robot_id
        self.get_logger().info(f"[{robot_id}] 등록 완료 (담당 종류={shoe_type}, 홈 슬롯={home_node})")
        return True

    def _on_pickup_list(self, msg):
        if not (0 <= msg.place < len(SHOE_TYPES)):
            self.get_logger().error(f"알 수 없는 place: {msg.place}")
            return
        shoe_type = SHOE_TYPES[msg.place]

        for length_mm in msg.shoes:
            shoe_size = _bucket_size(length_mm)
            target_node = TARGET_SLOT_NODE[(shoe_type, shoe_size)]
            self._next_shoe_id += 1
            self.task_queues[shoe_type].append({"shoe_id": self._next_shoe_id, "target_node": target_node})
            self.get_logger().info(
                f"[신규 태스크] shoe_id={self._next_shoe_id} 길이={length_mm}mm({shoe_size}) "
                f"→ {target_node} (담당 종류={shoe_type})"
            )
        self._pending_batch_count[shoe_type] += len(msg.shoes)

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

            if robot["path_idx"] >= len(robot["path"]) - 1:
                # 목적지 도착 완료
                if robot["tasks"]:
                    # 랙의 목표 지점에 도착 — 실제로 옮기지는 않지만, 신발을 내려놓는
                    # 가상의 시간만큼 여기 머무른다. 이 동안 이 노드(및 근접 통로의
                    # 경우 "280" 체크포인트를 이미 지나왔다면 그 지점)는 계속 락이
                    # 걸려있어, 같은 랙으로 오는 다른 로봇이 근접/detour를 판단할 때
                    # 반영된다.
                    self._publish_amr_carrying(robot_id, arrived_node)
                    robot["tasks"].pop(0)  # 방금 도착해서 내려놓을(dwelling 예정인) 태스크 제거
                    robot["state"] = "dwelling"
                    robot["dwell_until"] = now + Duration(seconds=SHOE_PLACEMENT_DWELL_SEC)
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

            if robot["tasks"]:
                # 이번 트립에서 이 로봇이 맡은 배치에 아직 안 내려놓은 신발이
                # 남아있다 — 픽업으로 복귀하지 않고 그대로 다음 목표(랙의 다음
                # 사이즈 슬롯)로 이어서 이동한다. 근접/detour는 배정 시점이
                # 아니라 지금(막 하나 내려놓은 시점) 다시 판단한다 — 방금
                # 로봇이 근접 슬롯을 떠나면서 락이 풀렸을 수도 있고, 그 사이
                # 다른 로봇이 들어왔을 수도 있어서다.
                next_task = robot["tasks"][0]
                if arrived_node.startswith(f"Rack{shoe_type}_"):
                    # 이미 랙 레인(근접이든 detour든) 안에 들어와 있다 — 근접↔detour는
                    # 입구(HUB)에서만 갈라질 뿐 랙 진입 후에는 서로 이어져 있지 않아서,
                    # 지금 레인을 바꾸려 하면 shortest_path가 반대쪽 레인 끝까지 다
                    # 지나 픽업/허브를 한 바퀴 통째로 돌아 재진입하는 경로를 돌려준다
                    # — 실제로 근접 레인 안에서(예: RackA_260) 마지막 신발만 detour로
                    # 재배정되는 바람에 트립 전체를 한 바퀴 더 도는 문제가 있었다
                    # (반대 방향인 detour→근접 전환도 똑같은 문제였음). _pick_rack_target로
                    # 재판단하지 않고, 이번 트립 동안은 지금 들어와 있는 레인 그대로 유지한다.
                    actual_target = (
                        f"{next_task['target_node']}_detour" if arrived_node.endswith("_detour")
                        else next_task["target_node"]
                    )
                else:
                    actual_target = self._pick_rack_target(next_task["target_node"], robot_id)
                next_task["target_node"] = actual_target

                if actual_target == arrived_node:
                    # 다음 목표가 방금 내려놓은 바로 그 자리다 — 같은 트립에서
                    # 같은 사이즈 신발을 연달아 같은 근접 슬롯에 놓는 경우
                    # (_pick_rack_target이 "자기 자신의 락"은 점유로 안 치므로
                    # 발생). 이동할 필요가 전혀 없으니 그 자리에서 곧바로 다음
                    # 배치 대기(dwelling)로 들어간다 — 경로가 없어서(출발=도착)
                    # 이동을 시도하면 도착 이벤트가 다시 안 와 영영 멈춰버린다.
                    self._publish_amr_carrying(robot_id, arrived_node)
                    robot["tasks"].pop(0)
                    robot["dwell_until"] = now + Duration(seconds=SHOE_PLACEMENT_DWELL_SEC)
                    robot["state"] = "dwelling"
                    self.get_logger().info(
                        f"[{robot_id}] 배치 대기 종료. 같은 자리({actual_target})에 이어서 다음 신발을 "
                        f"내려놓습니다 (남은 {len(robot['tasks'])}개)."
                    )
                    continue

                next_path = shortest_path(arrived_node, actual_target)
                if next_path:
                    robot["path"] = next_path
                    robot["path_idx"] = 0
                    robot["state"] = "waiting_next_hop"
                    self.get_logger().info(
                        f"[{robot_id}] 배치 대기 종료. 같은 트립의 다음 목표 {actual_target}(으)로 이동합니다 "
                        f"(남은 {len(robot['tasks'])}개)."
                    )
                else:
                    self.get_logger().error(f"[{robot_id}] 배치 내 다음 목표 경로 없음: {arrived_node} → {actual_target}")
                    robot["state"] = "idle"
                continue

            # 이 로봇이 맡은 이번 트립이 전부 끝났다 — 복귀 목적지는 항상
            # PICKUP(또는 이미 점유돼 있으면 PICKUP_WAIT)이다. batch_done(이
            # 종류의 전체 배치가 다 끝났는지)일 때만 도착 시 완료 보고
            # (AMR_STATE_COMPLETE)를 하도록 표시해둔다.
            target = self._pick_pickup_target(shoe_type)
            robot["batch_complete_target"] = target if batch_done else None
            self.get_logger().info(f"[{robot_id}] 트립 종료. {target}(으)로 복귀합니다.")
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
            # MIN_SHOES_TO_DISPATCH개 미만이면 idle 로봇이 있어도 배정하지 않고
            # 큐에 더 쌓일 때까지 기다린다 — 이 종류에 더 이상 안 들어올 마지막
            # 자투리(예: MIN보다 적게 남고 다음 PickupList가 안 옴)는 그 로봇이
            # idle인 채로 무기한 대기하게 되는데, 이건 의도된 트레이드오프다.
            while len(queue) >= MIN_SHOES_TO_DISPATCH:
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

                # 로봇 1대가 한 트립에 실을 수 있는 최대 개수(MAX_SHOES_PER_TRIP)만큼
                # 큐 앞에서 꺼낸다 — 신발 하나마다 왕복하지 않고 한 번 나가서
                # 여러 개를 연달아 내려놓고 돌아오게 하기 위함. 랙 진입이
                # 280→260→240 방향 일방통행이라, 꺼낸 묶음 안에서는 반드시 이
                # 순서로 재정렬해야 한다(작은 사이즈부터 들르면 큰 사이즈로
                # 되돌아갈 수 없다).
                batch = sorted(
                    queue[:MAX_SHOES_PER_TRIP],
                    key=lambda t: _SIZE_VISIT_ORDER[t["target_node"].split("_")[1]],
                )

                # 근접 통로의 첫 번째 저장소(280)를 아직 아무도 지나지 않은 상태(=락이
                # 비어있음)면 근접, 누가 아직 거기 있으면(이동 중이거나 배치 대기 중)
                # detour로 배정한다 — 배치의 첫 목표만 지금 판단하고, 나머지는 각자
                # 차례가 됐을 때(dwelling 종료 시점에) 다시 판단한다.
                first_task = batch[0]
                actual_target = self._pick_rack_target(first_task["target_node"])
                best_robot_id, best_path = None, None
                for robot_id, robot in candidates:
                    path = shortest_path(robot["current_node"], actual_target)
                    if path is not None:
                        best_robot_id, best_path = robot_id, path
                        break  # 번호가 가장 앞선 idle 로봇을 그대로 채택

                if best_robot_id is None:
                    self.get_logger().error(f"경로 불가: {actual_target} (담당 종류={shoe_type})")
                    break

                del queue[: len(batch)]
                first_task["target_node"] = actual_target
                robot = self.robots[best_robot_id]
                robot["tasks"] = batch
                robot["path"] = best_path
                robot["path_idx"] = 0
                robot["state"] = "waiting_next_hop"
                self.get_logger().info(
                    f"[{best_robot_id}] 트립 시작(신발 {len(batch)}개, 목표 순서 "
                    f"{[t['target_node'] for t in batch]}): {' → '.join(best_path)}"
                )
                # 예전엔 로봇이 실제로 PICKUP_X 노드에 "도착"하는 이벤트를 기준으로
                # amr_ready를 보냈는데, 홈 슬롯이 PICKUP_X 자체인 로봇(종류별 1번
                # 로봇)은 트립을 그 자리에서 바로 시작해버려서 "도착" 이벤트 자체가
                # 안 생기는 경우가 있었다 — 그러면 amr_ready가 영영 안 나간다. FMS가
                # 이 트립의 이동 명령을 로봇에게 실제로 내리는(=배정을 확정하는)
                # 지금 이 시점에 보내는 걸로 바꿔서, 홈 슬롯이 어디든 상관없이
                # 트립이 시작되면 항상 한 번은 보내지도록 한다.
                self._publish_amr_ready(best_robot_id, robot)

        # 1.5) 픽업 대기열 앞당기기 — 메인 컨트롤에서 새 작업 지시가 없어도(=
        # 위 1번에서 아무도 배정 못 받았어도), PICKUP_X가 비어있고 그 바로 뒤
        # 대기 슬롯(PICKUP_WAIT_X, WAIT2_X, ...)에 idle 로봇이 있으면 한 칸
        # 앞으로 당긴다 — 실제 대기줄이 맨 앞이 빌 때마다 자연스럽게 당겨지는
        # 것과 같은 동작이다. 이러면 다음 배치가 왔을 때 이미 PICKUP_X에
        # 로봇이 대기 중이라 더 빨리 실어줄 수 있다. 위 1번에서 실제 트립을
        # 배정받은 로봇은 이미 state가 idle이 아니게 됐으니 여기서 다시
        # 건드리지 않는다.
        for shoe_type in SHOE_TYPES:
            chain = [PICKUP_NODE[shoe_type]] + PICKUP_WAIT_SLOTS[shoe_type]
            for front, back in zip(chain, chain[1:]):
                if self.node_locks.get(front) is not None:
                    continue  # 앞자리가 이미 차있으면 당길 필요 없음
                back_holder = self.node_locks.get(back)
                if back_holder is None:
                    continue
                robot = self.robots.get(back_holder)
                if robot is None or robot["state"] != "idle":
                    continue
                advance_path = shortest_path(back, front)
                if advance_path:
                    robot["path"] = advance_path
                    robot["path_idx"] = 0
                    robot["state"] = "waiting_next_hop"
                    self.get_logger().info(f"[{back_holder}] 대기열 앞당김: {back} → {front}")

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
            and robot["tasks"]
            and not robot["tasks"][0]["target_node"].endswith("_detour")
        ):
            current_task = robot["tasks"][0]
            detour_target = f"{current_task['target_node']}_detour"
            detour_path = shortest_path(robot["current_node"], detour_target)
            if detour_path is not None:
                self.get_logger().info(
                    f"[{robot_id}] {next_node} 점유 중 → detour로 재경로 "
                    f"({current_task['target_node']} → {detour_target})"
                )
                current_task["target_node"] = detour_target
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

    def _pick_rack_target(self, canonical_target, robot_id=None):
        """canonical_target(예: RackA_240)에 대해 근접/detour 중 실제로 쓸 노드를 고른다.

        근접 통로는 280→260→240 순서의 일방통행 사슬이라, 목표가 240이면 280과
        260을 실제 노드로 반드시 거쳐가야(락을 잡아야) 한다. 그래서 "280 체크포인트"
        하나만 보면 안 되고, 체크포인트부터 목표 사이즈까지 근접 레인 구간
        전부가 비어있어야 근접을 쓴다 — 예를 들어 260 배치를 든 로봇이 근접
        260에서 오래 머무르는(같은 자리에 여러 개 연달아 내려놓는) 동안, 뒤따라와
        240을 배정받은 다른 로봇이 280은 비었다는 이유만으로 근접에 진입했다가
        260에서 오래 막히는 문제가 있었다(240은 260에 내려놓을 일이 없는데도).
        중간 어느 한 곳이라도 점유 중이면 그 지점부터는 못 지나가므로 바로
        detour로 보낸다.

        robot_id를 넘기면 "그 락을 자기 자신이 쥐고 있는 경우"는 점유로 치지
        않는다 — 같은 트립 안에서 같은 사이즈 신발을 연달아 내려놓을 때, 방금
        자신이 도착한 그 자리를 "남이 점유 중"으로 오판해 근접 레인 전체를
        통과해 픽업까지 돌아갔다가 detour로 재진입하는 거대한 우회를 막기
        위함이다(자기 자신이니 이동 없이 같은 자리에 이어서 내려놓으면 된다).
        새로 배정되는 로봇(아직 어떤 랙 락도 쥐고 있지 않음)에는 영향 없다.
        """
        rack_prefix, target_size = canonical_target.split("_")  # "RackA_240" → "RackA", "240"
        for size in _RACK_SIZES_IN_ORDER:  # ["280", "260", "240"] 순서로 체크포인트부터 검사
            if self.node_locks.get(f"{rack_prefix}_{size}") not in (None, robot_id):
                return f"{canonical_target}_detour"
            if size == target_size:
                break
        return canonical_target

    def _pick_pickup_target(self, shoe_type):
        """배치를 다 끝낸 로봇이 향할 곳 — 자기 종류 PICKUP_X가 비어있으면
        PICKUP_X, 이미 다른 로봇이 있으면 대기 슬롯(PICKUP_WAIT_X, WAIT2_X, ...)
        중 앞에서부터 비어있는 첫 자리를 고른다. ROBOTS_PER_TYPE이 2대여서
        슬롯이 하나뿐이면 이전과 동일하게 동작하고, 3대 이상으로 늘어나면
        fleet_config의 PICKUP_WAIT_SLOTS도 자동으로 늘어나 있어 그대로 재사용된다."""
        pickup_node = PICKUP_NODE[shoe_type]
        if self.node_locks.get(pickup_node) is None:
            return pickup_node
        for slot in PICKUP_WAIT_SLOTS[shoe_type]:
            if self.node_locks.get(slot) is None:
                return slot
        # 슬롯이 전부 점유 중이면 맨 뒤에서 대기시킨다 — 노드 락 시스템이 알아서
        # 순서대로 통과시켜주므로 여기서 더 정교하게 고를 필요는 없다.
        return PICKUP_WAIT_SLOTS[shoe_type][-1]

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

    def _publish_amr_ready(self, robot_id, robot):
        # 여기서 말하는 "신발 번호"는 PickupList.msg의 place와 같은 종류 인덱스
        # (A=0,B=1,C=2,D=3)이지 task 내부의 순번(shoe_id)이 아니다. 도착한 AMR의
        # 실제 번호(robot_id "amr_3" → 3, _report_amr_state와 동일한 방식)와
        # 이번에 실을 개수도 같이 묶어 보낸다(시뮬레이션이 신발 프림을 몇 개
        # 올려야 하는지는 이걸로 충분하고, 각 신발이 나중에 어느 선반으로
        # 갈지는 그때 가서 /fms/amr_carrying이 알려주므로 여기선 안 보낸다).
        tasks = robot["tasks"]  # 이번 트립에 실을 배치 전체(최대 MAX_SHOES_PER_TRIP개)
        msg = String()
        msg.data = json.dumps({
            "amr_id": int(robot_id.split("_")[1]),
            "count": len(tasks),
        })
        self.amr_ready_pub.publish(msg)
        self.get_logger().info(
            f"[{robot_id}] PICKUP 도착 — 신발 수령 준비 완료 (shoe_type={robot['shoe_type']}, "
            f"배치 {len(tasks)}개, shoe_ids={[t['shoe_id'] for t in tasks]})"
        )

    def _publish_amr_carrying(self, robot_id, arrived_node):
        canonical_node = _canonical_rack_node(arrived_node)
        shelf_num = SHELF_INDEX[canonical_node]
        self._shelf_box_count[canonical_node] += 1
        msg = String()
        msg.data = json.dumps({
            "amr_id": int(robot_id.split("_")[1]),
            "shelf_num": shelf_num,
            "box_count": self._shelf_box_count[canonical_node],
        })
        self.amr_carrying_pub.publish(msg)
        self.get_logger().info(
            f"[{robot_id}] {arrived_node}(선반 {shelf_num}) 도착 — 누적 박스 수 "
            f"{self._shelf_box_count[canonical_node]}"
        )

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
        # 예전엔 여기서 매 이벤트(감지/미복구/해소)마다 /fms/deadlock_alert 토픽도
        # 발행했는데, 그걸 구독해서 쓰는 대시보드가 없었고 메인 컨트롤 알림은
        # 어차피 이 함수 안의 _report_amr_state(임계치 이상일 때만)가 독립적으로
        # 담당하고 있어서 그 발행은 제거했다 — 로그 + 메인 컨트롤 보고만 남긴다.
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
