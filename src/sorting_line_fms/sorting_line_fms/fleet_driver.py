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
  3. go-to-goal 제어(각도 오차에 비례해 회전하면서 동시에 전진 — 오차가 클수록
     cos(오차)만큼 속도를 줄여 자연스럽게 감속, 완전 정지 후 제자리 회전은 안 함)로
     /<robot_id>/cmd_vel(Twist)을 계산해 내보낸다.
  4. 목표 반경 안에 들어오면 정지시키고 FMS에 /amr/status arrived를 보고한다.

주의: NODE_GRAPH의 랙 슬롯 Z값(0.3~2.1m)은 "선반 높이"이지 지상 로봇이 실제로
올라갈 좌표가 아니다 — Nova Carter는 지상 주행체라 내비게이션은 XY 평면만 쓰고
Z는 무시한다(신발을 몇 번째 선반에 놓을지는 이 노드가 관여하지 않는 별도 문제).
"""

import importlib
import json
import math
import sys

from rclpy.duration import Duration

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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

# fms_node.py/crossing_test_fms.py가 /fms/commands를 발행할 때 쓰는 QoS와
# 반드시 똑같이 맞춰야 한다 — TRANSIENT_LOCAL이라야, 이 노드(구독자)가 FMS
# 노드보다 늦게(launch에서 지연 시작 등) 뜨더라도 그 사이 발행된 이동 명령을
# 유실 없이 그대로 받는다. 예전엔 이게 안 맞아서 첫 이동 명령이 통째로
# 사라지고, FMS가 도착 응답을 못 받아 워치독 타임아웃(수십 초)이 지나서야
# 같은 명령을 재전송해 겨우 움직이는 문제가 있었다.
_COMMAND_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
)

# NODE_GRAPH/ROBOT_HOME_NODE/robot_spawn_yaw는 모듈 상단에서 바로 import하지
# 않고, __init__에서 config_module 파라미터로 받은 이름으로 동적 import한다 —
# crossing_test_fms.py처럼 운영 그래프(fleet_config)가 아닌 다른 그래프(예:
# edge_conflict 테스트 전용 crossing_test_config)로 이 노드를 그대로 재사용할
# 수 있게 하기 위함이다. 기본값은 fleet_config라 기존 운영 launch는 아무것도
# 안 바꿔도 그대로 동작한다.

ARRIVE_RADIUS_M = 0.12  # 로봇 트랙폭(0.4132m) 대비 15cm는 너무 헐거워서 8cm로 조정
# _subdivide_edge가 자동으로 만든 본선 세분화 칸("__"이 이름에 들어간 노드)은
# 항상 실제 두 노드 사이를 직선 보간(t=0~1)한 결과라서, 태생적으로 전부 일직선
# 위에 있다 — 방향이 꺾이는 지점은 항상 "__"가 없는 진짜 노드(APPROACH 등)
# 에서만 생기고, 세분화 칸끼리는 절대 안 꺾인다. 그래서 이 칸들만 도착 반경을
# 넉넉하게 잡아도 다음 칸으로 넘어갈 때 오차가 누적되지 않는다(다음 목표는
# 항상 로봇의 실시간 절대 위치에서 다시 계산되므로) — 굴절점(꺾이는 진짜
# 노드)은 여기 해당 안 되니 ARRIVE_RADIUS_M을 그대로 쓴다.
WAYPOINT_RADIUS_M = 0.35
# 예전엔 여기(비최종 홉이 도착 반경 0.3m 안에 들어오면 감속)에 OVERSHOOT_GUARD_M을
# 뒀었는데, 사실 비최종 홉은 전부 세분화 칸("__" 포함, 위 WAYPOINT_RADIUS_M
# 설명대로 항상 일직선)이라 애초에 꺾일 일이 없어서 이 감속 자체가 불필요했다
# — 코너링 문제는 실제로는 진짜 굴절점(HUB_A_APPROACH 등, "__" 없어서
# is_final=True로 이미 K_LINEAR*distance 감속이 걸림)에서 생기는 거라 여기서
# 따로 손 쓸 필요가 없었다. WAYPOINT_RADIUS_M을 넉넉하게 키운 지금은 세분화
# 칸에서 감속 없이 최고 속도를 계속 유지한다.
MAX_LINEAR_MPS = 1.0
MAX_ANGULAR_RPS = 1.2
K_LINEAR = 0.8
K_ANGULAR = 1.5
CONTROL_PERIOD_SEC = 0.05    # 20Hz
# 도착 후 단 한 번만 0 속도를 보내면, 물리 엔진의 잔여 관성(PhysX residual
# velocity)이 안 죽어서 로봇이 도착한 뒤에도 계속 미세하게 미끄러지는 문제가
# 있었다 — 실제로 fleet_driver/FMS는 명령을 딱 한 번만 보내는 걸 로그로
# 확인했으니 코드가 반복해서 잘못 보내는 게 아니라 순수 물리 잔여 속도
# 문제였다. 완전 정지가 실제로 중요한 지점(PICKUP_X, PICKUP_WAIT_X처럼
# 로봇끼리 만나서 실제로 겹치면 안 되는 자리)에서만, 도착 직후 이 시간
# 동안 0 속도를 반복 전송해서 PhysX가 확실히 정지할 시간을 벌어준다.
STOP_HOLD_SEC = 0.5


def _is_pickup_area_node(node_id):
    """PICKUP_X, PICKUP_WAIT_X, PICKUP_WAIT2_X... 처럼 로봇이 실제로 완전히
    멈춰서 다른 로봇과 자리를 주고받아야 하는 노드인지 판정한다. PICKUP_X_APPROACH나
    본선 세분화 칸("__" 포함)은 진짜 정지 지점이 아니라서 제외한다.

    crossing_test_config.py처럼 픽업 개념이 아예 없는 그래프도 있다 — 거기서도
    완전 정지가 필요한 지점(R1_HOME/R1_FAR/R2_HOME/R2_FAR)이 있는데 PICKUP_
    접두사가 없어서 안 걸렸었다. 그 그래프의 노드 이름 규칙("_HOME"/"_FAR"로
    끝남)도 같이 인정한다 — 운영 그래프(fleet_config 계열)엔 이 접미사를 쓰는
    노드가 없어서 서로 겹칠 일은 없다."""
    if node_id is None:
        return False
    if "__" in node_id:
        return False  # 본선 세분화 칸은 어떤 그래프든 진짜 정지 지점이 아님
    if node_id.startswith("PICKUP_") and "APPROACH" not in node_id:
        return True
    if node_id.endswith("_HOME") or node_id.endswith("_FAR"):
        return True
    return False


def _is_platoon_segment(node_id):
    """_subdivide_edge가 자동으로 만든 본선 세분화 칸인지 판정한다 — 이름에
    "__"(이중 언더스코어)이 들어있으면 항상 세분화 칸이고, 실제 노드 이름
    (PICKUP_A, HUB_A_APPROACH 등)은 전부 단일 언더스코어만 쓰므로 이 표시로
    구분할 수 있다(1_conveyor_sorter_env.py의 is_final_hop 판정과 동일한 기준)."""
    return node_id is not None and "__" in node_id


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
        self.declare_parameter("config_module", "fleet_config_test1")
        config_module_name = self.get_parameter("config_module").get_parameter_value().string_value
        config = importlib.import_module(config_module_name)
        self.NODE_GRAPH = config.NODE_GRAPH
        self.ROBOT_HOME_NODE = config.ROBOT_HOME_NODE
        self.robot_spawn_yaw = config.robot_spawn_yaw

        self.robots = {}  # robot_id -> {x, y, yaw, has_odom, target_xy, target_node, cmd_pub}
        # fms_node.py가 교착 반복(DEADLOCK_NOTIFY_THRESHOLD회 이상)을 감지하면
        # /fms/emergency_stop으로 전체 정지를 방송한다 — 이 로봇의 종류/그래프와
        # 무관하게 여기서 관리하는 모든 로봇을 즉시 멈추고, 해제({"stop": false})
        # 전까지는 새 이동 명령이 와도(_on_command가 target을 갱신해도) 무시한다.
        self._emergency_stopped = False

        self.status_pub = self.create_publisher(String, "/amr/status", 10)
        self.create_subscription(String, "/fms/commands", self._on_command, _COMMAND_QOS)
        self.create_subscription(Bool, "/fms/emergency_stop", self._on_emergency_stop, _COMMAND_QOS)

        for robot_id in self.ROBOT_HOME_NODE:
            self._ensure_robot(robot_id)
            self._publish_status(robot_id, "idle", "BOOT")

        self.create_timer(CONTROL_PERIOD_SEC, self._control_tick)
        self.get_logger().info(f"Fleet Driver 시작 — {len(self.robots)}대 담당 (cmd_vel/odom 기반)")

    def _ensure_robot(self, robot_id):
        if robot_id in self.robots:
            return
        # IsaacComputeOdometry는 로봇의 "스폰 지점을 원점, 스폰 자세를 0도로 하는"
        # 로컬 좌표를 낸다(실제 휠 오도메트리와 동일한 표준 동작 — 버그가 아니다).
        # 반면 NODE_GRAPH/목표 좌표는 전부 월드(절대) 좌표라서, 오도메트리 값을
        # 그대로 "현재 위치"로 쓰면 목표까지의 거리·방향 계산이 완전히 틀어진다
        # (실제로 로봇들이 마커와 무관한 방향으로 튀는 문제의 원인이었음). 스폰
        # 시점의 월드 좌표(=자기 홈 슬롯 위치)와 스폰 자세(yaw)를 오프셋으로
        # 저장해뒀다가, 오도메트리를 받을 때마다 회전+평행이동으로 월드 좌표로
        # 변환한다. 스폰 자세를 이제 첫 이동 방향과 맞춰서 0이 아닐 수 있으므로
        # (robot_spawn_yaw) 평행이동만으로는 부족하고 회전 보정이 반드시 필요하다.
        home_node = self.ROBOT_HOME_NODE[robot_id]
        spawn_x, spawn_y, _spawn_z = self.NODE_GRAPH[home_node]["position"]
        spawn_yaw = self.robot_spawn_yaw(robot_id)
        self.robots[robot_id] = {
            "x": 0.0, "y": 0.0, "yaw": 0.0, "has_odom": False,
            "spawn_x": spawn_x, "spawn_y": spawn_y, "spawn_yaw": spawn_yaw,
            "target_xy": None, "target_node": None, "target_is_final": True,
            "stop_until": None,  # 이 시각까지는 0 속도를 계속 반복 전송(아래 STOP_HOLD_SEC 참고)
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
        spawn_yaw = robot["spawn_yaw"]
        cos_yaw = math.cos(spawn_yaw)
        sin_yaw = math.sin(spawn_yaw)
        # 오도메트리는 "스폰 자세를 0도로 하는" 로컬 프레임 좌표라서, 스폰 자세가
        # 0이 아니면 평행이동만으로는 안 되고 스폰 yaw만큼 회전까지 시켜야
        # 월드 좌표로 정확히 바뀐다(표준 2D 강체 변환: 회전 후 평행이동).
        robot["x"] = robot["spawn_x"] + pos.x * cos_yaw - pos.y * sin_yaw
        robot["y"] = robot["spawn_y"] + pos.x * sin_yaw + pos.y * cos_yaw
        robot["yaw"] = _normalize_angle(spawn_yaw + _yaw_from_quaternion(msg.pose.pose.orientation))
        robot["has_odom"] = True

    def _on_emergency_stop(self, msg):
        self._emergency_stopped = msg.data
        if self._emergency_stopped:
            self.get_logger().error("[비상 정지] 전체 AMR 정지 (사유는 fms_node 로그 참고)")
            for robot in self.robots.values():
                robot["cmd_pub"].publish(Twist())
        else:
            self.get_logger().warn("[비상 정지 해제] 이동 재개")

    def _on_command(self, msg):
        data = json.loads(msg.data)
        robot_id = data["robot_id"]
        self._ensure_robot(robot_id)
        x, y, _z = data["position"]  # Z(선반 높이)는 지상 주행 목표에는 쓰지 않음
        self.robots[robot_id]["target_xy"] = (x, y)
        self.robots[robot_id]["target_node"] = data["node_id"]
        # FMS가 계산한, 경로의 마지막 홉인지 여부 — 없으면(구버전 FMS 등) 안전하게
        # "마지막"으로 간주해 정지시킨다.
        self.robots[robot_id]["target_is_final"] = data.get("is_final_hop", True)

    def _control_tick(self):
        if self._emergency_stopped:
            # 매 tick 0 속도를 반복 전송한다 — _on_emergency_stop에서 한 번만
            # 보내면 STOP_HOLD_SEC 로직과 마찬가지로 PhysX 잔여 관성이 안 죽어서
            # 계속 미끄러질 수 있다. target_xy/state는 건드리지 않고 그대로
            # 두므로, 해제되면 원래 향하던 목표로 자연스럽게 이어서 이동한다.
            for robot in self.robots.values():
                robot["cmd_pub"].publish(Twist())
            return

        now = self.get_clock().now()
        for robot_id, robot in self.robots.items():
            if robot["stop_until"] is not None:
                # PICKUP_X/PICKUP_WAIT_X 도착 직후 — PhysX 잔여 관성이 죽을
                # 때까지 0 속도를 반복 전송한다(STOP_HOLD_SEC 동안).
                if now < robot["stop_until"]:
                    robot["cmd_pub"].publish(Twist())
                    continue
                robot["stop_until"] = None

            target = robot["target_xy"]
            if target is None or not robot["has_odom"]:
                continue

            dx = target[0] - robot["x"]
            dy = target[1] - robot["y"]
            distance = math.hypot(dx, dy)

            # 세분화 칸("__" 포함)은 항상 일직선 위라 넉넉한 반경으로 판정해도
            # 다음 칸으로 넘어갈 때 오차가 안 쌓인다(WAYPOINT_RADIUS_M 설명 참고).
            # 진짜 목적지/굴절점은 기존 타이트한 반경 그대로.
            arrive_radius = WAYPOINT_RADIUS_M if _is_platoon_segment(robot["target_node"]) else ARRIVE_RADIUS_M
            if distance <= arrive_radius:
                arrived_node = robot["target_node"]
                is_final = robot["target_is_final"]
                robot["target_xy"] = None
                robot["target_node"] = None
                # 진짜 목적지(랙 슬롯, 홈 슬롯, 픽업 등)에서만 완전히 정지한다.
                # 본선 세분화 칸 같은 중간 경유지는 멈추지 않고 그대로 통과해야
                # 플래투닝(꼬리 물기)이 부드럽게 이어진다 — FMS가 0.5초 안에
                # 다음 홉 명령을 보내줄 거라고 가정한다.
                if is_final:
                    self._stop(robot_id)
                    if _is_pickup_area_node(arrived_node):
                        robot["stop_until"] = now + Duration(seconds=STOP_HOLD_SEC)
                self._publish_status(robot_id, "arrived", arrived_node)
                continue

            heading_error = _normalize_angle(math.atan2(dy, dx) - robot["yaw"])

            if robot["target_is_final"]:
                # 진짜 목적지/굴절점 — 도착 직전 자연스럽게 감속(P 제어).
                base_speed = min(K_LINEAR * distance, MAX_LINEAR_MPS)
            else:
                # 세분화 칸은 애초에 안 꺾이므로(WAYPOINT_RADIUS_M 설명 참고)
                # 감속 없이 최고 속도로 매끄럽게 통과한다.
                base_speed = MAX_LINEAR_MPS

            # 각도 오차가 크면 "완전 정지 후 제자리 회전 → 정렬되면 그제서야 직진"
            # 하던 것을, 회전과 전진을 동시에 하도록 바꿨다 — cos(heading_error)를
            # 곱해서 오차가 클수록 자연스럽게 속도를 줄이되(90도 근처에서 거의 0)
            # 완전히 끊지는 않는다. 오차가 90도를 넘으면(목표가 뒤쪽) 뒤로 가지
            # 않도록 0으로 clamp하고 그 자리에서 회전만 계속한다.
            speed_scale = max(0.0, math.cos(heading_error))

            twist = Twist()
            twist.linear.x = base_speed * speed_scale
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
