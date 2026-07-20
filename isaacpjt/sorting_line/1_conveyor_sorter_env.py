
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import random
import numpy as np
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from pxr import PhysxSchema, UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path

from fleet_config import NODE_GRAPH, ROBOT_HOME_NODE, ROBOT_SHOE_TYPE

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 파라미터                                                   ║
# ╚══════════════════════════════════════════════════════════════╝
# 컨베이어는 분류 없이 신발을 비전 검사 지점(=AMR 픽업 지점)까지만 옮긴다.
# 목적지 배정/이동은 AMR + FMS가 담당 (컨베이어에는 diverter 없음).
#
# ConveyorBelt_A06(패브릭 단일 트랙)은 표면이 root 기준 약 1.78m 위에 authored되어
# 있어서(실측), 지상 AMR이 닿을 수 있는 높이로 쓰려면 root 자체를 그만큼 아래로
# 내려야 한다 — 프레임 하단부가 바닥에 파묻히는 대신 벨트 표면 높이를 맞춘다.
BELT_LENGTH = 2.0            # set_local_scale의 X 배율 (원본 모듈 길이 약 2m → 실측 총 길이 약 4m)
BELT_SPEED = 0.4  # m/s
BELT_START_X = 0.0            # 실측: 벨트 원점이 중앙이 아니라 시작 모서리에 있음
BELT_END_X = 4.0               # 실측 X 최대값
BELT_WIDTH = 0.9                # 실측 Y 폭
BELT_SURFACE_Z = 0.794           # 목표 표면 높이 (AMR 접근 가능한 높이로 유지)
_BELT_ASSET_OFFSET = 1.7805298618227245  # 실측: root=0.025일 때 Belt 표면 top까지의 거리
BELT_TOP_Z = BELT_SURFACE_Z - _BELT_ASSET_OFFSET  # 벨트 root의 world Z 위치 (음수 = 바닥 아래로 내림)

SHOE_SIZE = np.array([0.15, 0.08, 0.05])
SHOE_SPAWN_POS = np.array([BELT_START_X + 0.3, 0.0, BELT_SURFACE_Z + SHOE_SIZE[2] / 2.0 + 0.01])
PICKUP_POINT = np.array([BELT_END_X - 0.3, 0.0, BELT_SURFACE_Z])  # 비전 검사 완료 후 AMR이 집어가는 지점

# ── AMR 함대 (Nova Carter x8, 실측 스펙) ──
# 제어 로직은 여기 없음 — 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 조종하고,
# 이 스크립트는 ROS2 디퍼렌셜 드라이브 브리지만 각 로봇에 붙여둔다.
WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

# ── 종류별 랙 + 크기별 슬롯 ──
# 신발 종류(A/B/C/D) 하나당 랙 하나, 그 랙 안에서 크기(소/중/대)에 따라 다른 높이에 배치.
# industrialsteelshelving_a01 실측 사용 가능 높이가 z=0.1~2.4m라서 그 안을 3등분한다.
SHOE_TYPES = ["A", "B", "C", "D"]
SHOE_SIZES = ["소", "중", "대"]
TYPE_COLORS = {
    "A": np.array([0.7, 0.2, 0.2]),
    "B": np.array([0.2, 0.5, 0.7]),
    "C": np.array([0.6, 0.5, 0.2]),
    "D": np.array([0.3, 0.6, 0.3]),
}
RACK_Y = 4.0
RACK_SLOT_Y = RACK_Y - 0.6  # 랙 정면(픽업 접근이 쉬운 쪽)으로 살짝 나온 지점
RACK_X = {"A": -3.75, "B": -1.25, "C": 1.25, "D": 3.75}
SIZE_SLOT_Z = {"소": 0.3, "중": 1.2, "대": 2.1}  # 실측 프레임 범위(0.1~2.4m) 3등분 (대략치)

REJECT_POS = [0.0, 5.7, 0.0]
BUFFER_POS = {"임시저장소1": [-1.5, 5.7, 0.0], "임시저장소2": [1.5, 5.7, 0.0]}


def get_target_slot(shoe_type, shoe_size):
    return np.array([RACK_X[shoe_type], RACK_SLOT_Y, SIZE_SLOT_Z[shoe_size]])

# ── SimReady / Isaac 표준 에셋 경로 ──
_ASSET_DIR = "/home/rokey/Downloads"
_WH1 = f"{_ASSET_DIR}/SimReady_Warehouse_01_NVD@10010/Assets/simready_content/common_assets/props"
_CT1 = f"{_ASSET_DIR}/SimReady_Containers_Shipping_01_NVD@10010/Assets/simready_content/common_assets/props"
_CT2 = f"{_ASSET_DIR}/SimReady_Containers_Shipping_02_NVD@10010/Assets/simready_content/common_assets/props"

SHELVING_USD = f"{_WH1}/industrialsteelshelving_a01/industrialsteelshelving_a01.usd"

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
CONVEYOR_USD = _assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A06.usd"  # 패브릭 벨트, 직선/평평, 단일 트랙

# 폐기/임시저장소는 종류별 랙과 별개로 존재 (지금은 라우팅 로직 없이 배치만)
REJECT_USD = f"{_CT1}/tote_a01/tote_a01.usd"
BUFFER_USD = {
    "임시저장소1": f"{_CT2}/blockpallet_a06/blockpallet_a06.usd",
    "임시저장소2": f"{_CT2}/blockpallet_b01/blockpallet_b01.usd",
}


def spawn_asset(usd_path, prim_path, position):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xform = SingleXFormPrim(prim_path)
    xform.set_world_pose(position=np.array(position))
    return xform


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 컨베이어 벨트 (Isaac 표준 ConveyorBelt_A06 에셋, 패브릭 단일 트랙)  ║
# ╚══════════════════════════════════════════════════════════════╝
# world.scene에 등록하지 않음: kinematic body는 World의 reset 시
# 자동 속도 초기화(setLinearVelocity)를 시도해 PhysX 에러가 남기 때문
belt = spawn_asset(CONVEYOR_USD, "/World/ConveyorBelt", position=[0.0, 0.0, BELT_TOP_Z])
belt.set_local_scale(np.array([BELT_LENGTH, 1.0, 1.0]))

# 이 에셋은 Belt 하위 프림에 이미 RigidBodyAPI가 붙어있다.
# 루트에 또 적용하면 "다중 RigidBodyAPI 계층" 충돌 에러가 나므로,
# 표면 속도만 이미 물리가 있는 하위 프림에 직접 적용한다.
for _sub_path in ["/World/ConveyorBelt/Belt"]:
    _sub_prim = stage.GetPrimAtPath(_sub_path)
    if _sub_prim.IsValid():
        surf_vel_api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(_sub_prim)
        surf_vel_api.CreateSurfaceVelocityEnabledAttr(True)
        surf_vel_api.CreateSurfaceVelocityAttr(Gf.Vec3f(BELT_SPEED, 0.0, 0.0))
    else:
        print(f"[경고] {_sub_path} 프림을 찾지 못함 — 에셋 내부 구조 확인 필요")

# 진단용: 실제 표면 높이/범위 확인 (A06은 start_level=1이라 A04의 Rollers와 높이가 다를 수 있음)
from pxr import Usd, UsdGeom
_bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for _diag_path in ["/World/ConveyorBelt", "/World/ConveyorBelt/Belt"]:
    _diag_prim = stage.GetPrimAtPath(_diag_path)
    if _diag_prim.IsValid():
        _r = _bbox_cache.ComputeWorldBound(_diag_prim).ComputeAlignedRange()
        print(f"[진단] {_diag_path} bbox min={_r.GetMin()} max={_r.GetMax()}")

pickup_marker = VisualCuboid(
    prim_path="/World/PickupPoint",
    name="pickup_point",
    position=PICKUP_POINT + np.array([0.0, 0.0, 0.001]),
    scale=np.array([0.2, BELT_WIDTH, 0.005]),
    color=np.array([0.0, 1.0, 0.3]),
)
world.scene.add(pickup_marker)


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. AMR 함대 (Nova Carter x8) — 순수 물리 + ROS2 브리지               ║
# ╚══════════════════════════════════════════════════════════════╝
# fms_node.py가 정해준 로봇별 홈 슬롯(WAIT_1~8) 위치에 스폰하고, 로봇마다
# 독립된 ROS2 디퍼렌셜 드라이브 그래프를 붙인다. 이동 제어는 전부 외부
# fleet_driver.py가 /<robot_id>/cmd_vel 로 담당 — 이 스크립트는 물리 세계만 제공.
#
# 주의: 그래프 노드 구성(Twist 벡터 분해, ROS2PublishOdometry 속성명)은 Isaac Sim
# 표준 디퍼렌셜 드라이브 샘플 패턴을 따랐지만 엔진 없이 작성한 부분이 있어
# 처음 실행 시 노드/속성 이름을 확인해야 할 수 있다.


def build_ros2_diffdrive_graph(robot_id, chassis_prim_path):
    graph_path = f"/World/{robot_id}/ROS_DiffDrive"
    keys = og.Controller.Keys

    (graph, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinear", "omni.graph.nodes.BreakVector3"),
                ("BreakAngular", "omni.graph.nodes.BreakVector3"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("SubscribeTwist.outputs:linearVelocity", "BreakLinear.inputs:tuple"),
                ("SubscribeTwist.outputs:angularVelocity", "BreakAngular.inputs:tuple"),
                ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
                ("BreakLinear.outputs:x", "DiffController.inputs:linearVelocity"),
                ("BreakAngular.outputs:z", "DiffController.inputs:angularVelocity"),
                # DifferentialController는 순수 계산 노드라 outputs:execOut이 없음 — 실행 검증됨.
                # ArticulationController의 트리거는 SubscribeTwist에서 바로 받는다
                # (DiffController의 데이터 출력은 같은 틱 안에서 데이터 의존성으로 먼저 계산됨).
                ("SubscribeTwist.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("DiffController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("OnTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
            ],
            keys.SET_VALUES: [
                ("SubscribeTwist.inputs:topicName", f"/{robot_id}/cmd_vel"),
                ("DiffController.inputs:wheelRadius", WHEEL_RADIUS_M),
                ("DiffController.inputs:wheelDistance", WHEEL_BASE_M),
                ("ArticulationController.inputs:targetPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("ArticulationController.inputs:jointNames", WHEEL_JOINT_NAMES),
                ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(chassis_prim_path)]),
                ("PublishOdometry.inputs:topicName", f"/{robot_id}/odom"),
                ("PublishOdometry.inputs:odomFrameId", f"{robot_id}/odom"),
                ("PublishOdometry.inputs:chassisFrameId", f"{robot_id}/base_link"),
            ],
        },
    )
    return graph


for _robot_id, _home_node in ROBOT_HOME_NODE.items():
    _home_pos = list(NODE_GRAPH[_home_node]["position"])
    _prim_path = f"/World/{_robot_id}"
    spawn_asset(NOVA_CARTER_USD, _prim_path, position=_home_pos)
    build_ros2_diffdrive_graph(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link")
    print(f"[스폰] {_robot_id} (담당 종류={ROBOT_SHOE_TYPE[_robot_id]}) @ {_home_node} {_home_pos}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 종류별 랙 + 폐기/임시저장소                                     ║
# ╚══════════════════════════════════════════════════════════════╝
for shoe_type in SHOE_TYPES:
    spawn_asset(SHELVING_USD, f"/World/Rack_{shoe_type}", position=[RACK_X[shoe_type], RACK_Y, 0.0])

spawn_asset(REJECT_USD, "/World/Storage_Reject", position=REJECT_POS)
for label, usd_path in BUFFER_USD.items():
    spawn_asset(usd_path, f"/World/Storage_{label}", position=BUFFER_POS[label])


# ╔══════════════════════════════════════════════════════════════╗
# ║  E. 테스트용 신발(큐브) 스폰 — 종류/크기 랜덤 배정 (비전 분류 결과 흉내)   ║
# ╚══════════════════════════════════════════════════════════════╝
shoe_count = 0
pending_shoes = []  # 벨트 위에서 픽업 대기 중인 신발 목록 (FIFO): [{"obj","type","size"}, ...]


def spawn_shoe():
    global shoe_count
    shoe_count += 1
    shoe_type = random.choice(SHOE_TYPES)
    shoe_size = random.choice(SHOE_SIZES)
    shoe = DynamicCuboid(
        prim_path=f"/World/Shoe_{shoe_count}",
        name=f"shoe_{shoe_count}",
        position=SHOE_SPAWN_POS,
        scale=SHOE_SIZE,
        color=TYPE_COLORS[shoe_type],
        mass=0.05,
    )
    world.scene.add(shoe)
    pending_shoes.append({"obj": shoe, "type": shoe_type, "size": shoe_size})
    print(f"[스폰] 신발 #{shoe_count} 종류={shoe_type} 크기={shoe_size} @ {SHOE_SPAWN_POS}")


world.scene.add_default_ground_plane()
world.reset()
spawn_shoe()

frame = 0
was_playing = False

print("\n" + "=" * 60)
print(f"[환경] 컨베이어 벨트 + Nova Carter {len(ROBOT_HOME_NODE)}대(ROS2 브리지) + "
      f"종류별 랙 {len(SHOE_TYPES)}개(각 {len(SHOE_SIZES)}슬롯) + 폐기/임시저장소 생성 완료")
print("  AMR 제어: 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 담당")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)
    is_playing = world.is_playing()

    if is_playing and not was_playing:
        frame = 0
        print("[재생] 시작")

    if is_playing:
        frame += 1

        # 5초(≈300 step)마다 새 신발 스폰
        if frame % 300 == 0:
            spawn_shoe()

        # pending_shoes는 픽업 지점 도달을 기다리는 FIFO 큐로 계속 쌓인다.
        # 실제로 이 신발을 로봇에 옮기는 로직(비전 srv 분류 + /fms/pickup_ready
        # 수신 후 프림을 로봇에 붙였다 랙에 내려놓는 처리)은 아직 이 스크립트에
        # 없다 — 다음 단계에서 이어서 구현.

    was_playing = is_playing

simulation_app.close()
