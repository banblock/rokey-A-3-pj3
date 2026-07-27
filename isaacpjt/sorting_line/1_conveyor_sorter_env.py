
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import math
import os

import numpy as np
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.storage.native import get_assets_root_path

from fleet_config import NODE_GRAPH, ROBOT_HOME_NODE, ROBOT_SHOE_TYPE, SHOE_TYPES, robot_spawn_yaw
from marker_config import MARKER_SIZE_M, build_marker_maps

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 파라미터                                                   ║
# ╚══════════════════════════════════════════════════════════════╝
# 이 스크립트는 Nova Carter만 스폰한다 — 컨베이어/랙/신발/폐기·임시저장소는
# 없음. 목적지 배정/이동은 AMR + FMS가 담당한다.

# ── AMR 함대 (Nova Carter x8, 실측 스펙) ──
# 제어 로직은 여기 없음 — 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 조종하고,
# 이 스크립트는 ROS2 디퍼렌셜 드라이브 브리지만 각 로봇에 붙여둔다.
WHEEL_RADIUS_M = 0.14
WHEEL_BASE_M = 0.4132  # 트랙폭(좌우 구동 바퀴 간격)
WHEEL_JOINT_NAMES = ["joint_wheel_left", "joint_wheel_right"]

_assets_root_path = get_assets_root_path()
if _assets_root_path is None:
    raise RuntimeError("Isaac Sim 기본 에셋 서버(Nucleus)에 연결할 수 없습니다.")
NOVA_CARTER_USD = _assets_root_path + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"

def spawn_asset(usd_path, prim_path, position, yaw=0.0):
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    xform = SingleXFormPrim(prim_path)
    # 스폰 자세를 yaw만큼 돌려서(기본은 항상 0=월드 X축 방향) 첫 이동 방향과
    # 맞춘다 — fleet_config.robot_spawn_yaw()가 계산해준 값을 그대로 받는다.
    # 안 맞추면 스폰 직후 첫 홉에서 제자리 회전부터 하고 출발하는 것처럼 보인다.
    half = yaw / 2.0
    orientation_wxyz = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    xform.set_world_pose(position=np.array(position), orientation=orientation_wxyz)
    return xform


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. AMR 함대 (Nova Carter x8) — 순수 물리 + ROS2 브리지               ║
# ╚══════════════════════════════════════════════════════════════╝
# ROBOT_HOME_NODE(각 종류의 PICKUP_X 또는 PICKUP_WAIT_X)에 스폰하고, 로봇마다
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
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
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
                ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
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


# ╔══════════════════════════════════════════════════════════════╗
# ║  C-2. AGV 마커(AprilTag) 인식용 하향 카메라                          ║
# ╚══════════════════════════════════════════════════════════════╝
# 실물에서는 각 Nova Carter가 이 카메라 스트림을 자기 온보드 컴퓨트(Jetson
# Orin)에서 직접 처리하지만, 시뮬레이션에는 로봇별로 분리된 물리 컴퓨트가
# 없다(2026-07-24 논의) — 그래서 검출 자체는 이 씬이 아니라 별도 ROS2 노드
# (agv_marker_localizer.py)가 담당하고, 이 씬은 "카메라 이미지를 토픽으로
# 내보내는 것"까지만 한다. 두 컴퓨터(시뮬레이션용 / FMS용) 구조에서
# agv_marker_localizer.py는 시뮬레이션 컴퓨터 쪽에서 띄우는 걸 전제로 한다 —
# 그래야 원본 카메라 영상이 네트워크를 안 넘어가고, 가벼운 보정값(JSON)만
# fleet_driver.py가 있는 FMS 컴퓨터로 건너간다.
MARKER_CAM_HEIGHT_M = 0.3  # 섀시 위 카메라 높이(실측 후 조정 예정)


def spawn_marker_camera(robot_id, chassis_prim_path, viewport_id):
    """chassis_prim_path 밑에 바닥을 내려다보는 카메라를 붙이고 ROS2로 퍼블리시한다.
    viewport_id는 로봇마다 달라야 한다(IsaacCreateViewport가 서로 겹치면 안 됨)."""
    camera_path = f"{chassis_prim_path}/MarkerCamera"
    camera_prim = UsdGeom.Camera(stage.DefinePrim(camera_path, "Camera"))
    xform_api = UsdGeom.XformCommonAPI(camera_prim)
    xform_api.SetTranslate(Gf.Vec3d(0.0, 0.0, MARKER_CAM_HEIGHT_M))
    # 카메라 기본(무회전) 방향이 로컬 -Z(아래)라 2_sneaker_camera.py처럼 무회전으로
    # 두면 바로 바닥을 본다 — 여기서는 섀시에 붙어서 섀시 자세를 그대로 따라가므로
    # 섀시가 수평이면 이대로 충분하다(엔진에서 실제로 바닥이 잡히는지 확인 필요).
    camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 2.0))
    camera_prim.GetHorizontalApertureAttr().Set(20.955)
    camera_prim.GetFocalLengthAttr().Set(9.5)

    graph_path = f"/World/{robot_id}/ROS_MarkerCamera"
    keys = og.Controller.Keys
    (graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
                ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
                ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                ("createViewport.inputs:viewportId", viewport_id),
                ("cameraHelperRgb.inputs:frameId", f"{robot_id}_marker_cam"),
                ("cameraHelperRgb.inputs:topicName", f"/{robot_id}/marker_cam/image_raw"),
                ("cameraHelperRgb.inputs:type", "rgb"),
                ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
            ],
        },
    )
    return graph


# 로봇 자체(섀시)에 색을 입히지 않고, 위에 종류별 색깔 비콘을 따로 띄운다 —
# Nova Carter는 이미 자체 머티리얼이 있는 복잡한 참조 에셋이라 내부 메시의
# 재질을 직접 바꾸는 건 실패 위험이 크다. 비콘은 매 시뮬레이션 스텝마다
# 섀시의 월드 위치를 읽어 같은 위치(+ 높이 오프셋)로 따라가게 한다
# (spawn_asset()과 동일한 set_world_pose 패턴 — 부모-자식 로컬 좌표 가정 없이
# 항상 안전하게 동작).
_ROBOT_TYPE_COLORS = {
    "A": np.array([0.80, 0.20, 0.20]),  # 빨강
    "B": np.array([0.85, 0.65, 0.13]),  # 노랑
    "C": np.array([0.20, 0.70, 0.30]),  # 초록
    "D": np.array([0.20, 0.40, 0.80]),  # 파랑
}
_BEACON_HEIGHT_OFFSET_M = 0.5
_robot_chassis_prims = {}
_robot_beacons = {}

for _viewport_id, (_robot_id, _home_node) in enumerate(ROBOT_HOME_NODE.items()):
    _home_pos = list(NODE_GRAPH[_home_node]["position"])
    _home_yaw = robot_spawn_yaw(_robot_id)
    _prim_path = f"/World/{_robot_id}"
    spawn_asset(NOVA_CARTER_USD, _prim_path, position=_home_pos, yaw=_home_yaw)
    build_ros2_diffdrive_graph(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link")
    spawn_marker_camera(_robot_id, chassis_prim_path=f"{_prim_path}/chassis_link", viewport_id=_viewport_id)
    print(
        f"[스폰] {_robot_id} (담당 종류={ROBOT_SHOE_TYPE[_robot_id]}) @ {_home_node} {_home_pos} "
        f"yaw={math.degrees(_home_yaw):.0f}도"
    )

    _robot_chassis_prims[_robot_id] = SingleXFormPrim(f"{_prim_path}/chassis_link")

    _beacon_pos = list(_home_pos)
    _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
    _beacon = VisualCuboid(
        prim_path=f"/World/{_robot_id}/ColorBeacon",
        name=f"beacon_{_robot_id}",
        position=np.array(_beacon_pos),
        scale=np.array([0.2, 0.2, 0.2]),
        color=_ROBOT_TYPE_COLORS[ROBOT_SHOE_TYPE[_robot_id]],
    )
    world.scene.add(_beacon)
    _robot_beacons[_robot_id] = _beacon


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 그래프 포인트 시각화 — NODE_GRAPH의 각 지점을 바닥에 색칠된      ║
# ║     정사각형으로 표시 (물리/충돌 없음, 순수 디버그용)                ║
# ╚══════════════════════════════════════════════════════════════╝
# 색은 노드 종류(픽업/허브/랙...)가 아니라 담당 신발 종류(A/B/C/D)로 칠한다 —
# 로봇 비콘 색(_ROBOT_TYPE_COLORS)과 동일한 매핑을 그대로 재사용해서, "이 로봇이
# 이 경로를 담당한다"는 게 색으로 바로 보이게 한다. 모양은 두 가지뿐: 픽업 대기
# 위치(PICKUP_WAIT_X)만 원기둥(동그라미로 보임), 나머지는 전부 사각기둥(네모).
# 세분화(_subdivide_edge)로 자동 생성된 중간 칸("__"이 이름에 들어간 노드)은
# 개수가 많아서 작게 그린다.


def _node_shoe_type(name):
    for _t in SHOE_TYPES:
        if name.startswith(f"PICKUP_WAIT_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"PICKUP_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"HUB_{_t}"):
            return _t
    for _t in SHOE_TYPES:
        if name.startswith(f"Rack{_t}"):
            return _t
    return None


for _node_name, _node_data in NODE_GRAPH.items():
    _node_type = _node_shoe_type(_node_name)
    _color = _ROBOT_TYPE_COLORS.get(_node_type, np.array([0.6, 0.6, 0.6]))  # 못 찾으면 회색(원래 없어야 함)
    _is_segment = "__" in _node_name
    _is_pickup_wait = any(_node_name.startswith(f"PICKUP_WAIT_{_t}") for _t in SHOE_TYPES)
    # 노드 이름이 전부 아스키(랙 사이즈는 240/260/280, 우회는 detour)라 USD 프림
    # 이름으로 그대로 써도 된다 — 예전엔 한글(소/중/대/우회)이 섞여 있어서
    # 별도 치환이 필요했지만 지금은 불필요.
    _size = 0.15 if _is_segment else 0.35
    _pos = list(_node_data["position"])
    _pos[2] = 0.01  # 바닥 위로 살짝 띄워서 그라운드 플레인과 Z-fighting 방지
    if _is_pickup_wait:
        marker = VisualCylinder(
            prim_path=f"/World/GraphMarkers/{_node_name}",
            name=f"marker_{_node_name}",
            position=np.array(_pos),
            radius=_size / 2.0,
            height=0.02,
            color=_color,
        )
    else:
        marker = VisualCuboid(
            prim_path=f"/World/GraphMarkers/{_node_name}",
            name=f"marker_{_node_name}",
            position=np.array(_pos),
            scale=np.array([_size, _size, 0.02]),
            color=_color,
        )
    world.scene.add(marker)


# ╔══════════════════════════════════════════════════════════════╗
# ║  E. AGV 마커(AprilTag) 배치 — 위 D의 색깔 마커는 카메라가 ID를 읽을 수  ║
# ║     없는 단색 도형이라(비트 패턴 없음) 실제 위치추정용으로는 못 쓴다.   ║
# ║     여기서는 generate_markers.py가 미리 만들어둔 실제 태그 이미지를    ║
# ║     텍스처로 입힌 평면을 정밀도가 중요한 지점(픽업/랙)에만 깐다.        ║
# ╚══════════════════════════════════════════════════════════════╝
MARKER_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers")


def _create_textured_marker_plane(prim_path, position, image_path, size_m):
    """position(바닥 위 살짝) 지점에 image_path 텍스처를 입힌 정사각 평면을 만든다.
    UsdPreviewSurface + UsdUVTexture 표준 조합 — 이 컴퓨터엔 Isaac Sim이 없어
    실행 검증은 못 했으니, 처음 띄울 때 텍스처가 제대로 보이는지 확인 필요."""
    plane = UsdGeom.Mesh.Define(stage, prim_path)
    half = size_m / 2.0
    plane.CreatePointsAttr([
        Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, -half, 0),
        Gf.Vec3f(half, half, 0), Gf.Vec3f(-half, half, 0),
    ])
    plane.CreateFaceVertexCountsAttr([4])
    plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane.CreateExtentAttr([(-half, -half, 0), (half, half, 0)])
    plane.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

    texcoords = UsdGeom.PrimvarsAPI(plane).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
    )
    texcoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    material_path = f"{prim_path}/Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)

    tex_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
    tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.ConnectToSource(tex_shader.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(plane).Bind(material)

    xform_api = UsdGeom.XformCommonAPI(plane)
    xform_api.SetTranslate(Gf.Vec3d(position[0], position[1], 0.005))  # 바닥 위 살짝(그라운드와 Z-fighting 방지)
    return plane


_marker_id_by_node, _ = build_marker_maps(NODE_GRAPH)
for _marker_node_name, _marker_id in _marker_id_by_node.items():
    _marker_image_path = os.path.join(MARKER_IMAGE_DIR, f"{_marker_node_name}.png")
    if not os.path.exists(_marker_image_path):
        print(f"[경고] {_marker_node_name}용 마커 이미지 없음({_marker_image_path}) — generate_markers.py 먼저 실행 필요")
        continue
    _create_textured_marker_plane(
        prim_path=f"/World/AgvMarkers/{_marker_node_name}",
        position=NODE_GRAPH[_marker_node_name]["position"],
        image_path=_marker_image_path,
        size_m=MARKER_SIZE_M,
    )
print(f"[AGV 마커] {len(_marker_id_by_node)}개 지점에 AprilTag 평면 배치 완료")


world.scene.add_default_ground_plane()
world.reset()

frame = 0
was_playing = False

print("\n" + "=" * 60)
print(f"[환경] Nova Carter {len(ROBOT_HOME_NODE)}대(ROS2 브리지) 생성 완료")
print("  AMR 제어: 외부 fleet_driver.py가 /<robot_id>/cmd_vel 로 담당")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)
    is_playing = world.is_playing()

    if is_playing and not was_playing:
        print("[재생] 시작")

    # 색깔 비콘이 로봇을 따라가게 매 스텝 섀시의 현재 월드 위치를 읽어
    # 비콘 위치를 갱신한다. 부모-자식 프림 계층에 기대지 않고 항상 절대
    # 위치를 다시 계산해서 넣기 때문에, 어떤 링크가 실제로 움직이는지와
    # 무관하게 항상 정확하다.
    if is_playing:
        for _robot_id, _chassis in _robot_chassis_prims.items():
            _chassis_pos, _ = _chassis.get_world_pose()
            _beacon_pos = np.array(_chassis_pos)
            _beacon_pos[2] += _BEACON_HEIGHT_OFFSET_M
            _robot_beacons[_robot_id].set_world_pose(position=_beacon_pos)

    was_playing = is_playing

simulation_app.close()
