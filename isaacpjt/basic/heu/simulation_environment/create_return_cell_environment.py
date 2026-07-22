"""Isaac Sim 5.1 - 반품 신발 셀 1차 정적 환경 생성기.

공정 흐름은 -X에서 +X 방향이다.

    투입 -> 투명 검사판(4 카메라) -> 이송 -> A/B/C/D 분류 -> 폐기

이 파일은 레이아웃과 안정적인 Prim 경로를 먼저 정의한다. 컨베이어 구동,
분류기 회전, 카메라 ROS 퍼블리셔와 AMR 주행은 별도 런타임 노드에서 연결한다.
모든 길이는 m, 각도 표기는 degree이다.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import os

import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path


ROOT = "/World/ReturnCell"
OUTPUT_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "return_cell_environment_v1.usd")
)

CONVEYOR_Z = 0.80

# 큐브 기반 컨베이어와 그 위 물체의 전체 높이 조정값.
# 너무 높으면 0.10, 아직 낮으면 0.30처럼 조절한다.
CONVEYOR_Z_OFFSET = 0.20
EFFECTIVE_CONVEYOR_Z = CONVEYOR_Z + CONVEYOR_Z_OFFSET

BELT_WIDTH = 0.62
BELT_THICKNESS = 0.10
SHOE_SIZE = (0.350, 0.240, 0.130)
SHOE_BOX_SIZE = (0.330, 0.200, 0.120)
AMR_SIZE = (0.722, 0.500, 0.556)

# 이후 런타임/ROS 노드가 그대로 사용할 공정 기준점.
WAYPOINTS = {
    "spawn": (-4.50, 0.00, EFFECTIVE_CONVEYOR_Z + 0.144),
    "inspection": (-2.50, 0.00, EFFECTIVE_CONVEYOR_Z + 0.144),
    "sort_a": (1.20, 0.00, EFFECTIVE_CONVEYOR_Z + 0.12),
    "sort_b": (2.40, 0.00, EFFECTIVE_CONVEYOR_Z + 0.12),
    "sort_c": (3.60, 0.00, EFFECTIVE_CONVEYOR_Z + 0.12),
    "sort_d": (4.80, 0.00, EFFECTIVE_CONVEYOR_Z + 0.12),
    "reject": (6.30, 0.00, EFFECTIVE_CONVEYOR_Z + 0.12),
}

COLORS = {
    "frame": (0.18, 0.20, 0.23),
    "belt": (0.08, 0.10, 0.12),
    "rail": (0.45, 0.48, 0.52),
    "glass": (0.35, 0.75, 0.95),
    "camera": (0.12, 0.12, 0.14),
    "a": (0.20, 0.65, 0.95),
    "b": (0.25, 0.80, 0.40),
    "c": (0.95, 0.70, 0.18),
    "d": (0.70, 0.35, 0.90),
    "reject": (0.90, 0.20, 0.18),
    "rack": (0.42, 0.30, 0.18),
    "amr": (0.18, 0.42, 0.72),
    "concrete": (0.42, 0.44, 0.46),
    "wall": (0.72, 0.75, 0.78),
    "steel": (0.12, 0.18, 0.24),
    "safety": (0.95, 0.72, 0.05),
}

WAREHOUSE_LENGTH = 44.0
WAREHOUSE_WIDTH = 32.0
WAREHOUSE_HEIGHT = 10.0

# ============================================================
# Z 높이 기준
# ============================================================
# ConcreteSlab 중심 Z=0.04, 두께=0.08 이므로 윗면은 0.08 m
FLOOR_TOP_Z = 0.08

# CONVEYOR_Z는 벨트의 중심 높이
CONVEYOR_BELT_TOP_Z = EFFECTIVE_CONVEYOR_Z + BELT_THICKNESS / 2.0

# 외부 USD 컨베이어는 자산 내부 원점 위치에 따라 추가 조정이 필요할 수 있다.
CUSTOM_CONVEYOR_Z_OFFSET = 0.20
# 외부 USD 자산은 자체 원점이 달라 별도 오프셋을 사용하며,
# CONVEYOR_Z_OFFSET과 중복 적용하지 않는다.


def center_z_on_surface(surface_z, object_height, gap=0.0):
    """물체 바닥이 surface_z 위에 놓이도록 중심 Z를 계산한다."""
    return float(surface_z) + float(object_height) / 2.0 + float(gap)


# 천장 조명 밝기. 창고 규모가 커서 기존 1800보다 높게 설정한다.
CEILING_LIGHT_INTENSITY = 80000.0

# True이면 검사 대상 신발에 Rigid Body가 적용되어 중력과 충돌의 영향을 받는다.
# 창고, 컨베이어, 벽은 FixedCuboid 그대로 유지한다.
ENABLE_RIGID_PRODUCT = True


def xform(path: str) -> None:
    UsdGeom.Xform.Define(omni.usd.get_context().get_stage(), path)


def box(world, path, position, size, color, collision=True, yaw_deg=0.0):
    """크기가 실제 meter 값인 정적 박스를 만든다."""
    obj = FixedCuboid(
        prim_path=path,
        name=path.replace("/", "_").strip("_").lower(),
        position=np.asarray(position, dtype=np.float64),
        orientation=np.asarray(
            [
                np.cos(np.deg2rad(yaw_deg) / 2.0),
                0.0,
                0.0,
                np.sin(np.deg2rad(yaw_deg) / 2.0),
            ],
            dtype=np.float64,
        ),
        scale=np.asarray(size, dtype=np.float64),
        color=np.asarray(color, dtype=np.float64),
    )
    
    world.scene.add(obj)
    if not collision:
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    return obj



def dynamic_box(world, path, position, size, color, mass=0.5, yaw_deg=0.0):
    """중력과 충돌의 영향을 받는 동적 박스를 만든다."""
    obj = DynamicCuboid(
        prim_path=path,
        name=path.replace("/", "_").strip("_").lower(),
        position=np.asarray(position, dtype=np.float64),
        orientation=np.asarray(
            [
                np.cos(np.deg2rad(yaw_deg) / 2.0),
                0.0,
                0.0,
                np.sin(np.deg2rad(yaw_deg) / 2.0),
            ],
            dtype=np.float64,
        ),
        scale=np.asarray(size, dtype=np.float64),
        color=np.asarray(color, dtype=np.float64),
        mass=float(mass),
    )
    world.scene.add(obj)
    return obj

def transparent_box(
    world, path, position, size, opacity=0.28, render_invisible=False
):
    obj = box(world, path, position, size, COLORS["glass"], collision=True)
    stage = omni.usd.get_context().get_stage()
    material_path = Sdf.Path(path + "_GlassMaterial")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*COLORS["glass"])
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(path)
    ).Bind(material)
    if render_invisible:
        # 렌더/RGB/Depth 영상에는 전혀 나타나지 않지만 물리 충돌은 유지한다.
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    return obj


# def conveyor(world, path, center, length, width=BELT_WIDTH, axis="x"):
#     """높이 오프셋을 적용한 벨트와 양쪽 프레임 컨베이어."""
    # x, y = center
    # conveyor_center_z = EFFECTIVE_CONVEYOR_Z

    # belt_size = (
    #     (length, width, BELT_THICKNESS)
    #     if axis == "x"
    #     else (width, length, BELT_THICKNESS)
    # )

    # box(
    #     world,
    #     path + "/Belt",
    #     (x, y, conveyor_center_z),
    #     belt_size,
    #     COLORS["belt"],
    # )

    # offset = width / 2.0 + 0.035
    # frame_height = 0.14

    # for side, sign in (("LeftFrame", 1.0), ("RightFrame", -1.0)):
    #     if axis == "x":
    #         position = (
    #             x,
    #             y + sign * offset,
    #             conveyor_center_z,
    #         )
    #         size = (
    #             length,
    #             0.07,
    #             frame_height,
    #         )
    #     else:
    #         position = (
    #             x + sign * offset,
    #             y,
    #             conveyor_center_z,
    #         )
    #         size = (
    #             0.07,
    #             length,
    #             frame_height,
    #         )

    #     box(
    #         world,
    #         path + "/" + side,
    #         position,
    #         size,
    #         COLORS["frame"],
    #     )

    # print(
    #     f"[컨베이어 높이] {path} 중심 Z={conveyor_center_z:.3f}, "
    #     f"벨트 윗면 Z={conveyor_center_z + BELT_THICKNESS / 2.0:.3f}"
    # )
convey_path = "/home/rokey/Collected_World0/convey/convey.usd"


def load_start_conveyor(path, pos, yaw_deg=0.0):
    """시작 직선 컨베이어 USD를 지정된 위치에 불러온다."""
    if not os.path.isfile(convey_path):
        print(f"[경고] 지정된 USD 파일을 찾을 수 없습니다: {convey_path}")
        return None

    stage = omni.usd.get_context().get_stage()

    add_reference_to_stage(
        usd_path=convey_path,
        prim_path=path,
    )
    simulation_app.update()

    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        print(f"[오류] 컨베이어 Prim 생성 실패: {path}")
        return None

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    if yaw_deg != 0.0:
        xform.AddRotateZOp().Set(float(yaw_deg))

    print(
        f"[시작 컨베이어] {path} "
        f"위치=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
        f"회전={yaw_deg:.1f}도"
    )

    return prim
    

def warehouse_shell(world):
    """반품 셀을 둘러싸는 물류창고 바닥, 벽, 골조와 조명을 만든다."""
    base = ROOT + "/Warehouse"
    half_x = WAREHOUSE_LENGTH / 2.0
    half_y = WAREHOUSE_WIDTH / 2.0

    # 기본 ground plane 위에 실제 창고 슬래브를 얹는다.
    box(world, base + "/Floor/ConcreteSlab", (0.0, 0.0, 0.04),
        (WAREHOUSE_LENGTH, WAREHOUSE_WIDTH, 0.08), COLORS["concrete"])

    wall_z = WAREHOUSE_HEIGHT / 2.0
    wall_t = 0.12
    box(world, base + "/Walls/North", (0.0, half_y, wall_z),
        (WAREHOUSE_LENGTH, wall_t, WAREHOUSE_HEIGHT), COLORS["wall"])
    box(world, base + "/Walls/South", (0.0, -half_y, wall_z),
        (WAREHOUSE_LENGTH, wall_t, WAREHOUSE_HEIGHT), COLORS["wall"])
    box(world, base + "/Walls/West", (-half_x, 0.0, wall_z),
        (wall_t, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT), COLORS["wall"])

    # 동쪽 벽에는 폭 3 m, 높이 3.2 m의 물류 출입구를 남긴다.
    door_width = 3.0
    side_width = (WAREHOUSE_WIDTH - door_width) / 2.0
    for name, y in (("EastNorth", (door_width + side_width) / 2.0),
                    ("EastSouth", -(door_width + side_width) / 2.0)):
        box(world, base + "/Walls/" + name, (half_x, y, wall_z),
            (wall_t, side_width, WAREHOUSE_HEIGHT), COLORS["wall"])
    header_height = WAREHOUSE_HEIGHT - 3.2
    box(world, base + "/Walls/EastDoorHeader",
        (half_x, 0.0, 3.2 + header_height / 2.0),
        (wall_t, door_width, header_height), COLORS["wall"])

    # 철골 기둥과 천장 횡보.
    column_xs = (-10.6, -5.3, 0.0, 5.3, 10.6)
    for index, column_x in enumerate(column_xs, start=1):
        for side, column_y in (("N", 7.65), ("S", -7.65)):
            box(world, f"{base}/Structure/Column{index}{side}",
                (column_x, column_y, WAREHOUSE_HEIGHT / 2.0),
                (0.18, 0.18, WAREHOUSE_HEIGHT), COLORS["steel"])
        box(world, f"{base}/Structure/RoofBeam{index}",
            (column_x, 0.0, WAREHOUSE_HEIGHT - 0.12),
            (0.18, 15.4, 0.24), COLORS["steel"])

    # 천장 부모 Prim을 명시적으로 생성한다.
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, base + "/Ceiling")

    # 얇은 천장 패널과 긴 LED 등기구. 등은 렌더링용 무충돌 형상이다.
    box(world, base + "/Ceiling/RoofPanel", (0.0, 0.0, WAREHOUSE_HEIGHT),
        (WAREHOUSE_LENGTH, WAREHOUSE_WIDTH, 0.10), (0.52, 0.56, 0.60))

    light_positions = (
        (-8.0, -4.0), (-8.0, 4.0),
        (-2.8, -4.0), (-2.8, 4.0),
        (2.8, -4.0), (2.8, 4.0),
        (8.0, -4.0), (8.0, 4.0),
        (13.0, 0.0),  # 추가 광원
    )

    for index, (light_x, light_y) in enumerate(light_positions, start=1):
        box(world, f"{base}/Ceiling/LED{index:02d}",
            (light_x, light_y, WAREHOUSE_HEIGHT - 0.22),
            (1.8, 0.16, 0.05), (1.0, 0.96, 0.78), collision=False)
        light = UsdLux.RectLight.Define(
            omni.usd.get_context().get_stage(), f"{base}/Ceiling/Light{index:02d}"
        )
        light.CreateWidthAttr(3.0)
        light.CreateHeightAttr(0.4)
        light.CreateIntensityAttr(CEILING_LIGHT_INTENSITY)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.82))
        light_xform = UsdGeom.Xformable(light.GetPrim())
        light_xform.AddTranslateOp().Set(
            Gf.Vec3d(light_x, light_y, WAREHOUSE_HEIGHT - 0.25)
        )
        light_xform.AddRotateXOp().Set(180.0)
        print(
            f"[조명 생성] {base}/Ceiling/Light{index:02d} "
            f"위치=({light_x:.1f}, {light_y:.1f}, {WAREHOUSE_HEIGHT - 0.25:.2f})"
        )

    # AMR 통행로와 작업구역을 눈으로 구분하는 바닥 마킹.
    box(world, base + "/FloorMarkings/AMRAisleNorth", (0.0, 4.25, 0.087),
        (19.0, 0.055, 0.012), COLORS["safety"], collision=False)
    box(world, base + "/FloorMarkings/AMRAisleSouth", (0.0, -4.25, 0.087),
        (19.0, 0.055, 0.012), COLORS["safety"], collision=False)
    for index, marking_x in enumerate((-8.5, -3.0, 2.5, 8.0), start=1):
        box(world, f"{base}/FloorMarkings/CrossLine{index}",
            (marking_x, 0.0, 0.087), (0.055, 12.5, 0.012),
            COLORS["safety"], collision=False)


# def curved_branch(world, path, start_x, direction, color):
#     """메인 라인에서 +/-Y로 90도 꺾이는 곡선 롤러 분기를 표현한다.

#     여러 개의 짧은 롤러를 원호에 접하도록 놓는다. 현재는 정적 형상이며,
#     후속 런타임에서 물품 경로 또는 실제 conveyor surface로 교체할 수 있다.
#     """
#     radius = 0.48
#     roller_length = 0.30
#     roller_width = 0.075
#     segment_count = 9
#     center_y = direction * radius

#     for index in range(segment_count):
#         ratio = index / (segment_count - 1)
#         angle_deg = -90.0 + 90.0 * ratio if direction > 0 else 90.0 - 90.0 * ratio
#         angle_rad = np.deg2rad(angle_deg)
#         x = start_x + radius * np.cos(angle_rad)
#         y = center_y + radius * np.sin(angle_rad)
#         tangent_deg = direction * 90.0 * ratio
#         box(
#             world,
#             path + f"/Roller{index + 1:02d}",
#             (x, y, EFFECTIVE_CONVEYOR_Z + 0.015),
#             (roller_length, roller_width, 0.075),
#             color,
#             yaw_deg=tangent_deg,
#         )

#     return start_x + radius, direction * radius


def tote_bin(
    world,
    path,
    center,
    color=(0.48, 0.50, 0.53),
):
    """
    GUI에서 토트 전체를 한 번에 이동할 수 있는 개방형 플라스틱 토트.

    Stage에서 path에 해당하는 부모 Xform을 선택한 뒤
    Move Tool(W)로 이동하면 모든 부품이 함께 움직인다.
    """
    stage = omni.usd.get_context().get_stage()

    x, y = center

    outer_x = 0.78
    outer_y = 0.62
    height = 0.52

    wall_t = 0.045
    base_t = 0.055

    # ========================================================
    # 토트 전체를 담는 부모 Xform
    # ========================================================
    tote_root = UsdGeom.Xform.Define(stage, path)
    tote_xform = UsdGeom.Xformable(tote_root.GetPrim())

    tote_xform.ClearXformOpOrder()
    tote_xform.AddTranslateOp().Set(
        Gf.Vec3d(
            x,
            y,
            FLOOR_TOP_Z,
        )
    )

    # 아래에서 만드는 모든 부품 좌표는 부모 기준 로컬 좌표이다.
    local_bottom_z = base_t / 2.0

    box(
        world,
        path + "/Bottom",
        (0.0, 0.0, local_bottom_z),
        (outer_x, outer_y, base_t),
        color,
    )

    wall_height = height - base_t
    wall_z = base_t + wall_height / 2.0

    box(
        world,
        path + "/WallEast",
        (
            outer_x / 2.0 - wall_t / 2.0,
            0.0,
            wall_z,
        ),
        (
            wall_t,
            outer_y,
            wall_height,
        ),
        color,
    )

    box(
        world,
        path + "/WallWest",
        (
            -outer_x / 2.0 + wall_t / 2.0,
            0.0,
            wall_z,
        ),
        (
            wall_t,
            outer_y,
            wall_height,
        ),
        color,
    )

    box(
        world,
        path + "/WallNorth",
        (
            0.0,
            outer_y / 2.0 - wall_t / 2.0,
            wall_z,
        ),
        (
            outer_x,
            wall_t,
            wall_height,
        ),
        color,
    )

    box(
        world,
        path + "/WallSouth",
        (
            0.0,
            -outer_y / 2.0 + wall_t / 2.0,
            wall_z,
        ),
        (
            outer_x,
            wall_t,
            wall_height,
        ),
        color,
    )

    # ========================================================
    # 상단 테두리
    # ========================================================
    rim_z = height

    rim_data = (
        (
            "RimEast",
            (outer_x / 2.0, 0.0, rim_z),
            (0.07, outer_y + 0.05, 0.07),
        ),
        (
            "RimWest",
            (-outer_x / 2.0, 0.0, rim_z),
            (0.07, outer_y + 0.05, 0.07),
        ),
        (
            "RimNorth",
            (0.0, outer_y / 2.0, rim_z),
            (outer_x + 0.05, 0.07, 0.07),
        ),
        (
            "RimSouth",
            (0.0, -outer_y / 2.0, rim_z),
            (outer_x + 0.05, 0.07, 0.07),
        ),
    )

    for name, position, size in rim_data:
        box(
            world,
            path + "/" + name,
            position,
            size,
            color,
        )

    # ========================================================
    # 보강 리브
    # ========================================================
    rib_height = 0.37
    rib_z = 0.30

    for index, dx in enumerate((-0.28, 0.28), start=1):
        box(
            world,
            path + f"/RibNorth{index}",
            (
                dx,
                outer_y / 2.0 + 0.006,
                rib_z,
            ),
            (
                0.055,
                0.028,
                rib_height,
            ),
            (0.38, 0.40, 0.43),
        )

        box(
            world,
            path + f"/RibSouth{index}",
            (
                dx,
                -outer_y / 2.0 - 0.006,
                rib_z,
            ),
            (
                0.055,
                0.028,
                rib_height,
            ),
            (0.38, 0.40, 0.43),
        )

    # GUI에서 쉽게 식별할 수 있도록 메타데이터 추가
    tote_root.GetPrim().CreateAttribute(
        "returnCell:role",
        Sdf.ValueTypeNames.String,
    ).Set(
        "movable_tote"
    )

    print(
        f"[GUI 이동 가능 토트] {path} "
        f"초기 위치=({x:.2f}, {y:.2f}, {FLOOR_TOP_Z:.2f})"
    )


def create_storage_slots(route, tote_path, center):
    """토트 안에 2열 x 2행 x 3층, 총 12개의 상자 정렬 위치를 만든다."""
    stage = omni.usd.get_context().get_stage()
    slots_root = tote_path + "/StorageSlots"
    UsdGeom.Xform.Define(stage, slots_root)
    slot_index = 1
    for layer in range(3):
        for dx in (-0.18, 0.18):
            for dy in (-0.12, 0.12):
                slot = UsdGeom.Xform.Define(stage, slots_root + f"/Slot{slot_index:02d}")
                slot.AddTranslateOp().Set(
                    Gf.Vec3d(center[0] + dx, center[1] + dy, 0.25 + layer * 0.13)
                )
                slot.GetPrim().CreateAttribute(
                    "returnCell:route", Sdf.ValueTypeNames.String
                ).Set(route)
                slot.GetPrim().CreateAttribute(
                    "returnCell:occupied", Sdf.ValueTypeNames.Bool
                ).Set(False)
                slot_index += 1


def transfer_trigger(world, route, path, position, size):
    """화면에는 보이지 않는 컨베이어 끝 이송 판정용 가상벽."""
    transparent_box(
        world, path, position, size, opacity=0.0, render_invisible=True
    )
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    prim.CreateAttribute("returnCell:role", Sdf.ValueTypeNames.String).Set(
        "storage_transfer_trigger"
    )
    prim.CreateAttribute("returnCell:route", Sdf.ValueTypeNames.String).Set(route)


def inspection_station(world):
    """두 레일 사이의 투명판과 4방향 카메라 검사 구간."""
    base = ROOT + "/Inspection"
    cx = WAYPOINTS["inspection"][0]

    # 비전 구간은 일반 컨베이어 벨트 대신 두 개 레일과 투명판을 사용한다.
    for name, y in (("RailLeft", 0.255), ("RailRight", -0.255)):
        box(world, base + "/Rails/" + name, (cx, y, EFFECTIVE_CONVEYOR_Z),
            (1.40, 0.09, 0.10), COLORS["rail"])
    
    
    # 요청 위치: 위/아래 z +/-0.5, 정면/후면 y +/-0.5 (검사 중심 기준).
    cameras = {
        "Top": ((cx, 0.0, EFFECTIVE_CONVEYOR_Z + 0.62), (180.0, 0.0, 0.0)),
        "Bottom": ((cx, 0.0, EFFECTIVE_CONVEYOR_Z - 0.50), (0.0, 0.0, 0.0)),
        "Front": ((cx, -0.50, EFFECTIVE_CONVEYOR_Z + 0.20), (90.0, 0.0, 0.0)),
        "Rear": ((cx, 0.50, EFFECTIVE_CONVEYOR_Z + 0.20), (-90.0, 0.0, 0.0)),
    }
    # D455 메쉬(add_reference_to_stage)를 이 루프 안에서 로드하면 트레이스백 없이
    # 프로세스가 죽는 문제가 있어 원인 조사 전까지 비활성화한다. 실제 촬영에 쓰이는
    # UsdGeom.Camera는 아래에서 항상 별도로 만들어지므로 기능에는 영향이 없다.
    LOAD_D455_MESH = False
    assets_root = get_assets_root_path()
    d455_usd = None

    
    if LOAD_D455_MESH and assets_root:
        # Isaac Sim 5.1 공식 Intel RealSense D455 자산.
        d455_usd = assets_root + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
        print(f"[D455 자산] {d455_usd}")
    else:
        print("[정보] D455 메쉬 로드를 건너뛰고 단순 하우징 placeholder를 사용합니다.")

    for name, (position, rotation_deg) in cameras.items():
        camera_path = base + "/Cameras/" + name + "D455"
        if d455_usd:
            asset_path = camera_path + "/D455Asset"
            add_reference_to_stage(usd_path=d455_usd, prim_path=asset_path)
            simulation_app.update()
            asset_prim = omni.usd.get_context().get_stage().GetPrimAtPath(asset_path)
            asset_xform = UsdGeom.Xformable(asset_prim)
            asset_xform.ClearXformOpOrder()
            asset_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            asset_xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_deg))
            metadata_prim = asset_prim
        else:
            box(world, camera_path + "/Housing", position, (0.12, 0.04, 0.035),
                COLORS["camera"], collision=False)
            metadata_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
                camera_path + "/Housing"
            )

        # 실제 자산 로드 실패 시에도 위치 기준으로 쓸 단순 렌더 카메라는 유지한다.
        camera = UsdGeom.Camera.Define(
            omni.usd.get_context().get_stage(), camera_path + "/Camera"
        )
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_deg))
        camera.CreateFocalLengthAttr(1.93)
        camera.CreateHorizontalApertureAttr(3.895)
        camera.CreateVerticalApertureAttr(2.453)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
        metadata_prim.CreateAttribute("returnCell:role", Sdf.ValueTypeNames.String).Set(
            "inspection_camera"
        )
        metadata_prim.CreateAttribute(
            "returnCell:rotationDeg", Sdf.ValueTypeNames.Float3
        ).Set(
            Gf.Vec3f(*rotation_deg)
        )
        print(f"[진단] 카메라 {name} 생성 완료")

    # 촬영 중 컨베이어 정지 판단에 사용할 가상 감지 영역.
    box(world, base + "/Triggers/PhotoStopZone", (cx, 0.0, 0.025),
        (0.60, 0.62, 0.05), (0.95, 0.85, 0.10), collision=False)


def sorter_and_buffers(world):
    """직렬 4분류기, A-D 분기, 폐기 직진 라인을 만든다."""
    
    # 불러올 커스텀 컨베이어/스위치 USD 파일 경로
    conveyor_usd_path = "/home/rokey/Collected_World0/convey_binary_switch.usd"
    
    # 파일 존재 여부 확인
    if not os.path.exists(conveyor_usd_path):
        print(f"[경고] 지정된 USD 경로를 찾을 수 없습니다: {conveyor_usd_path}")

    def load_custom_conveyor(path, pos, yaw_deg=0.0):
        """USD 에셋을 지정된 위치와 회전값으로 불러오는 헬퍼 함수"""
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, path)
        if os.path.exists(conveyor_usd_path):
            add_reference_to_stage(usd_path=conveyor_usd_path, prim_path=path)
            simulation_app.update()
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
            if yaw_deg != 0.0:
                xform.AddRotateZOp().Set(float(yaw_deg))
            print(f"[커스텀 컨베이어] {path} 위치={pos}")
            return prim
        return None

    # 1. 메인 이송 구간 및 Sorter Main 구간을 기존 큐브 대신 USD 에셋으로 대체
    # (필요에 따라 메인 라인 위치 좌표는 기존 값(-0.15, 3.95 등)을 활용합니다)

    # 2. A, B, C, D 분류 스테이션 구성
    station_data = (
        ("A", 1.60, 1.55, COLORS["a"], 180.0),
        ("B", 4.60, 1.55, COLORS["b"], 180.0),
        ("C", 7.60, 1.55, COLORS["c"], 180.0), 
        ("D", 10.60, 1.55, COLORS["d"], 180.0),
    )
    
    for label, sx, buffer_y, color, yaw in station_data:
        base = ROOT + "/Sorting/Station" + label
        
        # 각 스테이션 위치에 커스텀 컨베이어/분기 USD 배치
        load_custom_conveyor(
            base + "/CustomConveyor",
            (sx, 0.0, CONVEYOR_Z + CUSTOM_CONVEYOR_Z_OFFSET),
            yaw_deg=yaw,
        )
        print(f"[진단] Station{label} CustomConveyor 로드 완료")
        
        # 센서 및 물품이 토트로 빠져나가는 지점의 트리거/버퍼 구조는 유지
        box(world, base + "/PresenceSensor", (sx + 0.34, 0.0, EFFECTIVE_CONVEYOR_Z + 0.16),
            (0.035, BELT_WIDTH, 0.30), color, collision=False)

        # 기존 큐브 기반 curved_branch 대신 토트 및 이송 트리거 연결
        tote_path = ROOT + "/Buffers/Buffer" + label + "/Tote"
        tote_bin(world, tote_path, (sx, buffer_y))
        create_storage_slots(label, tote_path, (sx, buffer_y))
        
        direction = 1.0 if buffer_y > 0 else -1.0
        trigger_y = buffer_y - direction * 0.36
        transfer_trigger(
            world,
            label,
            ROOT + "/Buffers/Buffer" + label + "/TransferTrigger",
            (sx, trigger_y, EFFECTIVE_CONVEYOR_Z + 0.22),
            (0.70, 0.035, 0.44),
        )

    # 3. 어떤 분류기에도 걸리지 않은 물품이 직진해 들어가는 폐기(Reject) 버퍼 라인
    reject_tote_path = ROOT + "/Buffers/Reject/Tote"
    tote_bin(world, reject_tote_path, (6.55, 0.0), color=(0.48, 0.35, 0.35))
    create_storage_slots("REJECT", reject_tote_path, (6.55, 0.0))
    transfer_trigger(
        world,
        "REJECT",
        ROOT + "/Buffers/Reject/TransferTrigger",
        (6.13, 0.0, EFFECTIVE_CONVEYOR_Z + 0.22),
        (0.035, 0.70, 0.44),
    )
    print("[진단] Reject 버퍼 완료")


def rack(world, path, center, color):
    """신발 상자를 여러 개 적재하는 산업용 팔레트 랙."""
    x, y = center
    width, depth, height = 1.60, 0.90, 2.30
    for x_name, dx in (("L", -width / 2), ("R", width / 2)):
        for y_name, dy in (("F", -depth / 2), ("B", depth / 2)):
            box(world, path + f"/Post{x_name}{y_name}",
                (x + dx, y + dy, height / 2),
                (0.075, 0.075, height), (0.12, 0.28, 0.52))
    for index, z in enumerate((0.25, 0.95, 1.65)):
        box(world, path + f"/Shelf{index + 1}", (x, y, z),
            (width, depth, 0.075), color)
        # 선반 전면/후면의 하중 지지 빔.
        for beam_name, dy in (("Front", -depth / 2), ("Back", depth / 2)):
            box(world, path + f"/Beam{index + 1}{beam_name}",
                (x, y + dy, z + 0.04), (width, 0.10, 0.16),
                (0.92, 0.35, 0.08))


def storage_and_amrs(world):
    """컨베이어와 랙 사이에 넓은 Nova Carter 주행 통로를 둔다."""
    locations = {
        "A": (1.68, 6.20, COLORS["a"]),
        "B": (3.48, 6.20, COLORS["b"]),
        "C": (4.08, -6.20, COLORS["c"]),
        "D": (5.88, -6.20, COLORS["d"]),
        "Reject": (7.40, 6.20, COLORS["reject"]),
    }
    assets_root = get_assets_root_path()
    nova_usd = None
    if assets_root:
        nova_usd = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        print(f"[Nova Carter 자산] {nova_usd}")

    for label, (x, y, color) in locations.items():
        rack(world, ROOT + "/Storage/Rack" + label, (x, y), color)

        amr_y = 4.25 if y > 0 else -4.25
        amr_path = ROOT + "/AMRs/AMR_" + label
        if nova_usd:
            add_reference_to_stage(usd_path=nova_usd, prim_path=amr_path)
            simulation_app.update()
            amr_prim = omni.usd.get_context().get_stage().GetPrimAtPath(amr_path)
            amr_xform = UsdGeom.Xformable(amr_prim)
            amr_xform.ClearXformOpOrder()
            amr_xform.AddTranslateOp().Set(Gf.Vec3d(x, amr_y, 0.0))
            amr_xform.AddRotateZOp().Set(90.0 if y > 0 else -90.0)
        else:
            box(world, amr_path, (x, amr_y, AMR_SIZE[2] / 2.0), AMR_SIZE,
                COLORS["amr"])
        box(world, ROOT + "/Zones/AMRWait_" + label, (x, amr_y, 0.012),
            (1.10, 0.90, 0.024), color, collision=False)
        box(world, ROOT + "/Zones/RackDock_" + label,
            (x, y + (-0.85 if y > 0 else 0.85), 0.012),
            (1.10, 0.75, 0.024), color, collision=False)


def add_product_placeholders(world):
    """투입부터 신발과 함께 이동하는 투명 검사 캐리어를 만든다."""
    sx, sy, _ = WAYPOINTS["spawn"]
    assembly_path = ROOT + "/Products/InspectionCarrierAssembly"
    UsdGeom.Xform.Define(
        omni.usd.get_context().get_stage(),
        assembly_path,
    )

    carrier_size = (0.52, 0.38, 0.018)
    carrier_height = carrier_size[2]

    # 투명 캐리어의 바닥이 벨트 윗면보다 2 mm 위에 놓이도록 계산
    carrier_z = center_z_on_surface(
        CONVEYOR_BELT_TOP_Z,
        carrier_height,
        gap=0.002,
    )

    carrier = transparent_box(
        world,
        assembly_path + "/TransparentCarrier",
        (sx, sy, carrier_z),
        carrier_size,
        render_invisible=True,
    )

    carrier_top_z = carrier_z + carrier_height / 2.0

    # 신발 바닥이 캐리어 윗면보다 2 mm 위에 놓이도록 계산
    shoe_center_z = center_z_on_surface(
        carrier_top_z,
        SHOE_SIZE[2],
        gap=0.002,
    )

    if ENABLE_RIGID_PRODUCT:
        shoe = dynamic_box(
            world,
            assembly_path + "/Shoe",
            (sx, sy, shoe_center_z),
            SHOE_SIZE,
            (0.10, 0.35, 0.85),
            mass=0.7,
        )
        print(
            f"[Rigid Body] 검사 신발 DynamicCuboid 적용, "
            f"초기 중심 Z={shoe_center_z:.3f}"
        )
    else:
        shoe = box(
            world,
            assembly_path + "/Shoe",
            (sx, sy, shoe_center_z),
            SHOE_SIZE,
            (0.10, 0.35, 0.85),
        )

    stage = omni.usd.get_context().get_stage()

    stage.GetPrimAtPath(
        assembly_path + "/TransparentCarrier"
    ).CreateAttribute(
        "returnCell:role",
        Sdf.ValueTypeNames.String,
    ).Set(
        "moving_inspection_carrier"
    )

    stage.GetPrimAtPath(
        assembly_path + "/Shoe"
    ).CreateAttribute(
        "returnCell:carrier",
        Sdf.ValueTypeNames.String,
    ).Set(
        assembly_path
    )

    # AMR 위 신발 상자는 기존 고정 placeholder로 유지
    box(
        world,
        ROOT + "/Products/ShoeBoxOnAMR",
        (1.20, 1.80, 0.62),
        SHOE_BOX_SIZE,
        (0.68, 0.48, 0.26),
    )

    print(
        f"[높이 진단] 벨트 윗면={CONVEYOR_BELT_TOP_Z:.3f}, "
        f"캐리어 중심={carrier_z:.3f}, "
        f"신발 중심={shoe_center_z:.3f}"
    )


def add_metadata():
    """ROS/런타임 코드가 문자열 검색 없이 읽을 수 있는 씬 메타데이터."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(ROOT)
    prim.CreateAttribute("returnCell:flowDirection", Sdf.ValueTypeNames.String).Set("-X_to_+X")
    prim.CreateAttribute("returnCell:units", Sdf.ValueTypeNames.String).Set("m_deg")
    prim.CreateAttribute("returnCell:inspectionCameraCount", Sdf.ValueTypeNames.Int).Set(4)
    prim.CreateAttribute("returnCell:categories", Sdf.ValueTypeNames.StringArray).Set(
        ["A", "B", "C", "D", "REJECT"]
    )
    for name, position in WAYPOINTS.items():
        prim.CreateAttribute(
            "returnCell:waypoint:" + name, Sdf.ValueTypeNames.Double3
        ).Set(Gf.Vec3d(*position))


def create_environment(world):
    stage = omni.usd.get_context().get_stage()
    for path in (
        ROOT, ROOT + "/Conveyors", ROOT + "/Inspection", ROOT + "/Inspection/Rails",
        ROOT + "/Inspection/Cameras", ROOT + "/Inspection/Triggers", ROOT + "/Sorting",
        ROOT + "/Buffers", ROOT + "/Storage", ROOT + "/AMRs", ROOT + "/Zones",
        ROOT + "/Products", ROOT + "/Warehouse", ROOT + "/Warehouse/Floor",
        ROOT + "/Warehouse/Walls", ROOT + "/Warehouse/Structure",
        ROOT + "/Warehouse/Ceiling", ROOT + "/Warehouse/FloorMarkings",
    ):
        UsdGeom.Xform.Define(stage, path)

    world.scene.add_default_ground_plane()
    print("[진단] warehouse_shell 시작")
    warehouse_shell(world)
    ceiling_prim = stage.GetPrimAtPath(ROOT + "/Warehouse/Ceiling")
    print(f"[진단] Ceiling Prim 존재: {ceiling_prim.IsValid()}")
    print("[진단] warehouse_shell 완료")
    load_start_conveyor(ROOT + "/Conveyors/Input", (-4.15, 0.0, CONVEYOR_Z+CUSTOM_CONVEYOR_Z_OFFSET), yaw_deg=180.0)
    print("[진단] Input 컨베이어 완료")
    inspection_station(world)
    print("[진단] inspection_station 완료")
    sorter_and_buffers(world)
    print("[진단] sorter_and_buffers 완료")
    storage_and_amrs(world)
    print("[진단] storage_and_amrs 완료")
    add_product_placeholders(world)
    print("[진단] add_product_placeholders 완료")
    add_metadata()
    print("[진단] add_metadata 완료")


def main():
    print("[1/4] Isaac Sim World 생성")
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0,
                  rendering_dt=1.0 / 60.0)
    print("[2/4] 창고 및 반품 셀 생성")
    create_environment(world)
    world.reset()

    # 기본 카메라는 원점의 설비 내부에 놓일 수 있으므로 창고 내부 조감도로 이동한다.
    set_camera_view(
        eye=np.array([10.0, -7.0, 5.2], dtype=np.float64),
        target=np.array([1.0, 0.0, 1.0], dtype=np.float64),
        camera_prim_path="/OmniverseKit_Persp",
    )
    print("[3/4] 뷰포트 및 조명 렌더링")
    for _ in range(60):
        world.step(render=True)

    os.makedirs(os.path.dirname(OUTPUT_USD), exist_ok=True)
    if not omni.usd.get_context().get_stage().GetRootLayer().Export(OUTPUT_USD):
        raise RuntimeError(f"USD 저장 실패: {OUTPUT_USD}")
    print(f"[4/4 완료] 반품 셀 환경 저장: {OUTPUT_USD}")
    print(f"[기준 Prim] {ROOT}")

    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()