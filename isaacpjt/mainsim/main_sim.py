from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
})


import numpy as np
import omni.timeline
import omni.usd

from pxr import Gf, UsdGeom
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import open_stage


# ============================================================
# USD 및 Prim 경로
# ============================================================

USD_PATH = "/home/rokey/cobot3_ws/isaacpjt/basic/heu/stage_v7.usd"

INTERFACE_PRIM_PATH = "/World/ReturnCell"

SHOE_POOL_PATHS = [
    "/World/ObjectPool/Shoe_01",
    "/World/ObjectPool/Shoe_02",
    "/World/ObjectPool/Shoe_03",
    # TODO: 필요한 신발 Prim 경로 추가
]

SPAWN_POSITION = np.array([
    -2.03979,  #  X
    2.78587,  #   Y
    1.92203,  #   Z
])

SPAWN_ROTATION_DEG = np.array([
    0.0,  # Rotate X
    0.0,  # Rotate Y
    0.0,  # Rotate Z
])

CAMERA_TRAVEL_TIME = 1.55


# ============================================================
# Isaac Sim 초기화
# ============================================================

enable_extension("isaacsim.ros2.bridge")

open_stage(USD_PATH)

world = World(stage_units_in_meters=1.0)
world.reset()

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

interface_prim = stage.GetPrimAtPath(INTERFACE_PRIM_PATH)


# ============================================================
# TODO: Action Graph 공유 Attribute
# ============================================================

# Standalone -> Action Graph
conveyor_run_attr = interface_prim.GetAttribute(
    "standalone:conveyorRun"
)

vision_request_attr = interface_prim.GetAttribute(
    "standalone:visionRequest"
)

# Action Graph -> Standalone
sorter_end_sensor_attr = interface_prim.GetAttribute(
    "actionGraph:sorterEndSensor"
)


# ============================================================
# Object Pool 기능
# ============================================================

shoe_index = 0
current_shoe_path = None


def spawn_next_shoe():
    global shoe_index
    global current_shoe_path

    if shoe_index >= len(SHOE_POOL_PATHS):
        return

    current_shoe_path = SHOE_POOL_PATHS[shoe_index]

    shoe_prim = stage.GetPrimAtPath(current_shoe_path)
    shoe_prim.SetActive(True)

    xform_api = UsdGeom.XformCommonAPI(shoe_prim)

    xform_api.SetTranslate(
        Gf.Vec3d(
            float(SPAWN_POSITION[0]),
            float(SPAWN_POSITION[1]),
            float(SPAWN_POSITION[2]),
        )
    )

    xform_api.SetRotate(
        Gf.Vec3f(
            float(SPAWN_ROTATION_DEG[0]),
            float(SPAWN_ROTATION_DEG[1]),
            float(SPAWN_ROTATION_DEG[2]),
        ),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    shoe_index += 1


# ============================================================
# 실행
# ============================================================

timeline.play()

spawn_next_shoe()

conveyor_run_attr.Set(True)
vision_request_attr.Set(False)

spawn_time = world.current_time
previous_sensor_value = False
camera_request_sent = False


while simulation_app.is_running():
    world.step(render=True)

    current_time = world.current_time

    # --------------------------------------------------------
    # 신발이 촬영 위치에 도착
    # --------------------------------------------------------

    if (
        not camera_request_sent
        and current_time - spawn_time >= CAMERA_TRAVEL_TIME
    ):
        conveyor_run_attr.Set(False)
        vision_request_attr.Set(True)

        camera_request_sent = True

    # --------------------------------------------------------
    # 분류기 끝 센서 감지
    # --------------------------------------------------------

    sensor_value = bool(sorter_end_sensor_attr.Get())

    if sensor_value and not previous_sensor_value:
        spawn_next_shoe()

        conveyor_run_attr.Set(True)
        vision_request_attr.Set(False)

        spawn_time = current_time
        camera_request_sent = False

    previous_sensor_value = sensor_value


timeline.stop()
simulation_app.close()