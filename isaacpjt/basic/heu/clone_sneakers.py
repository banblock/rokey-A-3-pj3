import asyncio

import omni.kit.app
import omni.timeline
import omni.usd

from pxr import Gf, UsdGeom
from pxr import UsdPhysics

SOURCE_SNEAKERS_PATH = "/World/sneakers"

SPAWN_ROOT_PATH = "/World/SpawnedSneakers"

# 컨베이어 투입 위치
SPAWN_POSITION = Gf.Vec3d(-1.99806, 2.6772, 1.92203)

import omni.graph.core as og

SORTER_A_PATH = (
    "/World/ReturnCell/Sorting/StationA/"
    "CustomConveyor/ConveyorTrack/Sorter/ActionGraph/binary_switch"
)


def find_graph_node(node_path: str):
    for graph in og.get_all_graphs_and_subgraphs():
        for node in graph.get_nodes():
            if node.get_prim_path() == node_path:
                return node

    return None


def set_sorter(enabled: bool) -> bool:
    """StationA 분류기만 제어합니다."""

    node = find_graph_node(SORTER_A_PATH)

    if node is None:
        print("[ERROR] StationA binary_switch를 찾지 못했습니다.")
        print("경로:", SORTER_A_PATH)
        return False

    value_attr = node.get_attribute("inputs:value")

    if not value_attr.is_valid():
        print("[ERROR] inputs:value 속성이 없습니다.")
        print("실제 속성 목록:")

        for attr in node.get_attributes():
            print(" -", attr.get_name())

        return False

    og.Controller.set(value_attr, bool(enabled))
    current_value = og.Controller.get(value_attr)

    print(
        "[StationA]",
        "ON" if current_value else "OFF",
        f"(값: {current_value})"
    )

    return True


def set_position(prim_path: str, position: Gf.Vec3d) -> bool:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print("[ERROR] Prim을 찾지 못했습니다:", prim_path)
        return False

    xformable = UsdGeom.Xformable(prim)

    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(position)

    print("[POSITION SET]", prim_path)
    print("Translate:", position)

    return True


def prepare_sneakers(object_name: str) -> str | None:
    """
    원본 sneakers가 움직이기 전에 복제하고,
    사용할 때까지 비활성화합니다.
    """

    stage = omni.usd.get_context().get_stage()

    source_prim = stage.GetPrimAtPath(SOURCE_SNEAKERS_PATH)

    if not source_prim.IsValid():
        print("[ERROR] sneakers is not detected")
        print("path:", SOURCE_SNEAKERS_PATH)
        return None

    UsdGeom.Xform.Define(stage, SPAWN_ROOT_PATH)

    destination_path = f"{SPAWN_ROOT_PATH}/{object_name}"

    if stage.GetPrimAtPath(destination_path).IsValid():
        stage.RemovePrim(destination_path)

    result = omni.usd.duplicate_prim(
        stage,
        SOURCE_SNEAKERS_PATH,
        destination_path,
        duplicate_layers=False,
    )

    if not result:
        print("[ERROR] 복제 실패:", destination_path)
        return None

    if not set_position(destination_path, SPAWN_POSITION):
        return None

    reset_rigid_body_velocity(destination_path)

    spawned_prim = stage.GetPrimAtPath(destination_path)

    if not spawned_prim.IsValid():
        print("[ERROR] 복제한 sneakers를 찾지 못했습니다.")
        return None

    # 1번 신발이 이동하는 동안 2번 신발은 물리 시뮬레이션에서 제외
    spawned_prim.SetActive(False)

    print("[PREPARED]", destination_path)
    return destination_path


def activate_sneakers(prim_path: str) -> bool:
    """미리 생성해 둔 sneakers를 투입 위치에서 활성화합니다."""

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print("[ERROR] 활성화할 sneakers를 찾지 못했습니다:", prim_path)
        return False

    prim.SetActive(True)

    # 활성화하면서 위치를 다시 확실하게 지정
    if not set_position(prim_path, SPAWN_POSITION):
        return False

    reset_rigid_body_velocity(prim_path)

    print("[ACTIVATED]", prim_path)
    return True


def reset_rigid_body_velocity(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        return

    velocity_attr = prim.GetAttribute("physics:velocity")
    angular_velocity_attr = prim.GetAttribute("physics:angularVelocity")

    if velocity_attr.IsValid():
        velocity_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    if angular_velocity_attr.IsValid():
        angular_velocity_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


async def wait_seconds(seconds: float):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()

    start_time = timeline.get_current_time()

    while timeline.get_current_time() - start_time < seconds:
        await app.next_update_async()


async def run_station_a_demo():
    print("start---------------------------------------------")

    timeline = omni.timeline.get_timeline_interface()

    # 원본 sneakers가 움직이기 전에 2번 신발을 미리 복제
    sneakers_2 = prepare_sneakers("sneakers_2")

    if sneakers_2 is None:
        print("s2 None")
        return

    if not timeline.is_playing():
        timeline.play()
        await omni.kit.app.get_app().next_update_async()

    # 시작 상태: A 분류기 OFF
    set_sorter(False)

    # =====================================================
    # 1번 sneakers: A 분류기 ON
    # =====================================================

    print("[1번] sneakers_1 투입")

    set_sorter(True)
    print("[1번] StationA ON")

    # 분류기가 신발을 밀어내는 동안 유지
    await wait_seconds(10.0)

    set_sorter(False)
    print("[1번] StationA OFF")

    # =====================================================
    # 2번 sneakers: A 분류기 OFF 상태로 직진
    # =====================================================

    if not activate_sneakers(sneakers_2):
        print("s2 activate failed")
        return

    # 활성화가 Stage와 물리 장면에 반영되도록 한 프레임 대기
    await omni.kit.app.get_app().next_update_async()

    # 확실하게 OFF
    set_sorter(False)

    print("[2번] sneakers_2 투입")
    print("[2번] StationA OFF 상태이므로 직진")

    # 2번 신발이 A 분류기를 통과할 시간
    await wait_seconds(6.0)

    print("[완료] 1번은 분류, 2번은 직진")


asyncio.ensure_future(run_station_a_demo())