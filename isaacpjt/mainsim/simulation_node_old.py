"""Isaac Sim 신발 순환 제어.

기존 ROS2 그래프는 유지하되, Isaac Sim 5.1에서 오류를 내는 오래된 중복
ConveyorBeltGraph와 고정 신발 활성/비활성 노드만 World.reset() 전에 끕니다.
신발 랜덤 활성화와 Trigger 도착 후 복귀는 Python update()에서 처리합니다.
"""

from __future__ import annotations

import random
from typing import Final

import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom


WORKING_CONVEYOR_GRAPH: Final[str] = (
    "/World/ReturnCell/Conveyors/Input/ConveyorTrack/ConveyorBeltGraph"
)
TRIGGER_GRAPH_PATH: Final[str] = "/World/ShoeTrigger/ActionGraph"
TRIGGER_PATH: Final[str] = "/World/ShoeTrigger"
SHOES_ROOT_PATH: Final[str] = "/World/shoes"

CONVEYOR_RESTART_DELAY_SEC: Final[float] = 2.0
SHOE_ACTIVATION_DELAY_SEC: Final[float] = 10.0

SHOE_PATHS: Final[tuple[str, ...]] = (
    "/World/shoes/sneakers",
    "/World/shoes/sneaker_0001_red_240_ok_01",
    "/World/shoes/sneaker_0001_red_240_ok_02",
)

STALE_GRAPH_PATHS: Final[tuple[str, ...]] = (
    "/World/ReturnCell/Conveyors/Input/ConveyorBeltGraph",
    "/World/ReturnCell/Sorting/StationA/CustomConveyor/ConveyorBeltGraph",
)

CONVEYOR_CONFLICT_NODES: Final[tuple[str, ...]] = (
    "delay_02",
    "branch_01",
    "read_variable_01",
    "set_prim_active",
    "set_prim_active_01",
    "write_variable_04",
)

TRIGGER_CONFLICT_NODES: Final[tuple[str, ...]] = (
    "branch_01",
    "read_variable_01",
    "set_prim_active",
    "set_prim_active_01",
    "write_variable_02",
)


def _set_prim_inactive(stage: Usd.Stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsActive():
        prim.SetActive(False)
        print(f"[simulation] 비활성화: {path}")


def prepare_stage() -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("열려 있는 USD Stage가 없습니다.")

    for graph_path in STALE_GRAPH_PATHS:
        _set_prim_inactive(stage, graph_path)

    for node_name in CONVEYOR_CONFLICT_NODES:
        _set_prim_inactive(stage, f"{WORKING_CONVEYOR_GRAPH}/{node_name}")

    for node_name in TRIGGER_CONFLICT_NODES:
        _set_prim_inactive(stage, f"{TRIGGER_GRAPH_PATH}/{node_name}")

    delay_attr = stage.GetAttributeAtPath(
        f"{WORKING_CONVEYOR_GRAPH}/delay_01.inputs:duration"
    )
    if delay_attr.IsValid():
        delay_attr.Set(CONVEYOR_RESTART_DELAY_SEC)


def _capture_xform_state(prim: Usd.Prim) -> list[tuple[str, object]]:
    result: list[tuple[str, object]] = []
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        value = op.Get(Usd.TimeCode.Default())
        if value is not None:
            result.append((op.GetAttr().GetName(), value))
    return result


def _restore_xform_state(prim: Usd.Prim, state: list[tuple[str, object]]) -> None:
    for attr_name, value in state:
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(value)

    for attr_name in ("physics:velocity", "physics:angularVelocity"):
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _correct_child_local_translation(shoes_root: Usd.Prim, shoe: Usd.Prim) -> None:
    """USD에서 자식 신발에 월드 좌표가 중복 저장된 경우 로컬 좌표로 보정합니다."""
    parent_translate_attr = shoes_root.GetAttribute("xformOp:translate")
    shoe_translate_attr = shoe.GetAttribute("xformOp:translate")
    if not parent_translate_attr.IsValid() or not shoe_translate_attr.IsValid():
        return

    parent_value = parent_translate_attr.Get(Usd.TimeCode.Default())
    shoe_value = shoe_translate_attr.Get(Usd.TimeCode.Default())
    if parent_value is None or shoe_value is None:
        return

    # 현재 USD에서는 부모와 자식 모두 약 (-2, 2.8, 1.9)의 큰 좌표를 갖습니다.
    # 이 경우 자식 값은 그룹화 전 월드 좌표이므로 부모 좌표를 빼서 로컬 좌표로 만듭니다.
    if abs(float(shoe_value[0])) > 1.0 or abs(float(shoe_value[1])) > 1.0:
        local_value = Gf.Vec3d(
            float(shoe_value[0]) - float(parent_value[0]),
            float(shoe_value[1]) - float(parent_value[1]),
            float(shoe_value[2]) - float(parent_value[2]),
        )
        shoe_translate_attr.Set(local_value)
        print(f"[simulation] 신발 로컬 위치 보정: {shoe.GetPath()} -> {local_value}")


def _world_bounds(prim: Usd.Prim) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def _overlap(a: Gf.Range3d, b: Gf.Range3d) -> bool:
    amin, amax = a.GetMin(), a.GetMax()
    bmin, bmax = b.GetMin(), b.GetMax()
    return all(amin[i] <= bmax[i] and amax[i] >= bmin[i] for i in range(3))


class SimulationNode:
    def __init__(self) -> None:
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("열려 있는 USD Stage가 없습니다.")

        self._trigger = self._stage.GetPrimAtPath(TRIGGER_PATH)
        if not self._trigger.IsValid():
            raise RuntimeError(f"Trigger Prim을 찾을 수 없습니다: {TRIGGER_PATH}")

        shoes_root = self._stage.GetPrimAtPath(SHOES_ROOT_PATH)
        if not shoes_root.IsValid():
            raise RuntimeError(f"신발 루트를 찾을 수 없습니다: {SHOES_ROOT_PATH}")

        self._initial_xforms: dict[str, list[tuple[str, object]]] = {}

        for path in SHOE_PATHS:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"신발 Prim을 찾을 수 없습니다: {path}")

            _correct_child_local_translation(shoes_root, prim)
            self._initial_xforms[path] = _capture_xform_state(prim)
            prim.SetActive(False)

        self._timeline = omni.timeline.get_timeline_interface()
        self._trigger_was_occupied = False

        self._activate_random_shoe()
        self._next_activation_time = (
            self._timeline.get_current_time() + SHOE_ACTIVATION_DELAY_SEC
        )
        print("[simulation] 시작 신발 즉시 활성화")
        print("[simulation] 이후 10초마다 비활성 신발 랜덤 활성화")

    def _get_shoe(self, path: str) -> Usd.Prim:
        return self._stage.GetPrimAtPath(path)

    def _activate_random_shoe(self) -> None:
        candidates: list[str] = []
        for path in SHOE_PATHS:
            prim = self._get_shoe(path)
            if not prim.IsValid() or not prim.IsActive():
                candidates.append(path)

        if not candidates:
            return

        path = random.choice(candidates)

        # 비활성 Prim은 기존 Prim 객체와 속성 접근이 유효하지 않을 수 있습니다.
        # 먼저 활성화하고 Stage에서 Prim을 다시 얻은 뒤 원래 변환을 복원합니다.
        prim = self._get_shoe(path)
        if prim.IsValid():
            prim.SetActive(True)
        else:
            self._stage.GetPrimAtPath(path).SetActive(True)

        prim = self._get_shoe(path)
        if not prim.IsValid():
            raise RuntimeError(f"활성화 후 신발 Prim을 찾을 수 없습니다: {path}")

        _restore_xform_state(prim, self._initial_xforms[path])
        print(f"[simulation] 랜덤 신발 활성화: {path}")

    def _recycle_shoe(self, path: str, now: float) -> None:
        prim = self._get_shoe(path)
        if not prim.IsValid() or not prim.IsActive():
            return

        # 비활성화 전에 원래 위치와 속도를 복구합니다.
        _restore_xform_state(prim, self._initial_xforms[path])
        prim.SetActive(False)
        print(f"[simulation] Trigger 도착 → 복귀 및 비활성화: {path}")

    def update(self) -> None:
        if not self._timeline.is_playing():
            return

        now = self._timeline.get_current_time()
        trigger_bounds = _world_bounds(self._trigger)
        occupied = False

        for path in SHOE_PATHS:
            prim = self._get_shoe(path)
            if not prim.IsValid() or not prim.IsActive():
                continue
            if _overlap(_world_bounds(prim), trigger_bounds):
                occupied = True
                if not self._trigger_was_occupied:
                    self._recycle_shoe(path, now)
                break

        self._trigger_was_occupied = occupied

        if now >= self._next_activation_time:
            self._activate_random_shoe()
            self._next_activation_time = now + SHOE_ACTIVATION_DELAY_SEC

    def shutdown(self) -> None:
        pass


def create_simulation_node() -> SimulationNode:
    return SimulationNode()