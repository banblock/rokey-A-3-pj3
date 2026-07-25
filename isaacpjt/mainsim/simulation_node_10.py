"""Isaac Sim 신발 순환, Shoebox 순차 활성화 및 컨베이어 개별 제어.

주요 기능
1. 등록된 신발을 시작 시 비활성화합니다.
2. 시작 시 신발 하나를 랜덤하게 활성화합니다.
3. 이후 일정 시간마다 비활성 신발 하나를 랜덤하게 활성화합니다.
4. /World/ShoeTrigger에 도착한 신발을 비활성화합니다.
5. 신발 한 개가 ShoeTrigger에 도착할 때마다 Shoebox를 순차 활성화합니다.
"""

from __future__ import annotations

import random
import re
from typing import Final

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

import omni.timeline
import omni.usd
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings import _physx
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom



# =====================================================================
# 기본 경로
# =====================================================================

WORKING_CONVEYOR_GRAPH: Final[str] = (
    "/World/ReturnCell/Conveyors/Input/"
    "ConveyorTrack/ConveyorBeltGraph"
)

TRIGGER_GRAPH_PATH: Final[str] = "/World/ShoeTrigger/ActionGraph"
TRIGGER_PATH: Final[str] = "/World/ShoeTrigger"
TRIGGER4_GRAPH_PATH: Final[str] = "/World/ShoeTrigger_04/ActionGraph"
TRIGGER4_PATH: Final[str] = "/World/ShoeTrigger_04"
STOP_TRIGGER_PATH: Final[str] = "/World/StopTrigger"

SHOES_ROOT_PATH: Final[str] = "/World/shoes"
SHOEBOX_ROOT_PATH: Final[str] = "/World/shoeboxs"


# =====================================================================
# ROS2 통신
# =====================================================================

EMERGENCY_STOP_TOPIC: Final[str] = "/control/emergency_stop"
SHOE_STOP_TOPIC: Final[str] = "/shoe_stop"
START_SCENE_TOPIC: Final[str] = "/control/start_scene"
AMR_READY_TOPIC: Final[str] = "/fms/amr_ready"
AMR_CARRYING_COMPLETE_TOPIC: Final[str] = "/fms/amr_carrying_complete"

RELIABLE_EVENT_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class RosInterface(Node):
    def __init__(self) -> None:
        super().__init__("isaac_simulation_node")

        self.emergency_stop = False
        self.start_scene = False
        self.amr_ready = ""
        self.amr_carrying_complete = ""

        self.create_subscription(
            Bool, EMERGENCY_STOP_TOPIC, self._on_emergency_stop, 10
        )
        self.create_subscription(
            Bool, START_SCENE_TOPIC, self._on_start_scene, 10
        )
        self.create_subscription(
            String, AMR_READY_TOPIC, self._on_amr_ready, RELIABLE_EVENT_QOS
        )
        self.create_subscription(
            String,
            AMR_CARRYING_COMPLETE_TOPIC,
            self._on_amr_carrying_complete,
            RELIABLE_EVENT_QOS,
        )

        self._shoe_stop_publisher = self.create_publisher(
            Bool, SHOE_STOP_TOPIC, 10
        )

    def _on_emergency_stop(self, msg: Bool) -> None:
        self.emergency_stop = msg.data

    def _on_start_scene(self, msg: Bool) -> None:
        self.start_scene = msg.data

    def _on_amr_ready(self, msg: String) -> None:
        self.amr_ready = msg.data

    def _on_amr_carrying_complete(self, msg: String) -> None:
        self.amr_carrying_complete = msg.data

    def publish_shoe_stop(self) -> None:
        msg = Bool()
        msg.data = True
        self._shoe_stop_publisher.publish(msg)


# =====================================================================
# 시간 설정
# =====================================================================

CONVEYOR_RESTART_DELAY_SEC: Final[float] = 2.0
SHOE_ACTIVATION_DELAY_SEC: Final[float] = 8.0
STOP_TRIGGER_DELAY_SEC: Final[float] = 0.8
STOP_TRIGGER_RESTART_DELAY_SEC: Final[float] = 5.0


# =====================================================================
# 신발 경로
# =====================================================================

SHOE_PATHS: Final[tuple[str, ...]] = (
    "/World/shoes/sneakers",
    "/World/shoes/sneaker_0001_red_240_ok_01",
    "/World/shoes/sneaker_0001_red_240_ok_02",
)


# =====================================================================
# 기존 Action Graph 충돌 요소
# =====================================================================

CONVEYOR_CONFLICT_NODES: Final[tuple[str, ...]] = (
    # 실제 컨베이어 물리에 필요한 OnTick, read_speed, ConveyorNode는 유지합니다.
    # 기존 그래프에서 Python 대체 로직과 충돌하는 제어 노드만 비활성화합니다.
    "on_trigger",
    "constant_float",
    "write_variable",
    "write_variable_01",
    "ros2_context",
    "ros2_publisher",
    "constant_bool",
    "delay",
    "delay_01",
    "constant_float_01",
    "branch",
    "write_variable_02",
    "constant_bool_01",
    "read_variable",
    "write_variable_03",
    "constant_bool_02",
    "delay_02",
    "constant_double",
    "set_prim_active",
    "branch_01",
    "read_variable_01",
    "write_variable_04",
    "constant_bool_03",
    "set_prim_active_01",
)

TRIGGER_CONFLICT_NODES: Final[tuple[str, ...]] = (
    # Trigger 실행과 ROS2 Publisher는 유지합니다.
    # 특정 이름의 신발을 직접 끄는 기존 노드만 비활성화합니다.
    "set_prim_active",
    "set_prim_active_01",
)

TRIGGER4_CONFLICT_NODES: Final[tuple[str, ...]] = (
    # ShoeTrigger_04에서 지정된 신발을 끄는 기존 노드만 비활성화합니다.
    "set_prim_active",
    "set_prim_active_01",
)


# =====================================================================
# Stage 준비 함수
# =====================================================================

def _set_prim_inactive(
    stage: Usd.Stage,
    path: str,
) -> None:
    """Prim이 존재하고 활성 상태라면 비활성화합니다."""

    prim = stage.GetPrimAtPath(path)

    if prim.IsValid() and prim.IsActive():
        prim.SetActive(False)
        print(f"[simulation] 비활성화: {path}")


def _disable_missing_dome_light(
    stage: Usd.Stage,
) -> None:
    """누락된 텍스처를 참조하는 조명을 비활성화합니다."""

    target_name = "color_0C0C0C.exr"

    for prim in stage.TraverseAll():
        if not prim.IsValid():
            continue

        uses_missing_texture = False

        for attr in prim.GetAttributes():
            try:
                value = attr.Get(Usd.TimeCode.Default())
            except Exception:
                continue

            if value is not None and target_name in str(value):
                uses_missing_texture = True
                break

        if not uses_missing_texture:
            continue

        prim_path = str(prim.GetPath())

        try:
            prim.SetActive(False)

            print(
                "[simulation] 누락 텍스처 조명 비활성화: "
                f"{prim_path}"
            )

        except Exception:
            override_prim = stage.OverridePrim(prim.GetPath())
            override_prim.SetActive(False)

            print(
                "[simulation] 누락 텍스처 조명 override 비활성화: "
                f"{prim_path}"
            )


def prepare_stage() -> None:
    """World.reset() 전에 기존 충돌 그래프를 정리합니다."""

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("열려 있는 USD Stage가 없습니다.")

    _disable_missing_dome_light(stage)

    # 기존 ConveyorBeltGraph 전체는 유지합니다.
    # 실제 물리 노드(OnTick, read_speed, ConveyorNode)는 그대로 사용하고,
    # Python 대체 로직과 충돌하는 제어 노드만 비활성화합니다.
    for node_name in CONVEYOR_CONFLICT_NODES:
        _set_prim_inactive(
            stage,
            f"{WORKING_CONVEYOR_GRAPH}/{node_name}",
        )

    for node_name in TRIGGER_CONFLICT_NODES:
        _set_prim_inactive(
            stage,
            f"{TRIGGER_GRAPH_PATH}/{node_name}",
        )

    # ShoeTrigger_04의 기존 지정 신발 비활성화 노드는 모두 끕니다.
    trigger4_graph = stage.GetPrimAtPath(
        TRIGGER4_GRAPH_PATH
    )

    if trigger4_graph.IsValid():
        for child in trigger4_graph.GetChildren():
            if child.GetName().startswith(
                "set_prim_active"
            ):
                _set_prim_inactive(
                    stage,
                    str(child.GetPath()),
                )



# =====================================================================
# Transform 처리
# =====================================================================

def _capture_xform_state(
    prim: Usd.Prim,
) -> list[tuple[str, object]]:
    """Prim의 초기 Transform 상태를 저장합니다."""

    result: list[tuple[str, object]] = []

    xformable = UsdGeom.Xformable(prim)

    for op in xformable.GetOrderedXformOps():
        value = op.Get(Usd.TimeCode.Default())

        if value is None:
            continue

        result.append(
            (
                op.GetAttr().GetName(),
                value,
            )
        )

    return result


def _restore_xform_state(
    prim: Usd.Prim,
    state: list[tuple[str, object]],
) -> None:
    """저장해 둔 Transform과 물리 속도를 복구합니다."""

    for attr_name, value in state:
        attr = prim.GetAttribute(attr_name)

        if attr.IsValid():
            attr.Set(value)

    for attr_name in (
        "physics:velocity",
        "physics:angularVelocity",
    ):
        attr = prim.GetAttribute(attr_name)

        if attr.IsValid():
            attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _correct_child_local_translation(
    shoes_root: Usd.Prim,
    shoe: Usd.Prim,
) -> None:
    """자식 신발에 월드 좌표가 중복 저장된 경우 로컬 좌표로 보정합니다."""

    parent_translate_attr = shoes_root.GetAttribute(
        "xformOp:translate"
    )

    shoe_translate_attr = shoe.GetAttribute(
        "xformOp:translate"
    )

    if (
        not parent_translate_attr.IsValid()
        or not shoe_translate_attr.IsValid()
    ):
        return

    parent_value = parent_translate_attr.Get(
        Usd.TimeCode.Default()
    )

    shoe_value = shoe_translate_attr.Get(
        Usd.TimeCode.Default()
    )

    if parent_value is None or shoe_value is None:
        return

    if (
        abs(float(shoe_value[0])) > 1.0
        or abs(float(shoe_value[1])) > 1.0
    ):
        local_value = Gf.Vec3d(
            float(shoe_value[0]) - float(parent_value[0]),
            float(shoe_value[1]) - float(parent_value[1]),
            float(shoe_value[2]) - float(parent_value[2]),
        )

        shoe_translate_attr.Set(local_value)

        print(
            "[simulation] 신발 로컬 위치 보정: "
            f"{shoe.GetPath()} -> {local_value}"
        )


# =====================================================================
# Shoebox 검색 및 정렬
# =====================================================================

def _natural_sort_key(
    value: str,
) -> list[object]:
    """shoebox_2가 shoebox_10보다 먼저 오도록 정렬합니다."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def _find_inactive_child_paths_from_layers(
    stage: Usd.Stage,
    root_path: str,
    name_keyword: str,
) -> list[str]:
    """Layer를 검사하여 active=false인 자식 Prim 경로까지 찾습니다."""

    discovered_paths: set[str] = set()

    root_sdf_path = Sdf.Path(root_path)

    for layer in stage.GetLayerStack():
        root_spec = layer.GetPrimAtPath(root_sdf_path)

        if root_spec is None:
            continue

        try:
            child_specs = root_spec.nameChildren
        except Exception:
            continue

        for child_spec in child_specs:
            child_name = child_spec.name

            if name_keyword.lower() not in child_name.lower():
                continue

            discovered_paths.add(
                f"{root_path}/{child_name}"
            )

    return sorted(
        discovered_paths,
        key=lambda path: _natural_sort_key(
            path.rsplit("/", 1)[-1]
        ),
    )


# =====================================================================
# 컨베이어 속성 검색
# =====================================================================

def _find_velocity_attribute(
    stage: Usd.Stage,
    conveyor_root_path: str,
) -> Usd.Attribute:
    """컨베이어 Prim 아래에서 속도 제어용 Velocity 속성을 찾습니다."""

    conveyor_root = stage.GetPrimAtPath(
        conveyor_root_path
    )

    if not conveyor_root.IsValid():
        raise RuntimeError(
            "컨베이어 Prim을 찾을 수 없습니다: "
            f"{conveyor_root_path}"
        )

    exact_attribute_names = (
        "graph:variable:Velocity",
        "variables:Velocity",
        "graph:variable:velocity",
        "variables:velocity",
    )

    graph_velocity_candidates: list[Usd.Attribute] = []
    surface_velocity_candidates: list[Usd.Attribute] = []
    other_velocity_candidates: list[Usd.Attribute] = []

    for prim in Usd.PrimRange(conveyor_root):
        if not prim.IsValid():
            continue

        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            lower_name = attr_name.lower()

            if attr_name in exact_attribute_names:
                print(
                    "[simulation] 컨베이어 속도 속성 확인: "
                    f"{attr.GetPath()}"
                )

                return attr

            if (
                "velocity" in lower_name
                and "variable" in lower_name
            ):
                graph_velocity_candidates.append(attr)

            elif "surfacevelocity" in lower_name:
                surface_velocity_candidates.append(attr)

            elif "velocity" in lower_name:
                other_velocity_candidates.append(attr)

    candidates = (
        graph_velocity_candidates
        + surface_velocity_candidates
        + other_velocity_candidates
    )

    if candidates:
        selected_attr = candidates[0]

        print(
            "[simulation] 컨베이어 속도 후보 사용: "
            f"{selected_attr.GetPath()}"
        )

        return selected_attr

    raise RuntimeError(
        "컨베이어 아래에서 속도 속성을 찾을 수 없습니다: "
        f"{conveyor_root_path}"
    )


# =====================================================================
# SimulationNode
# =====================================================================

class SimulationNode:
    def __init__(self) -> None:
        self._stage = omni.usd.get_context().get_stage()

        if self._stage is None:
            raise RuntimeError(
                "열려 있는 USD Stage가 없습니다."
            )

        if not rclpy.ok():
            rclpy.init()

        self._ros_node = RosInterface()

        # -------------------------------------------------------------
        # ShoeTrigger
        # -------------------------------------------------------------

        self._trigger = self._stage.GetPrimAtPath(
            TRIGGER_PATH
        )

        if not self._trigger.IsValid():
            raise RuntimeError(
                "Trigger Prim을 찾을 수 없습니다: "
                f"{TRIGGER_PATH}"
            )

        self._trigger4 = self._stage.GetPrimAtPath(
            TRIGGER4_PATH
        )

        if not self._trigger4.IsValid():
            raise RuntimeError(
                "Trigger Prim을 찾을 수 없습니다: "
                f"{TRIGGER4_PATH}"
            )

        self._stop_trigger = self._stage.GetPrimAtPath(
            STOP_TRIGGER_PATH
        )

        if not self._stop_trigger.IsValid():
            raise RuntimeError(
                "Trigger Prim을 찾을 수 없습니다: "
                f"{STOP_TRIGGER_PATH}"
            )

        print(
            "[simulation] ShoeTrigger 등록: "
            f"{TRIGGER_PATH}"
        )

        print(
            "[simulation] ShoeTrigger_04 등록: "
            f"{TRIGGER4_PATH}"
        )

        # -------------------------------------------------------------
        # 신발 초기화
        # -------------------------------------------------------------

        shoes_root = self._stage.GetPrimAtPath(
            SHOES_ROOT_PATH
        )

        if not shoes_root.IsValid():
            raise RuntimeError(
                "신발 루트를 찾을 수 없습니다: "
                f"{SHOES_ROOT_PATH}"
            )

        self._initial_xforms: dict[
            str,
            list[tuple[str, object]],
        ] = {}

        for shoe_path in SHOE_PATHS:
            shoe_prim = self._stage.GetPrimAtPath(
                shoe_path
            )

            if not shoe_prim.IsValid():
                raise RuntimeError(
                    "신발 Prim을 찾을 수 없습니다: "
                    f"{shoe_path}"
                )

            _correct_child_local_translation(
                shoes_root,
                shoe_prim,
            )

            self._initial_xforms[shoe_path] = (
                _capture_xform_state(shoe_prim)
            )

            shoe_prim.SetActive(False)

            print(
                "[simulation] 시작 신발 비활성화: "
                f"{shoe_path}"
            )

        # -------------------------------------------------------------
        # Shoebox 초기화
        # -------------------------------------------------------------

        self._shoebox_paths: list[str] = []
        self._next_shoebox_index = 0

        self._initialize_shoeboxes()

        # -------------------------------------------------------------
        # 비전 컨베이어 및 일반 컨베이어 3개 등록
        # -------------------------------------------------------------

        self._working_conveyor_velocity_attr = (
            _find_velocity_attribute(
                self._stage,
                WORKING_CONVEYOR_GRAPH,
            )
        )

        # -------------------------------------------------------------
        # Timeline 및 상태 변수
        # -------------------------------------------------------------

        self._timeline = (
            omni.timeline.get_timeline_interface()
        )

        self._emergency_stopped = False
        self._working_speed_before_emergency = 1.0
        self._vision_stop_started_at: float | None = None
        self._vision_delay_remaining = 0.0
        self._vision_resume_deadline: float | None = None
        self._stop_trigger_sequences: list[dict[str, object]] = []

        self._scene_started = False
        self._previous_start_scene_signal = False

        initial_working_speed = self._get_attribute_float(
            self._working_conveyor_velocity_attr,
            default=1.0,
        )

        if initial_working_speed < 0.5:
            self._vision_stop_started_at = (
                self._timeline.get_current_time()
            )

        print(
            "[simulation] 비상정지 토픽 대기: "
            f"{EMERGENCY_STOP_TOPIC}"
        )

        # PhysX Trigger 진입 이벤트를 update()에서 처리하기 위한 대기열입니다.
        self._pending_trigger_shoes: list[str] = []
        self._pending_trigger4_shoes: list[str] = []
        self._pending_stop_trigger_shoes: list[str] = []

        self._physx_simulation = get_physx_simulation_interface()
        self._trigger_subscription_id = (
            self._physx_simulation.subscribe_physics_trigger_report_events(
                self._on_trigger_event
            )
        )

        print(
            "[simulation] PhysX ShoeTrigger 이벤트 구독: "
            f"{TRIGGER_PATH}"
        )

        self._next_activation_time: float | None = None

        print(
            "[simulation] 시작 신호 대기: "
            f"{START_SCENE_TOPIC} (std_msgs/msg/Bool)"
        )

    # =================================================================
    # Shoebox 초기화
    # =================================================================

    def _initialize_shoeboxes(self) -> None:
        """shoeboxs 아래 상자를 찾아 이름순으로 등록합니다."""

        shoebox_root = self._stage.GetPrimAtPath(
            SHOEBOX_ROOT_PATH
        )

        if not shoebox_root.IsValid():
            raise RuntimeError(
                "Shoebox 루트를 찾을 수 없습니다: "
                f"{SHOEBOX_ROOT_PATH}"
            )

        discovered_paths: set[str] = set()

        # 현재 활성 상태로 조회되는 자식 Prim도 확인합니다.
        for child in shoebox_root.GetAllChildren():
            if not child.IsValid():
                continue

            child_name = child.GetName()

            if "shoebox" not in child_name.lower():
                continue

            discovered_paths.add(
                str(child.GetPath())
            )

        # active=false 상태라 일반 Traversal에서 빠지는 Prim도 Layer에서 찾습니다.
        layer_paths = _find_inactive_child_paths_from_layers(
            self._stage,
            SHOEBOX_ROOT_PATH,
            "shoebox",
        )

        discovered_paths.update(layer_paths)

        self._shoebox_paths = sorted(
            discovered_paths,
            key=lambda path: _natural_sort_key(
                path.rsplit("/", 1)[-1]
            ),
        )

        self._next_shoebox_index = 0

        if not self._shoebox_paths:
            raise RuntimeError(
                "Shoebox를 한 개도 찾지 못했습니다. "
                f"{SHOEBOX_ROOT_PATH} 아래에 "
                "shoebox, shoebox_01 등의 Prim이 있는지 확인하세요."
            )

        # 시작 시 모든 Shoebox를 비활성화합니다.
        for shoebox_path in self._shoebox_paths:
            shoebox_prim = self._stage.GetPrimAtPath(
                shoebox_path
            )

            if shoebox_prim.IsValid():
                shoebox_prim.SetActive(False)
            else:
                override_prim = self._stage.OverridePrim(
                    shoebox_path
                )

                override_prim.SetActive(False)

        print(
            "[simulation] Shoebox 등록 개수: "
            f"{len(self._shoebox_paths)}"
        )

        for index, shoebox_path in enumerate(
            self._shoebox_paths,
            start=1,
        ):
            print(
                f"[simulation] Shoebox {index}: "
                f"{shoebox_path}"
            )

    # =================================================================
    # PhysX Trigger 처리
    # =================================================================

    def _resolve_shoe_path(
        self,
        other_path: str,
    ) -> str | None:
        """Trigger에 진입한 Collider/Body 경로를 등록된 신발 루트로 변환합니다."""

        for shoe_path in SHOE_PATHS:
            if (
                other_path == shoe_path
                or other_path.startswith(shoe_path + "/")
            ):
                return shoe_path

        return None

    def _on_trigger_event(
        self,
        event,
    ) -> None:
        """실제 PhysX Trigger 진입 이벤트를 수신합니다."""

        if (
            event.event_type
            != _physx.TriggerEventType.TRIGGER_ON_ENTER
        ):
            return

        trigger_collider_path = str(
            PhysicsSchemaTools.intToSdfPath(
                event.trigger_collider_prim_id
            )
        )

        trigger_body_path = str(
            PhysicsSchemaTools.intToSdfPath(
                event.trigger_body_prim_id
            )
        )

        other_collider_path = str(
            PhysicsSchemaTools.intToSdfPath(
                event.other_collider_prim_id
            )
        )

        other_body_path = str(
            PhysicsSchemaTools.intToSdfPath(
                event.other_body_prim_id
            )
        )

        shoe_path = (
            self._resolve_shoe_path(
                other_body_path
            )
            or self._resolve_shoe_path(
                other_collider_path
            )
        )

        if shoe_path is None:
            return

        is_shoe_trigger = (
            trigger_collider_path == TRIGGER_PATH
            or trigger_collider_path.startswith(
                TRIGGER_PATH + "/"
            )
            or trigger_body_path == TRIGGER_PATH
            or trigger_body_path.startswith(
                TRIGGER_PATH + "/"
            )
        )

        is_shoe_trigger4 = (
            trigger_collider_path == TRIGGER4_PATH
            or trigger_collider_path.startswith(
                TRIGGER4_PATH + "/"
            )
            or trigger_body_path == TRIGGER4_PATH
            or trigger_body_path.startswith(
                TRIGGER4_PATH + "/"
            )
        )

        is_stop_trigger = (
            trigger_collider_path == STOP_TRIGGER_PATH
            or trigger_collider_path.startswith(
                STOP_TRIGGER_PATH + "/"
            )
            or trigger_body_path == STOP_TRIGGER_PATH
            or trigger_body_path.startswith(
                STOP_TRIGGER_PATH + "/"
            )
        )

        if is_shoe_trigger:
            if shoe_path not in self._pending_trigger_shoes:
                self._pending_trigger_shoes.append(
                    shoe_path
                )

                print(
                    "[simulation] PhysX ShoeTrigger 진입: "
                    f"{shoe_path}"
                )

        elif is_shoe_trigger4:
            if shoe_path not in self._pending_trigger4_shoes:
                self._pending_trigger4_shoes.append(
                    shoe_path
                )

                print(
                    "[simulation] PhysX ShoeTrigger_04 진입: "
                    f"{shoe_path}"
                )

        elif is_stop_trigger:
            if shoe_path not in self._pending_stop_trigger_shoes:
                self._pending_stop_trigger_shoes.append(
                    shoe_path
                )

                print(
                    "[simulation] PhysX StopTrigger 진입: "
                    f"{shoe_path}"
                )

    # =================================================================
    # 신발 처리
    # =================================================================

    def _get_shoe(
        self,
        shoe_path: str,
    ) -> Usd.Prim:
        return self._stage.GetPrimAtPath(
            shoe_path
        )

    def _activate_random_shoe(self) -> None:
        """비활성 상태인 신발 중 하나를 랜덤하게 활성화합니다."""

        candidates: list[str] = []

        for shoe_path in SHOE_PATHS:
            shoe_prim = self._get_shoe(shoe_path)

            if (
                not shoe_prim.IsValid()
                or not shoe_prim.IsActive()
            ):
                candidates.append(shoe_path)

        if not candidates:
            print(
                "[simulation] 현재 활성화할 "
                "비활성 신발이 없습니다."
            )
            return

        selected_path = random.choice(candidates)
        shoe_prim = self._get_shoe(selected_path)

        if shoe_prim.IsValid():
            shoe_prim.SetActive(True)
        else:
            self._stage.OverridePrim(
                selected_path
            ).SetActive(True)

        shoe_prim = self._get_shoe(selected_path)

        _restore_xform_state(
            shoe_prim,
            self._initial_xforms[selected_path],
        )

        imageable = UsdGeom.Imageable(shoe_prim)

        if imageable:
            imageable.MakeVisible()

        print(
            "[simulation] ROS 명령으로 신발 활성화: "
            f"{selected_path}"
        )

    def _recycle_shoe(
        self,
        shoe_path: str,
    ) -> None:
        """ShoeTrigger에 들어온 신발을 초기 위치로 복구한 뒤 비활성화합니다."""

        shoe_prim = self._get_shoe(
            shoe_path
        )

        if (
            not shoe_prim.IsValid()
            or not shoe_prim.IsActive()
        ):
            return

        imageable = UsdGeom.Imageable(
            shoe_prim
        )

        if imageable:
            imageable.MakeInvisible()

        _restore_xform_state(
            shoe_prim,
            self._initial_xforms[shoe_path],
        )

        shoe_prim.SetActive(False)

        print(
            "[simulation] ShoeTrigger 접촉 신발 비활성화: "
            f"{shoe_path}"
        )

    # =================================================================
    # Shoebox 처리
    # =================================================================

    def _activate_next_shoebox(self) -> None:
        """다음 순서의 Shoebox 하나를 활성화합니다."""

        if (
            self._next_shoebox_index
            >= len(self._shoebox_paths)
        ):
            print(
                "[simulation] 활성화할 남은 Shoebox가 없습니다."
            )
            return

        shoebox_path = self._shoebox_paths[
            self._next_shoebox_index
        ]

        # 비활성 Prim에 현재 Stage의 active=true override를 작성합니다.
        shoebox_override = self._stage.OverridePrim(
            shoebox_path
        )

        shoebox_override.SetActive(True)

        # Compose 결과를 다시 조회합니다.
        shoebox_prim = self._stage.GetPrimAtPath(
            shoebox_path
        )

        if not shoebox_prim.IsValid():
            raise RuntimeError(
                "Shoebox 활성화 후 Prim을 찾을 수 없습니다: "
                f"{shoebox_path}"
            )

        imageable = UsdGeom.Imageable(
            shoebox_prim
        )

        if imageable:
            imageable.MakeVisible()

        # 이전 시뮬레이션의 이동 속도가 남아 있다면 초기화합니다.
        for attr_name in (
            "physics:velocity",
            "physics:angularVelocity",
        ):
            attr = shoebox_prim.GetAttribute(
                attr_name
            )

            if attr.IsValid():
                attr.Set(
                    Gf.Vec3f(0.0, 0.0, 0.0)
                )

        self._next_shoebox_index += 1

        print(
            "[simulation] Shoebox 순차 활성화: "
            f"{shoebox_path} "
            f"({self._next_shoebox_index}/"
            f"{len(self._shoebox_paths)})"
        )

    # =================================================================
    # 비상정지 처리
    # =================================================================

    @staticmethod
    def _get_attribute_float(
        attr: Usd.Attribute,
        default: float = 0.0,
    ) -> float:
        value = attr.Get()

        if value is None:
            return float(default)

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _stop_all_conveyors(self) -> None:
        self._working_conveyor_velocity_attr.Set(0.0)

    def _enter_emergency_stop(self, now: float) -> None:
        """비전 컨베이어 상태를 기록하고 전체 컨베이어를 정지합니다."""

        self._working_speed_before_emergency = (
            self._get_attribute_float(
                self._working_conveyor_velocity_attr,
                default=1.0,
            )
        )

        self._vision_delay_remaining = 0.0

        if (
            self._working_speed_before_emergency < 0.5
            and self._vision_stop_started_at is not None
        ):
            elapsed = max(
                0.0,
                now - self._vision_stop_started_at,
            )
            self._vision_delay_remaining = max(
                0.0,
                CONVEYOR_RESTART_DELAY_SEC - elapsed,
            )

        self._vision_resume_deadline = None
        self._emergency_stopped = True
        self._stop_all_conveyors()

        print(
            "[EMERGENCY] 전체 컨베이어 정지, "
            "비전 대기 잔여시간="
            f"{self._vision_delay_remaining:.3f}초"
        )

    def _leave_emergency_stop(self, now: float) -> None:
        """일반 컨베이어는 재개하고 비전 컨베이어 지연만 복원합니다."""

        self._emergency_stopped = False

        if (
            self._working_speed_before_emergency < 0.5
            and self._vision_delay_remaining > 0.0
        ):
            self._working_conveyor_velocity_attr.Set(0.0)
            self._vision_stop_started_at = now
            self._vision_resume_deadline = (
                now + self._vision_delay_remaining
            )

            print(
                "비전 컨베이어는 "
                f"{self._vision_delay_remaining:.3f}초 후 재개"
            )
        else:
            self._working_conveyor_velocity_attr.Set(1.0)
            self._vision_stop_started_at = None
            self._vision_resume_deadline = None

            print("[EMERGENCY] 해제: 전체 컨베이어 재개")

        self._vision_delay_remaining = 0.0

    def _update_vision_conveyor_state(self, now: float) -> None:
        """비전 컨베이어의 0→1 변화와 복원 중인 잔여 지연을 관리합니다."""

        if self._vision_resume_deadline is not None:
            if now < self._vision_resume_deadline:
                self._working_conveyor_velocity_attr.Set(0.0)
                return

            self._working_conveyor_velocity_attr.Set(1.0)
            self._vision_resume_deadline = None
            self._vision_stop_started_at = None
            print("[vision] 비상정지 전 남은 지연 완료: 컨베이어 재개")
            return

        current_speed = self._get_attribute_float(
            self._working_conveyor_velocity_attr,
            default=1.0,
        )

        if current_speed < 0.5:
            if self._vision_stop_started_at is None:
                self._vision_stop_started_at = now
        else:
            self._vision_stop_started_at = None

    # =================================================================
    # 입력 컨베이어 StopTrigger 시퀀스
    # =================================================================

    def _publish_shoe_stop(self) -> None:
        self._ros_node.publish_shoe_stop()
        print(f"[simulation] {SHOE_STOP_TOPIC}: data=True 발행")

    def _start_stop_trigger_sequence(self) -> None:
        self._stop_trigger_sequences.append(
            {
                "started_at": self._timeline.get_current_time(),
                "stopped": False,
                "restarted": False,
            }
        )

    def _update_stop_trigger_sequences(
        self,
        now: float,
    ) -> None:
        remaining: list[dict[str, object]] = []

        for sequence in self._stop_trigger_sequences:
            started_at = float(sequence["started_at"])
            elapsed = max(0.0, now - started_at)

            if (
                not bool(sequence["stopped"])
                and elapsed >= STOP_TRIGGER_DELAY_SEC
            ):
                self._working_conveyor_velocity_attr.Set(0.0)
                self._publish_shoe_stop()
                sequence["stopped"] = True

            if (
                not bool(sequence["restarted"])
                and elapsed
                >= (
                    STOP_TRIGGER_DELAY_SEC
                    + STOP_TRIGGER_RESTART_DELAY_SEC
                )
            ):
                self._working_conveyor_velocity_attr.Set(1.0)
                sequence["restarted"] = True

            if not (
                bool(sequence["stopped"])
                and bool(sequence["restarted"])
            ):
                remaining.append(sequence)

        self._stop_trigger_sequences = remaining

    # =================================================================
    # 매 프레임 실행
    # =================================================================

    def update(self) -> None:
        """시뮬레이션 실행 중 매 프레임 호출합니다."""

        if not self._timeline.is_playing():
            return

        rclpy.spin_once(self._ros_node, timeout_sec=0.0)

        now = self._timeline.get_current_time()

        start_scene_signal = self._ros_node.start_scene

        if start_scene_signal and not self._previous_start_scene_signal:
            self._scene_started = True
            self._activate_random_shoe()
            self._next_activation_time = (
                now + SHOE_ACTIVATION_DELAY_SEC
            )
            print("[simulation] 시작 신호로 첫 신발 즉시 활성화")

        elif not start_scene_signal and self._previous_start_scene_signal:
            self._scene_started = False
            self._next_activation_time = None
            print(f"[simulation] {START_SCENE_TOPIC}: False 수신")

        self._previous_start_scene_signal = start_scene_signal

        while self._pending_stop_trigger_shoes:
            self._pending_stop_trigger_shoes.pop(0)
            self._start_stop_trigger_sequence()

        self._update_stop_trigger_sequences(now)

        emergency_signal = self._ros_node.emergency_stop

        if emergency_signal and not self._emergency_stopped:
            self._enter_emergency_stop(now)

        elif not emergency_signal and self._emergency_stopped:
            self._leave_emergency_stop(now)

        if self._emergency_stopped:
            # 다른 Action Graph가 속도를 다시 쓰더라도 매 프레임 0으로 강제합니다.
            self._stop_all_conveyors()
            return

        self._update_vision_conveyor_state(now)

        while self._pending_trigger_shoes:
            shoe_path = self._pending_trigger_shoes.pop(0)

            # Trigger 진입 이벤트가 들어왔으면 Shoebox는 반드시 1개 활성화합니다.
            # 기존 Action Graph가 같은 프레임에 신발을 먼저 비활성화해도
            # Shoebox 활성화가 건너뛰어지지 않도록 먼저 처리합니다.
            self._activate_next_shoebox()

            shoe_prim = self._get_shoe(
                shoe_path
            )

            # 기존 Action Graph가 이미 신발을 비활성화했다면
            # Python에서는 중복 디스폰하지 않습니다.
            if (
                shoe_prim.IsValid()
                and shoe_prim.IsActive()
            ):
                self._recycle_shoe(
                    shoe_path
                )

        while self._pending_trigger4_shoes:
            shoe_path = self._pending_trigger4_shoes.pop(0)

            shoe_prim = self._get_shoe(
                shoe_path
            )

            if (
                shoe_prim.IsValid()
                and shoe_prim.IsActive()
            ):
                self._recycle_shoe(
                    shoe_path
                )

        if (
            self._scene_started
            and self._next_activation_time is not None
            and now >= self._next_activation_time
        ):
            self._activate_random_shoe()

            self._next_activation_time = (
                now + SHOE_ACTIVATION_DELAY_SEC
            )

    def shutdown(self) -> None:
        """PhysX Trigger 구독과 대기 중인 이벤트를 정리합니다."""

        self._pending_trigger_shoes.clear()
        self._pending_trigger4_shoes.clear()
        self._pending_stop_trigger_shoes.clear()
        self._stop_trigger_sequences.clear()

        if self._trigger_subscription_id is not None:
            self._physx_simulation.unsubscribe_physics_trigger_report_events(
                self._trigger_subscription_id
            )

            self._trigger_subscription_id = None

        self._ros_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("[simulation] SimulationNode 종료")


def create_simulation_node() -> SimulationNode:
    """메인 실행 코드에서 호출할 생성 함수입니다."""

    return SimulationNode()