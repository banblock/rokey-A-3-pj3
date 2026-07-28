"""Isaac Sim 신발 순환 및 컨베이어 개별 제어.

주요 기능
1. 등록된 신발을 시작 시 비활성화합니다.
2. 시작 시 시작 신호를 받고 신발 하나를 랜덤하게 활성화합니다.
3. 이후 일정 시간마다 비활성 신발 하나를 랜덤하게 활성화합니다.
4. /World/ShoeTrigger에 도착한 신발을 비활성화합니다.
"""

from __future__ import annotations

import json
import random
import re
from typing import Final

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32, String

import omni.timeline
import omni.usd
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings import _physx
from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdShade

from shoe_damage import (
    cache_mesh_topology,
    create_condition_materials,
    make_ellipse_patch,
    pick_tear_placement,
    update_ellipse_patch,
)



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


# =====================================================================
# ROS2 통신
# =====================================================================

EMERGENCY_STOP_TOPIC: Final[str] = "/control/emergency_stop"
SHOE_STOP_TOPIC: Final[str] = "/shoe_stop"
START_SCENE_TOPIC: Final[str] = "/control/start_scene"
AMR_READY_TOPIC: Final[str] = "/fms/amr_ready"
AMR_CARRYING_COMPLETE_TOPIC: Final[str] = "/fms/amr_carrying_complete"
SIM_PICK_DONE_TOPIC: Final[str] = "/sim/pick_done"
SIM_PLACE_DONE_TOPIC: Final[str] = "/sim/place_done"

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
        # amr_ready/amr_carrying_complete는 "최신값 하나만 저장"이 아니라 큐로
        # 쌓는다 - 최신값 덮어쓰기 방식이면 같은 내용(예: count가 우연히 같은
        # 두 번째 amr_ready)의 연속 이벤트를 구분 못 하고 놓칠 수 있다.
        self.amr_ready_queue: list[str] = []
        self.amr_carrying_complete_queue: list[str] = []

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
        self._pick_done_publisher = self.create_publisher(
            Int32, SIM_PICK_DONE_TOPIC, RELIABLE_EVENT_QOS
        )
        self._place_done_publisher = self.create_publisher(
            Int32, SIM_PLACE_DONE_TOPIC, RELIABLE_EVENT_QOS
        )

    def _on_emergency_stop(self, msg: Bool) -> None:
        self.emergency_stop = msg.data

    def _on_start_scene(self, msg: Bool) -> None:
        self.start_scene = msg.data

    def _on_amr_ready(self, msg: String) -> None:
        self.amr_ready_queue.append(msg.data)

    def _on_amr_carrying_complete(self, msg: String) -> None:
        self.amr_carrying_complete_queue.append(msg.data)

    def publish_shoe_stop(self) -> None:
        msg = Bool()
        msg.data = True
        self._shoe_stop_publisher.publish(msg)

    def publish_pick_done(self, amr_id: int) -> None:
        msg = Int32()
        msg.data = amr_id
        self._pick_done_publisher.publish(msg)

    def publish_place_done(self, amr_id: int) -> None:
        msg = Int32()
        msg.data = amr_id
        self._place_done_publisher.publish(msg)


# =====================================================================
# 시간 설정
# =====================================================================

CONVEYOR_RESTART_DELAY_SEC: Final[float] = 2.0
SHOE_ACTIVATION_DELAY_SEC: Final[float] = 8.0
STOP_TRIGGER_DELAY_SEC: Final[float] = 0.7
STOP_TRIGGER_RESTART_DELAY_SEC: Final[float] = 2.0

# StopTrigger는 신발 콜라이더의 "앞쪽 끝"이 트리거 볼륨에 닿는 순간을 감지한다
# - 신발 사이즈(240/260/280)가 다르면 그 순간의 신발 중심 world X가 사이즈만큼
# 달라져서, 그 뒤 같은 시간만큼 벨트가 움직여도 최종 정지 위치(중심 기준)가
# 사이즈마다 어긋난다(2026-07-28, 사용자 확인 - "사이즈 랜덤화 이후 카메라
# 이미지에서 신발 위치가 사이클마다 달라짐"). 그래서 정지 순간엔 목표 world XY로
# 신발 위치를 강제로 스냅해서, 사이즈와 무관하게 항상 같은 위치에서 멈추게
# 한다. 목표 좌표는 D455_3(바닥/실측용 카메라, isaac_shoe_bottom_sdg.py 참고)의
# world XY를 그대로 썼다(사용자 지정 - "d455 카메라 3번 위치랑 xy 위치 맞추면
# 될 거 같은데") - 카메라가 내려다보는 지점과 신발이 항상 정확히 겹치게 된다.
SHOE_STOP_TARGET_X: Final[float] = -1.2798820262455302
SHOE_STOP_TARGET_Y: Final[float] = 2.748851058482757


# =====================================================================
# 신발 경로
# =====================================================================

SHOE_PATHS: Final[tuple[str, ...]] = (
    "/World/shoes/sneakers",
    "/World/shoes/sneaker_0001_red_240_ok_01",
    "/World/shoes/sneaker_0001_red_240_ok_02",
)

# 신발 활성화 시 랜덤으로 뽑는 사이즈(mm, 실제 발길이 - 240/260/280 사이즈
# 체계) - 기본 에셋(sneaker_240)이 240 기준이라, 다른 사이즈는 이 비율로
# 균등 스케일해서 흉내낸다.
SHOE_SIZES_MM: Final[tuple[int, ...]] = (240, 260, 280)
BASE_SHOE_SIZE_MM: Final[int] = 240
SHOE_TEAR_PROBABILITY: Final[float] = 0.2


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


def prepare_stage() -> None:
    """World.reset() 전에 기존 충돌 그래프를 정리합니다."""

    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("열려 있는 USD Stage가 없습니다.")

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
    def __init__(self, amr_arm_controllers: dict[int, object] | None = None) -> None:
        # {amr 번호(int): AmrArmController} - /fms/amr_ready, /fms/amr_carrying_complete의
        # amr_id로 어떤 로봇의 로봇팔을 움직일지 찾는다. 지금은 amr_1(=1)만 등록돼
        # 있어도 되고, 아예 안 넘기면(비어있으면) pick&place 자동화 없이 기존
        # 컨베이어/신발 로직만 동작한다.
        self._amr_arm_controllers: dict[int, object] = amr_arm_controllers or {}

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

        # 신발마다: [(TearPatch mesh prim, tear 위치 선정용 토폴로지), ...]
        # (메시 하나당 하나) - _setup_shoe_damage_rig에서 한 번만 채운다.
        self._shoe_damage_rigs: dict[
            str,
            list[tuple[UsdGeom.Mesh, dict]],
        ] = {}
        self._shoe_damage_materials = create_condition_materials(
            self._stage
        )

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

            self._setup_shoe_damage_rig(
                shoe_path,
                shoe_prim,
            )

            shoe_prim.SetActive(False)

            print(
                "[simulation] 시작 신발 비활성화: "
                f"{shoe_path}"
            )

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
        self._stop_trigger_handled_shoes: set[str] = set()

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
        self._shoe_activation_delay_remaining: float | None = None

        print(
            "[simulation] 시작 신호 대기: "
            f"{START_SCENE_TOPIC} (std_msgs/msg/Bool)"
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
            if shoe_path not in self._stop_trigger_handled_shoes:
                self._stop_trigger_handled_shoes.add(shoe_path)
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

    def _setup_shoe_damage_rig(
        self,
        shoe_path: str,
        shoe_prim: Usd.Prim,
    ) -> None:
        """신발의 각 Mesh 밑에 TearPatch 오버레이 메시를 하나씩 만들어 둔다
        (기본은 숨김). 실제로 face를 잘라내는 대신(export_damaged_shoe_usd.py는
        일회성 파일 export라 그렇게 해도 되지만, 이 풀은 활성화/비활성화를
        반복 재사용하므로 되돌릴 수 없는 변형은 쓸 수 없다) 위에 얇은 패치를
        덧씌워서 tear처럼 보이게 한다."""

        meshes = [
            UsdGeom.Mesh(d)
            for d in Usd.PrimRange(shoe_prim)
            if d.IsA(UsdGeom.Mesh)
        ]

        rigs: list[tuple[UsdGeom.Mesh, dict]] = []

        for mesh in meshes:
            UsdShade.MaterialBindingAPI(mesh).Bind(
                self._shoe_damage_materials["normal"]
            )
            topo = cache_mesh_topology(mesh)

            patch = make_ellipse_patch(
                self._stage,
                mesh.GetPath().AppendChild("TearPatch"),
            )
            UsdShade.MaterialBindingAPI(patch).Bind(
                self._shoe_damage_materials["tear"]
            )
            UsdGeom.Imageable(patch).MakeInvisible()

            rigs.append((patch, topo))

        self._shoe_damage_rigs[shoe_path] = rigs

    def _apply_random_shoe_variant(
        self,
        shoe_path: str,
        shoe_prim: Usd.Prim,
    ) -> None:
        """활성화되는 신발에 랜덤 사이즈(240/260/280)와 상태(정상/tear)를
        입힌다."""

        size_mm = random.choice(SHOE_SIZES_MM)
        scale = size_mm / BASE_SHOE_SIZE_MM

        xformable = UsdGeom.Xformable(shoe_prim)
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        # Z는 스케일하지 않는다 - 높이까지 늘리면 사이즈가 커질수록 신발이
        # 컨베이어 바닥 기준으로 떠버리거나 파묻힌다(2026-07-28, 사용자 확인).
        scale_op.Set(Gf.Vec3f(scale, scale, 1.0))

        condition = "tear" if random.random() < SHOE_TEAR_PROBABILITY else "ok"
        rigs = self._shoe_damage_rigs.get(shoe_path, [])

        for patch, topo in rigs:
            if condition == "tear":
                center, axis1, axis2, normal, rx, ry = pick_tear_placement(
                    topo, random
                )
                update_ellipse_patch(patch, center, axis1, axis2, normal, rx, ry)
                UsdGeom.Imageable(patch).MakeVisible()
            else:
                UsdGeom.Imageable(patch).MakeInvisible()

        print(
            "[simulation] 신발 변형 적용: "
            f"{shoe_path} size={size_mm} condition={condition}"
        )

    def _get_shoe(
        self,
        shoe_path: str,
    ) -> Usd.Prim:
        return self._stage.GetPrimAtPath(
            shoe_path
        )

    def _activate_random_shoe(self) -> None:
        """비활성 상태인 신발 중 하나를 랜덤하게 활성화합니다."""

        if (
            self._emergency_stopped
            or self._ros_node.emergency_stop
        ):
            print(
                "[SHOE ACTIVATE BLOCKED] "
                "비상정지 상태이므로 신발 활성화를 차단합니다."
            )
            return

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

        # 다시 투입되는 신발에 대해서만 StopTrigger 감지를 재허용합니다.
        self._stop_trigger_handled_shoes.discard(selected_path)

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

        self._apply_random_shoe_variant(
            selected_path,
            shoe_prim,
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
        """비전 컨베이어 상태와 신발 생성 타이머를 기록한 뒤 정지합니다."""

        self._working_speed_before_emergency = (
            self._get_attribute_float(
                self._working_conveyor_velocity_attr,
                default=1.0,
            )
        )

        # 신발 생성까지 남은 시간을 저장합니다.
        if self._next_activation_time is not None:
            self._shoe_activation_delay_remaining = max(
                0.0,
                self._next_activation_time - now,
            )
            self._next_activation_time = None
        else:
            self._shoe_activation_delay_remaining = None

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
            "[EMERGENCY] 전체 컨베이어 정지"
        )

    def _leave_emergency_stop(self, now: float) -> None:
        """컨베이어와 신발 생성 타이머를 복원합니다."""

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
        else:
            self._working_conveyor_velocity_attr.Set(1.0)
            self._vision_stop_started_at = None
            self._vision_resume_deadline = None

        self._vision_delay_remaining = 0.0

        # 비상정지 전에 남아 있던 시간부터 다시 시작합니다.
        if (
            self._scene_started
            and self._shoe_activation_delay_remaining is not None
        ):
            self._next_activation_time = (
                now + self._shoe_activation_delay_remaining
            )

        self._shoe_activation_delay_remaining = None

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

    def _read_shoe_world_x(
        self,
        shoe_path: str,
    ) -> float | None:
        shoe_prim = self._get_shoe(shoe_path)

        if not shoe_prim.IsValid():
            return None

        world_xf = UsdGeom.Xformable(shoe_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        return float(world_xf.ExtractTranslation()[0])

    def _start_stop_trigger_sequence(
        self,
        shoe_path: str,
    ) -> None:
        """신발별 StopTrigger 시퀀스를 시작합니다."""

        # 같은 신발의 시퀀스가 이미 진행 중이면 중복 시작하지 않습니다.
        for sequence in self._stop_trigger_sequences:
            if str(sequence["shoe_path"]) == shoe_path:
                print(
                    "[STOP_SEQUENCE DUPLICATE] "
                    f"shoe={shoe_path}, "
                    "이미 진행 중인 시퀀스이므로 무시"
                )
                return

        started_at = self._timeline.get_current_time()

        self._stop_trigger_sequences.append(
            {
                "shoe_path": shoe_path,
                "started_at": started_at,
                "start_x": self._read_shoe_world_x(shoe_path),
                "stopped": False,
                "restarted": False,
            }
        )

        print(
            "[STOP_SEQUENCE START] "
            f"shoe={shoe_path}, "
            f"time={started_at:.3f}, "
            f"active_sequences={len(self._stop_trigger_sequences)}"
        )

    def _snap_shoe_to_stop_position(
        self,
        shoe_path: str,
    ) -> None:
        """정지 순간 신발의 world XY를 SHOE_STOP_TARGET_X/Y로 강제 스냅한다(Z는
        그대로 둔다) - 사이즈별 정지 위치 불일치 문제 참고(SHOE_STOP_TARGET_X
        정의부 주석)."""

        shoe_prim = self._get_shoe(shoe_path)

        if not shoe_prim.IsValid():
            return

        xformable = UsdGeom.Xformable(shoe_prim)
        translate_op = None

        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            return

        world_xf = xformable.ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_pos = world_xf.ExtractTranslation()

        parent_xf = UsdGeom.Xformable(
            shoe_prim.GetParent()
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        target_world = Gf.Vec3d(
            SHOE_STOP_TARGET_X,
            SHOE_STOP_TARGET_Y,
            world_pos[2],
        )
        target_local = parent_xf.GetInverse().Transform(target_world)

        translate_op.Set(target_local)

        for attr_name in (
            "physics:velocity",
            "physics:angularVelocity",
        ):
            attr = shoe_prim.GetAttribute(attr_name)

            if attr.IsValid():
                attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _update_stop_trigger_sequences(
        self,
        now: float,
    ) -> None:
        remaining: list[dict[str, object]] = []

        for sequence in self._stop_trigger_sequences:
            shoe_path = str(sequence["shoe_path"])
            started_at = float(sequence["started_at"])
            elapsed = max(0.0, now - started_at)

            if not bool(sequence["stopped"]):
                # 원래는 "접촉 후 고정 시간"만 보고 멈췄는데, 그 시점의 실제
                # 위치는 목표(SHOE_STOP_TARGET_X)를 이미 살짝 지나쳐 있어서
                # _snap_shoe_to_stop_position이 뒤로 순간이동시키는 게 눈에
                # 보였다(2026-07-28, 사용자 확인 - "정지 위치를 살짝 지나쳤다가
                # 다시 정지위치로 순간이동"). 그래서 매 프레임 신발의 실제
                # world X가 시작 위치 기준으로 목표를 "막 지나친 순간"을 감지해서
                # 그때 바로 멈추면, 스냅 보정량이 거의 0이라 순간이동이 안
                # 보인다. 무언가에 걸려 목표에 영영 도달하지 못하는 경우에
                # 대비해 STOP_TRIGGER_DELAY_SEC를 그대로 안전장치(최대 대기
                # 시간)로 남겨둔다.
                start_x = sequence.get("start_x")
                current_x = self._read_shoe_world_x(shoe_path)

                reached_target = (
                    start_x is not None
                    and current_x is not None
                    and (start_x - SHOE_STOP_TARGET_X)
                    * (current_x - SHOE_STOP_TARGET_X)
                    <= 0.0
                )
                timed_out = elapsed >= STOP_TRIGGER_DELAY_SEC

                if reached_target or timed_out:
                    self._working_conveyor_velocity_attr.Set(0.0)
                    self._snap_shoe_to_stop_position(shoe_path)
                    self._publish_shoe_stop()
                    sequence["stopped"] = True

                    print(
                        "[STOP_SEQUENCE STOP] "
                        f"shoe={shoe_path}, "
                        f"reached_target={reached_target}, "
                        f"timed_out={timed_out}, "
                        f"elapsed={elapsed:.3f}, "
                        f"time={now:.3f}"
                    )

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

                print(
                    "[STOP_SEQUENCE RESTART] "
                    f"shoe={shoe_path}, "
                    f"elapsed={elapsed:.3f}, "
                    f"time={now:.3f}"
                )

            if not (
                bool(sequence["stopped"])
                and bool(sequence["restarted"])
            ):
                remaining.append(sequence)
            else:
                print(
                    "[STOP_SEQUENCE COMPLETE] "
                    f"shoe={shoe_path}, "
                    f"remaining_sequences={len(remaining)}"
                )

        self._stop_trigger_sequences = remaining

    # =================================================================
    # AMR 로봇팔 pick/place (amr_pick_place.AmrArmController)
    # =================================================================

    def _update_amr_pick_place(self) -> None:
        """/fms/amr_ready -> pick 시작, /fms/amr_carrying_complete -> place
        시작, 매 프레임 각 컨트롤러를 진행시키고 완료되면 /sim/pick_done,
        /sim/place_done을 발행한다."""

        while self._ros_node.amr_ready_queue:
            payload = self._ros_node.amr_ready_queue.pop(0)
            try:
                parsed = json.loads(payload)
                amr_id = int(parsed["amr_id"])
                count = int(parsed.get("count", 1))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"[amr_pick_place] amr_ready 메시지 파싱 실패: {payload!r}", flush=True)
                continue
            controller = self._amr_arm_controllers.get(amr_id)
            if controller is None:
                continue  # 이 AMR은 로봇팔 자동화 대상이 아님(아직 amr_1만 등록)
            if not controller.start_pick(count):
                print(
                    f"[amr_pick_place] amr_{amr_id}: pick(count={count}) 시작 실패(phase={controller.phase})",
                    flush=True,
                )

        while self._ros_node.amr_carrying_complete_queue:
            payload = self._ros_node.amr_carrying_complete_queue.pop(0)
            try:
                parsed = json.loads(payload)
                amr_id = int(parsed["amr_id"])
                shelf_num = int(parsed["shelf_num"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                print(f"[amr_pick_place] amr_carrying_complete 메시지 파싱 실패: {payload!r}", flush=True)
                continue
            controller = self._amr_arm_controllers.get(amr_id)
            if controller is None:
                continue
            if not controller.start_place(shelf_num):
                print(
                    f"[amr_pick_place] amr_{amr_id}: place(shelf_num={shelf_num}) 시작 실패"
                    f"(phase={controller.phase}, 채워진 슬롯 {controller.occupied_slot_count}개)",
                    flush=True,
                )

        for amr_id, controller in self._amr_arm_controllers.items():
            event = controller.update()
            if event == "pick_done":
                self._ros_node.publish_pick_done(amr_id)
                print(f"[amr_pick_place] amr_{amr_id}: {SIM_PICK_DONE_TOPIC} 발행", flush=True)
            elif event == "place_done":
                self._ros_node.publish_place_done(amr_id)
                print(f"[amr_pick_place] amr_{amr_id}: {SIM_PLACE_DONE_TOPIC} 발행", flush=True)
            elif event == "pick_failed":
                print(f"[amr_pick_place] amr_{amr_id}: pick 실패 - amr_ready 재발행 필요", flush=True)

    # =================================================================
    # 매 프레임 실행
    # =================================================================

    def update(self) -> None:
        """시뮬레이션 실행 중 매 프레임 호출합니다."""

        if not self._timeline.is_playing():
            return

        rclpy.spin_once(self._ros_node, timeout_sec=0.0)

        # 컨베이어/신발 생성과 마찬가지로 비상정지 중엔 로봇팔 pick&place도
        # 새 명령을 내리지 않는다 - 마지막으로 적용된 관절 명령이 그대로
        # 유지되니 팔은 그 자리에서 멈추고, 해제되면 하던 동작을 이어서
        # 계속한다(2026-07-27, 사용자 요청).
        if not self._emergency_stopped:
            self._update_amr_pick_place()

        now = self._timeline.get_current_time()

        emergency_signal = self._ros_node.emergency_stop

        if emergency_signal and not self._emergency_stopped:
            self._enter_emergency_stop(now)

        elif not emergency_signal and self._emergency_stopped:
            self._leave_emergency_stop(now)

        start_scene_signal = self._ros_node.start_scene

        if start_scene_signal and not self._previous_start_scene_signal:
            self._scene_started = True

            if self._emergency_stopped:
                # 비상정지 중에는 신발을 즉시 활성화하지 않습니다.
                # 해제 후 설정된 시간만큼 기다렸다가 활성화합니다.
                self._shoe_activation_delay_remaining = (
                    SHOE_ACTIVATION_DELAY_SEC
                )
                self._next_activation_time = None

                print(
                    "[simulation] 비상정지 중 시작 신호 수신: "
                    "신발 활성화 대기"
                )

            else:
                self._activate_random_shoe()

                self._next_activation_time = (
                    now + SHOE_ACTIVATION_DELAY_SEC
                )

                print(
                    "[simulation] 시작 신호로 첫 신발 즉시 활성화"
                )

        elif (
            not start_scene_signal
            and self._previous_start_scene_signal
        ):
            self._scene_started = False
            self._next_activation_time = None
            self._shoe_activation_delay_remaining = None

            print(
                f"[simulation] {START_SCENE_TOPIC}: False 수신"
            )

        self._previous_start_scene_signal = start_scene_signal

        if self._emergency_stopped:
            # 다른 Action Graph가 속도를 다시 쓰더라도
            # 매 프레임 컨베이어 속도를 0으로 강제합니다.
            self._stop_all_conveyors()
            return

        while self._pending_stop_trigger_shoes:
            shoe_path = self._pending_stop_trigger_shoes.pop(0)

            print(
                "[STOP_PENDING POP] "
                f"shoe={shoe_path}, "
                f"remaining_pending="
                f"{len(self._pending_stop_trigger_shoes)}"
            )

            self._start_stop_trigger_sequence(shoe_path)

        self._update_stop_trigger_sequences(now)

        # 기존 비전 컨베이어 정지 및 재시작 시간 처리
        self._update_vision_conveyor_state(now)

        while self._pending_trigger_shoes:
            shoe_path = self._pending_trigger_shoes.pop(0)

            shoe_prim = self._get_shoe(shoe_path)

            if (
                shoe_prim.IsValid()
                and shoe_prim.IsActive()
            ):
                self._recycle_shoe(shoe_path)

        while self._pending_trigger4_shoes:
            shoe_path = self._pending_trigger4_shoes.pop(0)

            shoe_prim = self._get_shoe(shoe_path)

            if (
                shoe_prim.IsValid()
                and shoe_prim.IsActive()
            ):
                self._recycle_shoe(shoe_path)

        if (
            self._scene_started
            and not self._emergency_stopped
            and not self._ros_node.emergency_stop
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
        self._stop_trigger_handled_shoes.clear()
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


def create_simulation_node(
    amr_arm_controllers: dict[int, object] | None = None,
) -> SimulationNode:
    """메인 실행 코드에서 호출할 생성 함수입니다."""

    return SimulationNode(amr_arm_controllers=amr_arm_controllers)