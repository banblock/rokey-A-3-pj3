"""AMR(nova_carter_ur5e_surface_gripper) 로봇팔의 pick(픽업)/place(랙 배치) 자동화.

FMS(fms_node.py)가 로봇이 픽업 지점에 도착하면 /fms/amr_ready(이번 트립에 실을
박스 수 count 포함)를, 랙 목표 지점에 도착하면 /fms/amr_carrying_complete를
(박스 1개당 1번씩, 트립당 여러 번) 발행한다. simulation_node.py가 이 두 토픽을
감지해서 여기 있는 AmrArmController로 실제 로봇팔(RMPflow 기반, UR5E/rmpflow/
의 검증된 pick&place 파이프라인 재사용)을 움직이고, 끝나면 /sim/pick_done,
/sim/place_done을 발행해서 FMS에 알린다(fms_node.py는 이미 이 두 토픽을
구독해서 "loading"/"dwelling" 상태를 해제하도록 만들어져 있다).

pick과 place는 서로 다른 시점(그 사이에 AMR이 실제로 픽업→랙까지 주행)에
트리거된다. AMR 뒤에 붙은 StorageBox(build_v3_assembler.py 참고, chassis_link에
로컬 오프셋만으로 고정된 5슬롯짜리 선반 - fms_node.py의 MAX_SHOES_PER_TRIP과
동일)를 중간 적재소로 써서, 사이클 하나하나는 AMR이 정지해 있는 동안(픽업
대기 중, 랙 도착 후) 완결되는 독립적인 pick&place로 나눈다:

1. amr_ready(count)가 오면: count번 반복해서 - "pallet"(픽업대) 위에 새 박스를
   스폰 -> 흡착 -> 비어있는 슬롯 위에 내려놓기 -> 그 슬롯에 FixedJoint로
   용접(AMR 주행 중 안 떨어지게) - 전부 끝나면 /sim/pick_done 발행.
2. amr_carrying_complete(shelf_num 포함)가 올 때마다(그 사이 AMR은 실제로
   랙까지 주행했다): 그 시점의 AMR 실제 pose로 RMPflow base pose/장애물을
   갱신 -> 가장 먼저 채워진(FIFO) 슬롯의 용접 해제 -> 그 박스를 다시 흡착 ->
   shelf_num에 해당하는 pallet(0=pallet_01/280, 1=pallet_02/260,
   2=pallet_03/240) 위에 내려놓기(그 자리에 그대로 둔다 - 랙에 쌓이는 모습을
   보여주기 위함) -> /sim/place_done 발행.

pick/place 목표는 demo0725.usd의 실제 pallet prim(PICK_PALLET_PATH,
PLACE_PALLET_PATHS)을 쓴다 - fleet_config_test1.py의 내비게이션 노드 좌표는
이 pallet들의 "중심"이 아니라 통로 쪽 정차 위치라서(2026-07-26 확인, pallet
크기의 절반만큼 떨어져 있음), pallet의 중심이 아니라 AMR과 가장 가까운 윗면
가장자리 지점을 목표로 계산한다(_nearest_point_on_pallet_top).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.types import ArticulationAction

_RMPFLOW_DIR = Path("/home/rokey/cobot3_ws/isaacpjt/UR5E/rmpflow")
if str(_RMPFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_RMPFLOW_DIR))

from ur5e_surface_gripper_pick_place_controller import PickPlaceController  # noqa: E402
from trigger_surface_gripper import TriggerSurfaceGripper  # noqa: E402

CONE_CONTACT_LOCAL_OFFSET = (0.0, 0.0, -0.0015)
# event1(pick 하강)/event4(pick 후 상승)는 원래 0.0015/0.00125로 매우 느렸다 -
# RMPflow가 pallet 근처에서 불안정해지던 시절(관절 속도 폭주) 안전 마진으로
# 낮춰둔 값인데, end_effector_initial_height를 pallet 높이에 맞게 고친 뒤로는
# (2026-07-26) 그 불안정 자체가 해소돼서 더 빨라도 된다 - 사용자가 "내려갈 때/
# 올라올 때 느리다"고 확인, 두 phase를 3배 빠르게 올렸다. event1을 다시
# 느리게(0.0025) 낮춰도 접촉 시 박스가 밀리는 정도(XY로 약 2.5cm)가 거의
# 똑같아서(헤드리스로 실측 비교) 하강 속도 자체가 원인은 아닌 것으로
# 보인다 - 트리거 콘의 접촉 판정 허용 오차 때문에 생기는 것으로 보이고,
# 속도를 늦춰도 줄어들지 않으니 굳이 느리게 할 필요가 없다.
EVENTS_DT = [0.008, 0.004, 0.05, 0.3, 0.0035, 0.01, 0.00125, 0.15, 0.008, 0.08]
PICK_DESCEND_OFFSET_Z = -0.015  # 표면보다 살짝 아래를 목표로 잡아 실제 접촉을 보장(gui_pick_place_demo.py와 동일)
# 직전 반복(흡착/용접)이 끝난 직후엔 그 충격(jolt)으로 섀시가 아주 잠깐
# 흔들릴 수 있다 - 그 순간에 바로 다음 반복의 섀시 pose를 읽으면(단 1프레임
# 스냅샷) 목표 위치가 완전히 엉뚱하게 계산되는 회귀가 실제로 있었다(헤드리스로
# 확인 - 보관함 목표 z가 정상 범위 밖으로 튐 -> 흡착 실패). 다음 반복 시작 전
# 이만큼 프레임 동안 아무 것도 안 하고 흔들림이 가라앉기를 기다린다.
COOLDOWN_STEPS = 60

# 사이클이 끝난 뒤 팔이 멈추는 관절 자세는(RMPflow가 스스로 고른 경로라)
# 매번 조금씩 다르다 - 그 애매한 자세에서 다음 목표(특히 pallet처럼 높은
# 곳)를 향해 IK를 풀면 가끔 특이점 근처에서 RMPflow가 불안정해져서 관절
# 속도가 초당 10rad/s 이상으로 치솟고, 그 반작용으로 AMR 차체가 실제로
# 밀리는 문제가 헤드리스로 확인됐다. 그래서 매 반복이 끝날 때마다 RMPflow를
# 거치지 않고 직접 관절 제어로 항상 같은 스폰 홈 자세로 확실히 복귀시킨
# 뒤에만 다음 목표의 IK를 시작한다.
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
SPAWN_JOINT_ANGLES_DEG = [180, -90, 90, -90, -90, 0]  # build_v3_assembler.py와 동일해야 함
RETURN_HOME_STEPS = 200
RETURN_HOME_TOLERANCE_RAD = 0.02

# pallet처럼 멀리/높이 있는 목표를 향할 때 RMPflow가 가끔(특이점 근처로
# 보임) 불안정해져서 관절 속도가 초당 10rad/s 이상으로 치솟는 게 헤드리스로
# 확인됐다(장애물 회피를 꺼도 재현 - RMPflow 자체 불안정) - 원인을 RMPflow
# 내부에서 고치는 대신, 매 스텝 명령을 이 이상 못 튀게 직접 제한한다.
MAX_JOINT_STEP_RAD = 0.03  # 60Hz 기준 대략 1.8rad/s에 해당

# 위의 여러 안전장치(팔레트 장애물 등록, 홈 복귀, 속도 클램프)를 적용해도
# RMPflow가 가끔 이번 시도에서만 흡착에 실패할 수 있다 - "픽업은 무조건
# 성공해야 한다"는 요구사항을 실질적으로 만족시키기 위해, 실패한 슬롯은
# 건너뛰지 않고 이 횟수만큼 같은 자리에서 재시도한다.
MAX_PICK_RETRIES = 3

DEFAULT_BOX_SIZE = np.array([0.28, 0.20, 0.11])

# StorageBox(build_v3_assembler.py의 STORAGE_BOX_LOCAL_POS/SIZE와 반드시 같이
# 맞춰야 한다) 위 6칸(2개씩 3층)의 chassis-local 안착 위치 - fms_node.py의
# MAX_SHOES_PER_TRIP(5)보다 1칸 여유. 폭(Y)은 차체 폭과 같아서 옆으로는 2개
# (±0.13, 박스폭 0.23 기준 살짝 여유)만 놓고, 그 위로 박스를 쌓아 올린다(이송
# 중엔 FixedJoint로 용접되니 실제로 박스 위에 박스를 얹어도 안 떨어진다).
# 층 간격 = 박스 높이(0.13) + 여유(0.02). 아래층부터 순서대로 채운다.
STORAGE_LAYER_HEIGHT = 0.15
# RMPflow에 이 좌표를 목표로 그대로 넘기면, 아래층일수록 목표보다 한참
# 위(최대 ~10cm)에서 멈춰버린다 - "저장소까지 안 내려가짐"의 원인이었다
# (2026-07-26, check_all_slots_gap.py로 6칸 전부 실측). 처음엔 "부족한 만큼
# 목표를 낮추면 되겠지"라 생각했는데, 실제로 해보니(목표를 낮췄더니) 실제
# 도달 높이(apex_z)는 거의 그대로고 격차만 더 벌어졌다 - 즉 목표가 낮아서
# 못 미치는 게 아니라, 그 XY 위치에서 팔이 실제로 내려갈 수 있는 물리적
# 한계(자체 충돌 회피로 보임)가 따로 있고 그 이하로는 목표를 아무리 낮춰도
# 소용없다는 뜻이다. 그래서 반대로, 실측된 "실제 도달 높이"에 맞춰 목표를
# 그만큼 올려서(목표=한계 부근) 팔이 목표에 최대한 정확히 도달하게 한다 -
# _weld_box_to_slot은 박스의 "실제 도착 위치"에 용접하므로 이렇게 목표를
# 올려도 최종적으로 그 높이에 안정적으로 놓인다.
_STORAGE_LAYER_Z_COMPENSATION = [0.094, 0.075, 0.0]
STORAGE_SLOT_LOCAL_POSITIONS = [
    np.array([-0.65, y, 0.44 + layer * STORAGE_LAYER_HEIGHT + _STORAGE_LAYER_Z_COMPENSATION[layer]])
    for layer in range(3)
    for y in (-0.13, 0.13)
]

# 실제 demo0725.usd의 pallet prim들을 pick/place 대상으로 쓴다(2026-07-26
# 확정): 픽업은 "pallet", 랙 배치는 크기별로 pallet_01(280)/pallet_02(260)/
# pallet_03(240) - fleet_config_test1.py의 PICKUP_A/RackA_280/RackA_260/
# RackA_240과 각각 짝을 이룬다(같은 순서로 SHELF_INDEX가 매겨짐 - fms_node.py
# 참고). fleet_config 노드 좌표는 pallet의 "중심"이 아니라 아이슬 쪽 통로
# 위치라 pallet bbox 중심과는 X는 거의 일치하지만 Y가 pallet 크기(약 1.3m)의
# 절반만큼 떨어져 있다 - 그래서 pallet의 중심이 아니라 AMR과 가장 가까운
# 윗면 가장자리 지점을 목표로 계산한다(_nearest_point_on_pallet_top 참고).
PICK_PALLET_PATH = "/World/pallet"
PLACE_PALLET_PATHS = ["/World/pallet_01", "/World/pallet_02", "/World/pallet_03"]
# pick pallet 바로 옆(약 1.2m)에 실제 컨베이어(convey_01)가 있어서 "옆 컨베이어와
# 부딪힘" 문제가 있었다(2026-07-26). 등록을 아예 껐다가 헤드리스로 비교해보니
# (관절 속도 최대치/완료까지 걸리는 스텝 수 모두) 등록 안 한 쪽이 오히려 더
# 나빴다(등록: ~5100스텝/최대 3.1rad/s, 미등록: ~7000스텝/최대 7rad/s) - 즉
# 이 장애물이 실제로는 안정화에 도움이 된다. 등록은 유지하고, "부딪힘"으로
# 보인 것이 실제 충돌인지 그냥 카메라 각도상 가까워 보이는 정상 이동인지는
# 시각적으로 다시 확인이 필요하다.
EXTRA_STATIC_OBSTACLE_PATHS = ["/World/ReturnCell/convey_01/ConveyorTrack"]  # shelf_num 0/1/2 = 280/260/240
# pallet 가장자리 바로 그 지점은 팔 자신의 정지 자세와 겹칠 수 있어 안쪽으로
# 이만큼 들여서 목표를 잡는다(GUI에서 확인된 회귀 - 박스가 스폰되자마자 팔과
# 충돌해 튕겨나감).
PALLET_EDGE_MARGIN_M = 0.30
# pallet 장애물의 윗면을 이만큼 깎아내린다 - 안 그러면 회피 장애물의 윗면과
# 박스를 놓는/집는 목표 지점(pallet 윗면 바로 위)이 거의 같은 높이라 "닿아야
# 한다"와 "피해야 한다"가 충돌해서 팔이 아예 접근을 거부하는 문제가 있었다
# (헤드리스로 확인 - 장애물 등록 직후 pick이 전부 실패로 바뀜).
PALLET_OBSTACLE_TOP_MARGIN_M = 0.20

# pallet은 벽면만 있고 위가 뚫린 케이지 구조라, 팔이 옆에서 접근하면 실제
# 벽에 스치기 쉽다(GUI에서 확인) - RMPflow의 장애물 회피에 통째로 맡기는
# 대신, 먼저 벽 높이보다 확실히 위(목표와 같은 XY, 벽 위)로 이동시킨 뒤
# 그 지점에서 수직으로만 내려가게 직접 경로를 지정한다.
SAFE_HOVER_MARGIN_M = 0.20
HOVER_STEP_BUDGET = 300
HOVER_TOLERANCE_M = 0.03


def prepare_cone_triggers(robot_prim_path: str) -> None:
    """Cone_0~3에 PhysxTriggerStateAPI를 붙여 접촉 판정을 가능하게 한다.

    반드시 world.reset()(첫 physics step) 이전에 호출해야 한다 - PhysX가
    최초 reset 시점에 콜리전/트리거 표현을 스냅샷하는 것으로 보이며, reset
    이후에 API를 붙이면 HasAPI()는 True로 보여도 실제 겹침 조회
    (GetTriggeredCollisionsRel)가 영원히 빈 리스트만 반환한다(헤드리스로
    직접 확인한 회귀 - reset 전/후로 나눠 비교했을 때 전자만 정상 동작).
    """
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    for i in range(4):
        cone_prim = stage.GetPrimAtPath(f"{robot_prim_path}/arm_mount/ur5e/GripperBase/Cone_{i}")
        if cone_prim.IsValid() and not cone_prim.HasAPI(PhysxSchema.PhysxTriggerStateAPI):
            PhysxSchema.PhysxTriggerStateAPI.Apply(cone_prim)


class AmrArmController:
    """AMR 한 대(robot_id)의 로봇팔 pick/place 상태 머신.

    phase: "idle" -> "pick_to_storage"(count번 내부 반복) -> "idle" ->
    "place_from_storage" -> "idle"
    """

    def __init__(
        self,
        robot_id: str,
        robot_prim_path: str,
        box_size: np.ndarray = DEFAULT_BOX_SIZE,
        physics_sim_view=None,
    ) -> None:
        self.robot_id = robot_id
        self._robot_prim_path = robot_prim_path
        self._chassis_path = f"{robot_prim_path}/chassis_link"
        self._ur5e_base_link_path = f"{robot_prim_path}/arm_mount/ur5e/base_link"
        self._gripper_base_path = f"{robot_prim_path}/arm_mount/ur5e/GripperBase"
        self._storage_box_path = f"{self._chassis_path}/StorageBox"
        self._cone_paths = [f"{self._gripper_base_path}/Cone_{i}" for i in range(4)]
        self._grasp_joint_path = f"{self._gripper_base_path}/GraspJoint"
        self._box_size = np.array(box_size, dtype=float)

        from isaacsim.core.prims import SingleArticulation

        # NOTE: prepare_cone_triggers(robot_prim_path)를 world.reset() 전에 이미
        # 호출해뒀어야 한다 - 여기(생성자, reset 이후에 호출됨)서 다시 적용해봐야
        # 늦다. 안전망으로 한 번 더 시도는 하되, 이게 실제로 필요했다면 이미 늦은
        # 것이니 호출하는 쪽의 순서를 고쳐야 한다.
        prepare_cone_triggers(robot_prim_path)

        self._robot = SingleArticulation(prim_path=self._chassis_path, name=f"{robot_id}_articulation")
        self._robot.initialize()

        dof_names = list(self._robot.dof_names)
        self._arm_joint_indices = np.array([dof_names.index(name) for name in UR5E_JOINT_NAMES])
        self._arm_home_positions = np.array([np.deg2rad(a) for a in SPAWN_JOINT_ANGLES_DEG], dtype=float)

        # 콘 트리거가 우리가 스폰한 박스(_pick_box_{robot_id}_*) 말고 pallet
        # 벽면 같은 주변 지오메트리와도 겹칠 수 있어서, 실제로 흡착/접촉으로
        # 인정할 대상을 이 패턴으로 제한한다(GUI에서 pallet 벽면을 흡착해버린
        # 사고 확인 후 추가).
        box_prefix = f"/_pick_box_{robot_id}_"

        def _is_our_pick_box(prim_path: str) -> bool:
            return box_prefix in prim_path

        self._target_filter = _is_our_pick_box

        self._gripper = TriggerSurfaceGripper(
            end_effector_prim_path=self._gripper_base_path,
            trigger_paths=self._cone_paths,
            joint_path=self._grasp_joint_path,
            tip_local_offset=CONE_CONTACT_LOCAL_OFFSET,
            target_filter=self._target_filter,
        )
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_num_dofs=self._robot.num_dof,
        )
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()

        base_position, base_orientation = self._read_base_pose()
        # exclude_names=("StorageBox",)로 StorageBox 자체는 챗시 장애물 bbox
        # 계산에서 빼지만, 형제 prim인 "visual"(차체 전체를 뭉뚱그린 단일
        # 메시)의 bbox가 StorageBox 실제 높이(~1.40m)보다 훨씬 위인 1.5m
        # 근처까지 그대로 덮고 있어서, 맨 아래 칸(슬롯 목표 1.467m)에 내려놓을
        # 때 RMPflow가 그 상공을 장애물로 착각해 목표보다 한참 위에서 멈추는
        # 문제가 있었다(2026-07-26, 사용자 확인 "저장소까지 안 내려가짐" -
        # check_chassis_obstacle_box.py로 실측). StorageBox의 실제 윗면보다
        # 살짝만 위까지만 챗시 장애물로 인정하도록 상한을 넘긴다.
        stage = omni.usd.get_context().get_stage()
        storage_prim = stage.GetPrimAtPath(self._storage_box_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        storage_top_z = float(bbox_cache.ComputeWorldBound(storage_prim).ComputeAlignedRange().GetMax()[2])
        self._controller = PickPlaceController(
            name=f"{robot_id}_pick_place_controller",
            gripper=self._gripper,
            robot_articulation=self._robot,
            end_effector_initial_height=0.55,
            events_dt=EVENTS_DT,
            robot_base_position=base_position,
            robot_base_orientation=base_orientation,
            chassis_obstacle_prim_path=self._chassis_path,
            chassis_obstacle_exclude_names=("StorageBox",),
            chassis_obstacle_max_z=storage_top_z + 0.02,
            contact_trigger_path=self._cone_paths,
            contact_target_filter=self._target_filter,
        )
        self._controller.reset()
        self._register_pallet_obstacles()

        self._phase = "idle"
        self._contact_seen = False
        self._grasp_confirmed = False
        self._welded_this_rep = False
        self._retry_count = 0
        self._cycle_target: Optional[np.ndarray] = None

        self._next_box_serial = 0
        # 슬롯 인덱스 -> 그 슬롯에 놓인 박스의 prim path (비어있으면 None)
        self._slot_box_paths: List[Optional[str]] = [None] * len(STORAGE_SLOT_LOCAL_POSITIONS)
        # 채워진 순서(FIFO) - place는 항상 이 리스트의 맨 앞 슬롯부터 뺀다.
        self._slot_fill_order: List[int] = []
        self._current_box_prim_path: Optional[str] = None
        self._current_slot_index: Optional[int] = None
        self._pick_remaining = 0
        self._cooldown_remaining = 0
        self._return_home_steps_left = 0
        # "returning_home" 단계가 끝난 뒤 실제로 할 일 - "next_rep"(다음 반복
        # 시작) 또는 이번 update() 호출에서 바깥으로 돌려줄 이벤트 문자열.
        self._after_return_home: Optional[str] = None

    # -----------------------------------------------------------------
    # 내부 helper - pose
    # -----------------------------------------------------------------

    def _read_base_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._ur5e_base_link_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        position = np.array(world.ExtractTranslation())
        quat = world.ExtractRotationQuat()
        orientation = np.array([quat.GetReal(), *quat.GetImaginary()])
        return position, orientation

    def _read_chassis_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._chassis_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        position = np.array(world.ExtractTranslation())
        quat = world.ExtractRotationQuat()
        return position, quat

    def _read_box_pose(self, box_prim_path: str) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(box_prim_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        return np.array(world.ExtractTranslation())

    def _refresh_base_pose_and_obstacle(self) -> None:
        """AMR이 실제로 이동한 뒤 pick/place를 이어서 하려면, RMPflow가 아는
        base pose와 차체 장애물(spawn 시점에 한 번만 등록된 static obstacle)을
        지금 실제 위치로 다시 맞춰야 한다."""
        position, orientation = self._read_base_pose()
        cspace = self._controller._cspace_controller
        cspace._motion_policy.set_robot_base_pose(robot_position=position, robot_orientation=orientation)

        obstacle_path = "/World/_rmpflow_chassis_obstacle"
        stage = omni.usd.get_context().get_stage()
        obstacle_prim = stage.GetPrimAtPath(obstacle_path)
        if obstacle_prim.IsValid():
            from isaacsim.core.api.objects import VisualCuboid

            # RMPFlowController._add_chassis_obstacle와 동일하게 StorageBox는
            # bbox 계산에서 제외한다 - 안 그러면 팔이 실제로 도달해야 하는
            # 적재 선반 자체가 "피해야 할 장애물"이 돼버려 목표와 회피가
            # 충돌한다.
            chassis_prim = stage.GetPrimAtPath(self._chassis_path)
            bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
            bmin = None
            bmax = None
            for child in chassis_prim.GetChildren():
                if child.GetName() == "StorageBox":
                    continue
                child_rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
                child_min, child_max = np.array(child_rng.GetMin()), np.array(child_rng.GetMax())
                if child_min[0] > child_max[0]:
                    continue
                bmin = child_min if bmin is None else np.minimum(bmin, child_min)
                bmax = child_max if bmax is None else np.maximum(bmax, child_max)
            if bmin is not None:
                center = (bmin + bmax) / 2.0
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)

    def _register_pallet_obstacles(self) -> None:
        """pallet(들)을 RMPflow의 정적 회피 장애물로 등록한다.

        chassis_obstacle_prim_path로 섀시는 이미 등록해뒀지만, pallet(케이지
        구조물) 자체는 등록이 안 돼 있어서 팔이 접근/이동하는 도중 실제로
        forearm_link가 pallet과 겹치는(충돌하는) 게 물리 오버랩 쿼리로
        확인됐다 - 그 충격으로 관절 속도가 초당 10rad/s 이상 치솟고 AMR 차체
        까지 밀리는 문제로 이어졌다. pallet은 고정된 static prim이라 여기서
        한 번만 등록하면 된다.
        """
        from isaacsim.core.api.objects import VisualCuboid

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rmp_flow = self._controller._cspace_controller.rmp_flow
        pick_pallet_bmin = None
        for i, pallet_path in enumerate([PICK_PALLET_PATH] + PLACE_PALLET_PATHS):
            pallet_prim = stage.GetPrimAtPath(pallet_path)
            if not pallet_prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
            if i == 0:
                pick_pallet_bmin = bmin.copy()
            bmax = bmax.copy()
            bmax[2] = max(bmin[2], bmax[2] - PALLET_OBSTACLE_TOP_MARGIN_M)
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            obstacle_path = f"/World/_rmpflow_pallet_obstacle_{self.robot_id}_{i}"
            if not stage.GetPrimAtPath(obstacle_path).IsValid():
                obstacle = VisualCuboid(prim_path=obstacle_path, position=center, scale=size, visible=False)
            else:
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)
            rmp_flow.add_cuboid(obstacle, static=True)
            print(f"[amr_pick_place] {self.robot_id}: registered pallet obstacle {obstacle_path} center={center} size={size}", flush=True)

        for j, extra_path in enumerate(EXTRA_STATIC_OBSTACLE_PATHS):
            extra_prim = stage.GetPrimAtPath(extra_path)
            if not extra_prim.IsValid():
                continue
            rng = bbox_cache.ComputeWorldBound(extra_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())
            bmax = bmax.copy()
            # 컨베이어(convey_01)의 원본 bbox가 실제로 pick pallet의 bbox와
            # X방향으로 0.7m 가까이 겹친다(2026-07-26 확인) - 그대로 등록하면
            # pick 목표 지점 자체가 이 장애물 안에 들어가버려서 RMPflow가
            # 영원히 목표에 도달 못 하고 헤맨다(헤드리스로 확인 - 관절 속도
            # 계속 치솟고 반복 재시도). pallet 쪽으로는 pick pallet의 가장자리
            # 못 미쳐서 잘라내(clip) 목표 지점을 침범하지 않게 한다.
            if pick_pallet_bmin is not None:
                bmax[0] = min(bmax[0], pick_pallet_bmin[0] - 0.02)
            if bmax[0] <= bmin[0]:
                continue
            center = (bmin + bmax) / 2.0
            size = bmax - bmin
            obstacle_path = f"/World/_rmpflow_extra_obstacle_{self.robot_id}_{j}"
            if not stage.GetPrimAtPath(obstacle_path).IsValid():
                obstacle = VisualCuboid(prim_path=obstacle_path, position=center, scale=size, visible=False)
            else:
                obstacle = VisualCuboid(prim_path=obstacle_path)
                obstacle.set_world_pose(position=center)
            rmp_flow.add_cuboid(obstacle, static=True)
            print(f"[amr_pick_place] {self.robot_id}: registered extra obstacle {obstacle_path} center={center} size={size}", flush=True)

    def _pallet_yaw(self, pallet_path: str) -> float:
        """pallet의 world-frame yaw(라디안, Z축 기준)를 계산한다.

        place pallet들(pallet_01/02/03)은 축 정렬(yaw=0)이지만, pick
        pallet(/World/pallet)은 실제로 45도 회전되어 있다(2026-07-26 확인,
        check_pallet_yaw2.py로 world_x_axis 직접 측정). 박스를 world 축에
        맞춰(회전 없이) 스폰하면 실제로는 마름모꼴로 놓인 pallet 바닥
        모서리 밖으로 박스 모서리가 삐져나가 AMR 쪽으로 떨어지는 문제가
        있었다(사용자 확인) - 스폰 방향을 이 yaw만큼 맞춰 돌려야 한다.
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        mat = UsdGeom.Xformable(pallet_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x_axis = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        return float(np.arctan2(x_axis[1], x_axis[0]))

    def _nearest_point_on_pallet_top(self, pallet_path: str) -> np.ndarray:
        """pallet의 윗면에서 AMR(팔 기준점)과 가장 가까운 지점을 계산한다.

        fleet_config 노드는 pallet "중심"이 아니라 통로 쪽 정차 위치라, pallet
        중심을 그대로 목표로 쓰면 pallet 크기(약 1.3m)의 절반만큼 팔 리치
        밖으로 벗어난다(2026-07-26 확인). 대신 pallet의 실제 bbox 안에서 AMR과
        가장 가까운 XY 지점(= AMR이 서 있는 쪽 가장자리)을 목표로 삼는다.

        가장자리에 딱 붙이면(margin=0) 그 지점이 팔 자신의 정지 자세
        범위(shoulder_link/upper_arm_link 등)와 거의 겹쳐서, 박스가 스폰되자마자
        팔과 충돌해 튕겨나가는 문제가 있었다(GUI에서 확인된 회귀) - 그래서
        가장자리에서 PALLET_EDGE_MARGIN_M만큼 안쪽으로 들여서 스폰/목표 지점을
        잡는다. 이 마진에는 박스 자신의 반너비도 더한다 - 마진을 중심점
        기준으로만 계산하면 박스 자체가 커서 몸통 절반이 pallet 바닥 밖으로
        걸쳐 나가 AMR 쪽으로 떨어지는 문제가 있었다(GUI에서 확인된 회귀).
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())

        reference_xy, _ = self._read_base_pose()
        margin = PALLET_EDGE_MARGIN_M + self._box_size[:2] / 2.0
        margin = np.minimum(margin, (bmax[:2] - bmin[:2]) / 2.0 - 0.01)
        nearest_xy = np.clip(reference_xy[:2], bmin[:2] + margin, bmax[:2] - margin)
        target_z = bmax[2] + self._box_size[2] / 2.0
        return np.array([nearest_xy[0], nearest_xy[1], target_z])

    def _compute_safe_hover_height(self, pallet_path: str) -> float:
        """pallet 벽 꼭대기보다 확실히 높은 절대 높이를 계산한다.

        base PickPlaceController의 event 0("move above pick")은
        end_effector_initial_height를 목표보다 얼마나 "더 높이"가 아니라
        절대 높이(self._h1)로 쓴다 - 지면 위 데모(z~0.065)에서 튜닝된
        0.55m는 pallet처럼 목표 자체가 1.5m 이상인 경우 오히려 목표보다
        훨씬 낮은 높이라서, "위로 이동"이 실제로는 옆 벽 높이 아래로
        내려갔다가 다시 오르내리며 벽에 스치는 원인이었다(GUI에서 확인된
        회귀). 매 반복마다 실제 목표(pallet)에 맞는 절대 높이를 다시
        계산해서 self._controller.reset(end_effector_initial_height=...)로
        넘겨줘야 한다.
        """
        stage = omni.usd.get_context().get_stage()
        pallet_prim = stage.GetPrimAtPath(pallet_path)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bbox_cache.ComputeWorldBound(pallet_prim).ComputeAlignedRange()
        wall_top_z = rng.GetMax()[2]
        return float(wall_top_z + SAFE_HOVER_MARGIN_M)

    def _compute_slot_target(self, slot_index: int) -> np.ndarray:
        """슬롯의 chassis-local 오프셋을 실제 world 좌표로 바꾼다.

        예전엔 Gf.Matrix3d(Gf.Rotation(quat)) * Gf.Vec3d(offset)로 직접
        회전시켰는데, Gf의 행렬*벡터 곱 관례상 이게 실제로는 반대 방향
        회전(R 대신 사실상 R^T)을 적용해서, 계산된 슬롯 목표가 실제
        StorageBox(섀시 뒤, chassis yaw=45도 기준 world Y~-1.44) 대신 정반대
        방향인 pick pallet/컨베이어 쪽(world Y~-0.6)으로 나왔다(2026-07-26,
        check_slot_targets.py로 실제 StorageBox bbox와 비교해서 확인 - "이상한
        곳(컨베이어 방향)으로 간다"는 사용자 리포트의 원인). ComputeLocalToWorldTransform
        (다른 곳의 _pallet_yaw/_nearest_point_on_pallet_top과 동일한 방식)으로
        섀시의 실제 전체 변환을 직접 써서 이 실수를 피한다.
        """
        stage = omni.usd.get_context().get_stage()
        chassis_prim = stage.GetPrimAtPath(self._chassis_path)
        mat = UsdGeom.Xformable(chassis_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_point = mat.Transform(Gf.Vec3d(*STORAGE_SLOT_LOCAL_POSITIONS[slot_index]))
        return np.array(world_point)

    # -----------------------------------------------------------------
    # 내부 helper - 픽업 박스 스폰
    # -----------------------------------------------------------------

    def _clamp_action(self, action, current_joints: np.ndarray):
        """RMPflow가 이번 스텝에 요청한 관절 목표가 현재 위치에서 너무 크게
        튀지 않도록 델타를 MAX_JOINT_STEP_RAD로 제한한다(위 상수 설명 참고 -
        pallet처럼 먼 목표에서 RMPflow가 가끔 불안정해지는 것에 대한 안전판)."""
        if action.joint_positions is not None and action.joint_indices is not None:
            idx = np.array(action.joint_indices)
            current_at_idx = np.array(current_joints)[idx]
            target = np.array(action.joint_positions, dtype=float)
            delta = np.clip(target - current_at_idx, -MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD)
            action.joint_positions = current_at_idx + delta
        return action

    def _spawn_pick_box(self) -> str:
        from isaacsim.core.api.objects import DynamicCuboid

        prim_path = f"/World/_pick_box_{self.robot_id}_{self._next_box_serial}"
        self._next_box_serial += 1
        spawn_pos = self._nearest_point_on_pallet_top(PICK_PALLET_PATH)
        # pallet 바닥의 실제 회전(yaw)에 박스도 맞춰 돌려서 스폰한다 - 그래야
        # world 축 기준으로는 넓어 보이는 마름모꼴 바닥 안에 박스가 실제로도
        # 딱 들어간다(위 _pallet_yaw 설명 참고).
        yaw = self._pallet_yaw(PICK_PALLET_PATH)
        orientation = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        DynamicCuboid(
            prim_path=prim_path,
            name=prim_path.replace("/", "_"),
            position=spawn_pos,
            orientation=orientation,
            scale=self._box_size,
            color=np.array([0.0, 0.0, 1.0]),
            mass=1.1,
        )
        return prim_path

    # -----------------------------------------------------------------
    # 내부 helper - StorageBox 용접(gripper와 별개, 슬롯<->박스 고정)
    # -----------------------------------------------------------------

    def _weld_joint_path(self, slot_index: int) -> str:
        return f"{self._storage_box_path}/StorageWeldJoint_{slot_index}"

    def _weld_box_to_slot(self, slot_index: int, box_prim_path: str) -> None:
        stage = omni.usd.get_context().get_stage()
        joint_path = self._weld_joint_path(slot_index)
        if stage.GetPrimAtPath(joint_path).IsValid():
            return
        storage_prim = stage.GetPrimAtPath(self._storage_box_path)
        box_prim = stage.GetPrimAtPath(box_prim_path)
        if not storage_prim.IsValid() or not box_prim.IsValid():
            return
        storage_mat = UsdGeom.Xformable(storage_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        box_mat = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel_local = storage_mat.GetInverse().Transform(box_mat.ExtractTranslation())
        rel_rot = storage_mat.ExtractRotationQuat().GetInverse() * box_mat.ExtractRotationQuat()

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._storage_box_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(box_prim_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_local))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_rot))
        joint.GetPrim().GetAttribute("physics:excludeFromArticulation").Set(True)

    def _unweld_slot(self, slot_index: int) -> None:
        stage = omni.usd.get_context().get_stage()
        joint_path = self._weld_joint_path(slot_index)
        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)

    # -----------------------------------------------------------------
    # 외부 API
    # -----------------------------------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def occupied_slot_count(self) -> int:
        return len(self._slot_fill_order)

    def start_pick(self, count: int = 1) -> bool:
        """idle일 때만 pick(픽업 -> 빈 슬롯) 시작. count번 내부적으로
        반복한다(슬롯이 모자라면 들어가는 만큼만)."""
        if self._phase != "idle" or count < 1:
            return False
        free_slots = sum(1 for p in self._slot_box_paths if p is None)
        if free_slots < 1:
            print(f"[amr_pick_place] {self.robot_id}: 보관함에 빈 슬롯이 없음 - pick 시작 불가", flush=True)
            return False
        if free_slots < count:
            print(
                f"[amr_pick_place] {self.robot_id}: 빈 슬롯 부족(요청 {count}, 여유 {free_slots}) - "
                f"{free_slots}개만 픽업합니다",
                flush=True,
            )
            count = free_slots
        self._pick_remaining = count
        self._begin_next_pick_repetition()
        return True

    def _begin_next_pick_repetition(self) -> None:
        self._refresh_base_pose_and_obstacle()
        self._current_box_prim_path = self._spawn_pick_box()
        self._current_slot_index = next(i for i, p in enumerate(self._slot_box_paths) if p is None)
        # NOTE: PickPlaceController.reset()이 내부적으로 RmpFlow.reset()도
        # 호출하는데, 이게 add_cuboid로 등록해둔 장애물(pallet 포함)을 전부
        # 지워버린다(헤드리스로 확인 - __init__에서 한 번만 등록해뒀더니 첫
        # reset()에서 바로 사라짐). reset() 뒤엔 항상 다시 등록해야 한다.
        # end_effector_initial_height도 매번 pallet 벽 높이에 맞게 다시
        # 계산해서 넘긴다(위 _compute_safe_hover_height 설명 참고).
        self._controller.reset(end_effector_initial_height=self._compute_safe_hover_height(PICK_PALLET_PATH))
        self._register_pallet_obstacles()
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()
        self._contact_seen = False
        self._grasp_confirmed = False
        self._welded_this_rep = False
        self._cycle_target = self._compute_slot_target(self._current_slot_index)
        self._phase = "pick_to_storage"

    def start_place(self, shelf_num: int) -> bool:
        """비어있지 않은 가장 오래된(FIFO) 슬롯 하나를 꺼내 place(슬롯 ->
        shelf_num에 해당하는 pallet) 시작. shelf_num은 fms_node.py의
        SHELF_INDEX와 동일한 규약(0=280/pallet_01, 1=260/pallet_02,
        2=240/pallet_03). idle이 아니거나 채워진 슬롯이 없거나 shelf_num이
        범위 밖이면 False."""
        if self._phase != "idle" or not self._slot_fill_order:
            return False
        if not (0 <= shelf_num < len(PLACE_PALLET_PATHS)):
            print(f"[amr_pick_place] {self.robot_id}: 알 수 없는 shelf_num={shelf_num}", flush=True)
            return False
        slot_index = self._slot_fill_order[0]
        box_prim_path = self._slot_box_paths[slot_index]
        if box_prim_path is None:
            return False
        self._refresh_base_pose_and_obstacle()
        self._unweld_slot(slot_index)
        self._current_slot_index = slot_index
        self._current_box_prim_path = box_prim_path
        # NOTE: PickPlaceController.reset()이 내부적으로 RmpFlow.reset()도
        # 호출하는데, 이게 add_cuboid로 등록해둔 장애물(pallet 포함)을 전부
        # 지워버린다(헤드리스로 확인 - __init__에서 한 번만 등록해뒀더니 첫
        # reset()에서 바로 사라짐). reset() 뒤엔 항상 다시 등록해야 한다.
        # end_effector_initial_height도 이번에 놓을 pallet 벽 높이에 맞게
        # 다시 계산해서 넘긴다(위 _compute_safe_hover_height 설명 참고).
        target_pallet_path = PLACE_PALLET_PATHS[shelf_num]
        self._controller.reset(end_effector_initial_height=self._compute_safe_hover_height(target_pallet_path))
        self._register_pallet_obstacles()
        self._gripper.set_default_state(opened=True)
        self._gripper.post_reset()
        self._contact_seen = False
        self._grasp_confirmed = False
        self._cycle_target = self._nearest_point_on_pallet_top(target_pallet_path)
        self._phase = "place_from_storage"
        return True

    def update(self) -> Optional[str]:
        """매 프레임 호출. "pick_done"/"place_done"/"pick_failed" 이벤트가 막
        발생했으면 그 문자열을 반환하고, 아니면 None을 반환한다. count>1인
        pick은 내부적으로 여러 번 반복하고 전부 끝났을 때만 "pick_done"을
        반환한다(중간 반복은 None)."""
        if self._phase == "idle":
            return None

        if self._phase == "cooldown":
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._begin_next_pick_repetition()
            return None

        if self._phase == "returning_home":
            return self._step_return_home()

        current_joints = self._robot.get_joint_positions()
        source_pos = self._read_box_pose(self._current_box_prim_path)

        # _is_contact_now()는 contact_target_filter를 적용해서, 우리가 스폰한
        # 박스가 아니라 pallet 벽면 같은 주변 지오메트리에 스치는 건 접촉으로
        # 안 친다(위 클래스 docstring과 trigger_surface_gripper.py 참고).
        overlap_now = self._controller._is_contact_now()
        if not self._contact_seen and overlap_now:
            self._contact_seen = True
        pick_offset_z = 0.0 if self._contact_seen else PICK_DESCEND_OFFSET_Z
        if self._gripper.is_closed():
            self._grasp_confirmed = True

        actions = self._controller.forward(
            picking_position=source_pos,
            placing_position=self._cycle_target,
            current_joint_positions=current_joints,
            end_effector_offset=np.array([0.0, 0.0, pick_offset_z]),
            end_effector_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self._robot.apply_action(self._clamp_action(actions, current_joints))

        # 박스를 슬롯에 용접하는 건 사이클이 다 끝난 뒤(is_done(), 팔이
        # 다시 들어올려 복귀까지 마친 시점)가 아니라, 그리퍼가 실제로 놓는
        # "바로 그 순간"(event 7)에 해야 한다 - 늦게 용접하면 그 사이(event
        # 8~9, 팔이 올라갔다 돌아오는 동안) 박스가 아무 지지 없이 중력으로
        # 바닥까지 떨어져버린 뒤에야 용접하는 꼴이 된다(헤드리스로 확인한
        # 회귀 - 슬롯 목표(z~0.44)가 아니라 항상 바닥(z~0.065)에 놓였었음).
        if (
            self._phase == "pick_to_storage"
            and self._grasp_confirmed
            and not self._welded_this_rep
            and not self._gripper.is_closed()
        ):
            self._weld_box_to_slot(self._current_slot_index, self._current_box_prim_path)
            self._welded_this_rep = True

        if not self._controller.is_done():
            return None

        if self._phase == "pick_to_storage":
            if self._grasp_confirmed:
                self._slot_box_paths[self._current_slot_index] = self._current_box_prim_path
                self._slot_fill_order.append(self._current_slot_index)
                self._retry_count = 0
                self._pick_remaining -= 1
                self._after_return_home = "next_rep" if self._pick_remaining > 0 else (
                    "pick_done" if self._slot_fill_order else "pick_failed"
                )
            else:
                self._retry_count += 1
                if self._retry_count <= MAX_PICK_RETRIES:
                    # "픽업은 무조건 성공해야 한다" - 실패한 슬롯은 건너뛰지
                    # 않고 같은 자리에서 재시도한다(_begin_next_pick_repetition이
                    # 매번 "아직 안 채워진" 가장 앞 슬롯을 다시 고르므로,
                    # pick_remaining을 그대로 두면 자동으로 이 슬롯을 다시
                    # 시도하게 된다).
                    print(
                        f"[amr_pick_place] {self.robot_id}: pick 흡착 실패 - "
                        f"재시도 {self._retry_count}/{MAX_PICK_RETRIES}",
                        flush=True,
                    )
                    self._after_return_home = "next_rep"
                else:
                    print(
                        f"[amr_pick_place] {self.robot_id}: pick 흡착 실패 - "
                        f"재시도 {MAX_PICK_RETRIES}회 모두 실패, 박스 1개 건너뜀",
                        flush=True,
                    )
                    self._retry_count = 0
                    self._pick_remaining -= 1
                    self._after_return_home = "next_rep" if self._pick_remaining > 0 else (
                        "pick_done" if self._slot_fill_order else "pick_failed"
                    )
            # 다음에 뭘 할지(반복/재시도를 더 할지, 이번 update()가 이벤트를
            # 반환할지)와 무관하게, RMPflow에 다음 목표를 맡기기 전엔 항상
            # 먼저 관절 공간 홈 자세로 확실히 복귀시킨다 - 그래야 IK가 매번
            # 같은 안전한 자세에서 시작해서 특이점 근처에서 불안정해지는 걸
            # 막을 수 있다(위 SPAWN_JOINT_ANGLES_DEG 설명 참고).
            self._start_return_home()
            return None

        if self._phase == "place_from_storage":
            self._slot_box_paths[self._current_slot_index] = None
            self._slot_fill_order.remove(self._current_slot_index)
            self._after_return_home = "place_done"
            self._start_return_home()
            return None

        return None

    def _start_return_home(self) -> None:
        self._phase = "returning_home"
        self._return_home_steps_left = RETURN_HOME_STEPS

    def _step_return_home(self) -> Optional[str]:
        self._robot.apply_action(
            ArticulationAction(joint_positions=self._arm_home_positions, joint_indices=self._arm_joint_indices)
        )
        self._return_home_steps_left -= 1
        current = np.array(self._robot.get_joint_positions())[self._arm_joint_indices]
        settled = float(np.max(np.abs(current - self._arm_home_positions))) < RETURN_HOME_TOLERANCE_RAD
        if not settled and self._return_home_steps_left > 0:
            return None

        after = self._after_return_home
        self._after_return_home = None
        if after == "next_rep":
            # 다음 반복을 바로 시작하지 않고 잠깐 더 대기한다 - 방금 홈
            # 복귀하며 생긴 충격이 가라앉기 전에 섀시 pose를 읽으면 목표가
            # 잘못 계산될 수 있다(위 COOLDOWN_STEPS 설명 참고).
            self._phase = "cooldown"
            self._cooldown_remaining = COOLDOWN_STEPS
            return None
        self._phase = "idle"
        return after
