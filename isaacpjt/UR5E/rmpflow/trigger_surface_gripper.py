from typing import Callable, List, Optional, Union

import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class TriggerSurfaceGripper(Gripper):
    """트리거 기반 흡착 그리퍼.

    isaacsim.robot.surface_gripper 플러그인(robot_schema 기반, raycast로
    attachment point를 찾는 방식) 대신, GripperBase 밑에 미리 만들어둔
    트리거 prim(들)의 겹침(overlap)을 직접 조회해서 접촉 여부를 판정하고,
    닿아있는 물체를 그 자리에서 UsdPhysics.FixedJoint로 용접한다.

    trigger_paths에는 별도의 전용 ContactTrigger 하나만 줄 수도 있고,
    (이제 Cone 자체도 PhysxTriggerAPI가 적용돼 있으므로) 여러 개의 Cone
    prim 경로를 리스트로 줘서 그 합집합(OR)으로 판정할 수도 있다 - 여러
    prim 중 어느 하나라도 겹치면 접촉으로 본다.

    - self._target_prim_path를 미리 알 필요 없음 - 트리거에 걸린 물체를
      그 순간 자동으로 타겟으로 삼는다.
    - horiz_dist/vert_gap 같은 수동 거리 계산이 필요 없음 - 트리거 겹침
      자체가 "충분히 가까이 닿았다"는 판정이라 콘 사이 빈틈에 걸쳐 있어도
      감지된다.
    """

    def __init__(
        self,
        end_effector_prim_path: str,
        trigger_paths: Union[str, List[str]],
        joint_path: str,
        tip_local_offset: Gf.Vec3d = Gf.Vec3d(0.0, 0.0, 0.0),
        target_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._gripper_body_path = end_effector_prim_path
        self._trigger_paths = [trigger_paths] if isinstance(trigger_paths, str) else list(trigger_paths)
        self._joint_path = joint_path
        self._tip_local_offset = tip_local_offset
        # trigger_paths(콘)가 실제 픽업 대상(예: 박스) 말고도 주변 씬 지오메트리
        # (예: pallet 벽면 메시)와도 겹칠 수 있다 - 그 경우 target_filter 없이
        # "겹친 것 중 아무거나 첫 번째"를 잡으면 pallet 자체를 흡착해버리는
        # 사고가 난다(실제로 GUI에서 발생 확인 - 사실상 고정된 구조물을
        # 들어올리려다 팔이 격렬하게 불안정해짐). target_filter(prim_path)가
        # True를 반환하는 후보만 흡착 대상으로 인정한다.
        self._target_filter = target_filter
        self._attached = False
        self._target_prim_path: Optional[str] = None
        self._articulation_num_dofs: Optional[int] = None

    def initialize(self, physics_sim_view=None, articulation_num_dofs: Optional[int] = None) -> None:
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._articulation_num_dofs = articulation_num_dofs
        if self._default_state is None:
            self._default_state = not self.is_closed()

    def close(self) -> None:
        """트리거 발바닥에 뭔가 닿아있으면 그 물체를 자동으로 찾아서 용접한다."""
        if self._attached:
            return
        stage = omni.usd.get_context().get_stage()
        overlapping = []
        for trigger_path in self._trigger_paths:
            trigger_prim = stage.GetPrimAtPath(trigger_path)
            if not trigger_prim.IsValid():
                continue
            trigger_state = PhysxSchema.PhysxTriggerStateAPI(trigger_prim)
            overlapping.extend(trigger_state.GetTriggeredCollisionsRel().GetTargets())
        if not overlapping:
            return  # 발바닥에 닿은 물체 없음

        target_path = None
        for candidate in overlapping:
            candidate_str = str(candidate)
            if self._target_filter is not None and not self._target_filter(candidate_str):
                continue
            if stage.GetPrimAtPath(candidate_str).IsValid():
                target_path = candidate_str
                break
        if target_path is None:
            return  # 겹친 것 중 실제로 흡착해도 되는 대상이 없음
        target_prim = stage.GetPrimAtPath(target_path)

        gripper_mat = UsdGeom.Xformable(
            stage.GetPrimAtPath(self._gripper_body_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        target_mat = UsdGeom.Xformable(target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        tip_world = gripper_mat.Transform(self._tip_local_offset)
        # The tolerant trigger volume (needed for reliable detection) means
        # the object's ACTUAL position at the exact instant of detection can
        # be a bit above or below true flush contact - welding that raw
        # position bakes in a visible gap or interpenetration that (now that
        # Plate/Cone_* are trigger-only, not real colliders) nothing ever
        # corrects afterward. Snap the captured contact point's world Z to
        # the target's own known top face instead, so the result is always
        # exactly flush regardless of when within the trigger's tolerance
        # band the grab happened. Only applies to UsdGeom.Cube targets
        # (our known use case: boxes approached from directly above); falls
        # back to the raw captured pose for anything else.
        target_cube = UsdGeom.Cube(target_prim)
        if target_cube:
            size = target_cube.GetSizeAttr().Get()
            scale_z = 1.0
            for op in UsdGeom.Xformable(target_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_z = op.Get()[2]
                    break
            half_height = (size if size is not None else 1.0) * scale_z / 2.0
            target_top_z = target_mat.ExtractTranslation()[2] + half_height
            tip_world = Gf.Vec3d(tip_world[0], tip_world[1], target_top_z)

        rel_local = target_mat.GetInverse().Transform(tip_world)
        # LocalPos0(GripperBase 쪽 조인트 프레임)를 고정된 self._tip_local_offset
        # 그대로 쓰면, 위에서 tip_world의 Z를 박스 표면으로 스냅한 만큼
        # LocalPos0/LocalPos1이 서로 다른 실제 지점을 가리키게 된다 - 이 두
        # 프레임의 world 위치가 어긋난 채로 조인트가 생성되면 PhysX가 다음
        # 스텝에 그 간격만큼 강제로 스냅해서 GripperBase(따라서 팔 전체,
        # 섀시까지)에 충격을 준다("found a joint with disjointed body
        # transforms" 경고 - 2026-07-27, 사용자가 흡착 시 로봇 전체가 흔들리는
        # 것으로 확인, 박스 속도는 그 시점에 이미 0에 가까워 타이밍 문제가
        # 아니라 순수 기하학적 불일치였음을 실측으로 확인). LocalPos0도 스냅된
        # tip_world를 GripperBase 로컬 좌표로 역변환해서 두 프레임이 항상
        # 같은 지점을 가리키게 한다.
        local_pos0 = gripper_mat.GetInverse().Transform(tip_world)
        gripper_rot = gripper_mat.ExtractRotationQuat()
        target_rot = target_mat.ExtractRotationQuat()
        local_rot1 = target_rot.GetInverse() * gripper_rot

        joint = UsdPhysics.FixedJoint.Define(stage, self._joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._gripper_body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(target_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local_pos0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(rel_local))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(local_rot1))
        joint.GetPrim().GetAttribute("physics:excludeFromArticulation").Set(True)

        self._target_prim_path = str(target_path)
        self._attached = True
        print(f"  [흡착] target={target_path} -> {self._joint_path} 생성", flush=True)

    def open(self) -> None:
        """붙어있던 joint를 제거하고 놓는다."""
        if not self._attached:
            return
        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(self._joint_path)
        if joint_prim.IsValid():
            stage.RemovePrim(self._joint_path)
        self._attached = False
        self._target_prim_path = None

    def is_closed(self) -> bool:
        return self._attached

    def is_open(self) -> bool:
        return not self._attached

    def set_default_state(self, opened: bool) -> None:
        self._default_state = opened

    def get_default_state(self) -> dict:
        return {"opened": self._default_state}

    def post_reset(self) -> None:
        Gripper.post_reset(self)
        if self._default_state:
            self.open()
        else:
            self.close()

    def forward(self, action: str) -> ArticulationAction:
        if self._articulation_num_dofs is None:
            raise Exception(
                "Num of dofs of the articulation needs to be passed to initialize in order to use this method"
            )
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        else:
            raise Exception(f"action {action} is not defined for TriggerSurfaceGripper")
        return ArticulationAction(joint_positions=[None] * self._articulation_num_dofs)
