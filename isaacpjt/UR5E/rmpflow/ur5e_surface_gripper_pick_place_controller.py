from typing import Callable, List, Optional, Union

import numpy as np
import omni.usd
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from pxr import PhysxSchema, Usd, UsdGeom

from ur5e_surface_gripper_rmpflow_controller import RMPFlowController

# gripper_tip_link's local offset from GripperBase (see ur5e_surface_gripper.urdf's
# "gripper_base-gripper_tip" joint) - used to compute the end-effector's actual
# achieved world pose directly from USD, without needing a live FK query API.
_GRIPPER_TIP_LOCAL_OFFSET = (0.0, 0.0, -0.0015)


class PickPlaceController(manipulators_controllers.PickPlaceController):
    """UR5e + surface gripper(흡착)용 pick & place controller.

    manipulators_controllers.PickPlaceController는 gripper 인자로
    isaacsim.robot.manipulators.grippers.gripper.Gripper의 서브클래스만 받으면
    되고(forward(action="open"/"close")만 구현하면 됨), ParallelGripper 전용이
    아니다.

    contact_trigger_path를 넘기면 두 가지가 달라진다:

    1. 실제 흡착 컵처럼, phase(event)가 몇 번이든 상관없이 매 스텝 접촉
       여부를 확인해서 닿으면 즉시 gripper.close()를 시도한다 (원래
       PickPlaceController는 event==3일 때만 close를 호출해서, 그 전
       단계에서 스치듯 닿아도 그냥 딱딱한 충돌로 튕겨나갈 뿐이었다).
    2. 하강(phase 1) 도중 접촉이 감지되면, 그 순간 실제로 도달한
       end-effector의 world pose를 그대로 "목표"로 고정해서 계속
       유지시킨다(phase만 다음 단계로 넘기는 게 아니라, RMPflow에 넘어가는
       target 자체를 현재 위치로 덮어씀). end_effector_offset로 표면보다
       살짝 아래를 목표로 잡아둬도(RMPflow가 정확히 수렴하지 못하는 잔여
       오차를 메우기 위한 여유), 실제로 닿는 순간부터는 그 (도달 불가능한)
       목표를 더 이상 쫓지 않기 때문에 물체를 계속 밀어붙이는 현상이
       생기지 않는다.
    3. place 하강(phase 6) 도중에는 ContactTrigger 대신 - 이미 물체가
       그리퍼에 용접돼 있어 트리거는 상시 겹쳐있는 상태라 구분이 안 됨 -
       붙잡고 있는 물체 자체의 바닥면이 지면(z=0)에 닿았는지를 직접
       확인한다. 닿으면 그 즉시 놓기(phase 7)로 바로 넘어가서, pick 때와
       마찬가지로 바닥을 계속 눌러대는 현상을 막는다.
    """

    def __init__(
        self,
        name: str,
        gripper: Gripper,
        robot_articulation: SingleArticulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        urdf_path: Optional[str] = None,
        robot_description_path: Optional[str] = None,
        rmpflow_config_path: Optional[str] = None,
        end_effector_frame_name: str = "gripper_tip_link",
        robot_base_position=None,
        robot_base_orientation=None,
        chassis_obstacle_prim_path: Optional[str] = None,
        chassis_obstacle_exclude_names: tuple = (),
        chassis_obstacle_max_z: Optional[float] = None,
        contact_trigger_path: Optional[Union[str, List[str]]] = None,
        contact_target_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.0025, 1.0, 0.008, 0.08]

        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
                urdf_path=urdf_path,
                robot_description_path=robot_description_path,
                rmpflow_config_path=rmpflow_config_path,
                end_effector_frame_name=end_effector_frame_name,
                robot_base_position=robot_base_position,
                robot_base_orientation=robot_base_orientation,
                chassis_obstacle_prim_path=chassis_obstacle_prim_path,
                chassis_obstacle_exclude_names=chassis_obstacle_exclude_names,
                chassis_obstacle_max_z=chassis_obstacle_max_z,
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
        self._contact_trigger_paths = (
            [contact_trigger_path] if isinstance(contact_trigger_path, str) else contact_trigger_path
        )
        self._gripper_base_path = (
            self._contact_trigger_paths[0].rsplit("/", 1)[0] if self._contact_trigger_paths else None
        )
        self._contact_target_filter = contact_target_filter
        self._trigger_states = None
        self._frozen_target_pos = None
        self._frozen_target_orient = None
        self._frozen_place_pos = None
        self._frozen_place_orient = None
        self._place_cut_short = False

    def _get_trigger_states(self):
        if self._trigger_states is None and self._contact_trigger_paths:
            stage = omni.usd.get_context().get_stage()
            states = []
            for path in self._contact_trigger_paths:
                trigger_prim = stage.GetPrimAtPath(path)
                if trigger_prim.IsValid():
                    states.append(PhysxSchema.PhysxTriggerStateAPI(trigger_prim))
            self._trigger_states = states
        return self._trigger_states or []

    def _is_contact_now(self) -> bool:
        """트리거에 뭐라도 겹쳤는지가 아니라, contact_target_filter가 있으면
        그걸 통과하는(=실제로 흡착해도 되는) 대상이 겹쳤는지를 본다 - 안
        그러면 콘이 픽업 대상이 아닌 주변 지오메트리(예: pallet 벽면)에
        스치기만 해도 "접촉했다"고 오판해서 그 자리에서 얼어붙거나, 심하면
        그 지오메트리 자체를 흡착해버리는 사고가 난다(실제로 GUI에서 발생
        확인)."""
        for state in self._get_trigger_states():
            targets = state.GetTriggeredCollisionsRel().GetTargets()
            if not targets:
                continue
            if self._contact_target_filter is None:
                return True
            if any(self._contact_target_filter(str(t)) for t in targets):
                return True
        return False

    def _get_current_end_effector_pose(self):
        stage = omni.usd.get_context().get_stage()
        gripper_base_prim = stage.GetPrimAtPath(self._gripper_base_path)
        world = UsdGeom.XformCache().GetLocalToWorldTransform(gripper_base_prim)
        tip_pos = np.array(world.Transform(_GRIPPER_TIP_LOCAL_OFFSET))
        quat = world.ExtractRotationQuat()
        tip_orient = np.array([quat.GetReal(), *quat.GetImaginary()])
        return tip_pos, tip_orient

    def _is_attached_object_touching_ground(self, ground_z: float = 0.0, epsilon: float = 0.01) -> bool:
        target_path = getattr(self._gripper, "_target_prim_path", None)
        if not target_path:
            return False
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(target_path)
        if not prim.IsValid():
            return False
        # NOTE: UsdGeom.BBoxCache here gives wrong (doubly-scaled) results for
        # isaacsim's DynamicCuboid helper - it authors the `extent` attribute
        # already pre-multiplied by the object's own xformOp:scale, so
        # BBoxCache's world-bound computation ends up applying that scale a
        # second time. Compute the bottom face directly from the object's own
        # world translate and its Cube size*scale instead.
        world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        center_z = world.ExtractTranslation()[2]
        half_height = 0.5
        cube_geom = UsdGeom.Cube(prim)
        if cube_geom:
            size = cube_geom.GetSizeAttr().Get()
            scale_z = 1.0
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_z = op.Get()[2]
                    break
            half_height = (size if size is not None else 1.0) * scale_z / 2.0
        bottom_z = center_z - half_height
        return bottom_z <= ground_z + epsilon

    def forward(self, *args, **kwargs):
        contact_now = self._is_contact_now()

        # attempt to grab on ANY contact during the approach/descend/settle/
        # close/lift phases (0-4) - matches how a real suction cup behaves
        # (grabs the instant it touches something), instead of only trying
        # during the scripted "close" phase (event==3), by which point the
        # object may have already been bumped away by ordinary rigid-body
        # collision during approach/descend. Extended from <=3 to <=4
        # (2026-07-27, storage retrieval bug): for targets far below the
        # hover height (e.g. the bottom-layer storage slot, much lower than
        # a tall rack pallet's wall - hover height is sized off the *place*
        # pallet, not the pick source), event1's fixed-step descend budget
        # can undershoot and never register contact during 0-3, yet RMPflow's
        # residual convergence lag keeps the arm sinking toward the real
        # target even after the state machine has already moved on to event4
        # (lift) - contact then lands squarely inside event4, which used to
        # be outside the grab window entirely (confirmed via headless
        # per-event logging: contact_seen only flipped true partway through
        # event4, gripper stayed open the rest of the cycle). event4 is still
        # safely before event 7's release, so this doesn't fight that gate's
        # original purpose.
        if contact_now and self._event <= 4 and not self._gripper.is_closed():
            self._gripper.close()

        # the FIRST time contact happens during the descend phase, capture the
        # ACTUAL achieved end-effector pose right now and hold it fixed - not
        # just skip to the next phase (which alone doesn't stop the
        # commanded target from continuing to reference a possibly
        # past-surface value for a stray step or two), but directly override
        # what gets sent to RMPflow going forward.
        if contact_now and self._event == 1 and self._frozen_target_pos is None:
            self._frozen_target_pos, self._frozen_target_orient = self._get_current_end_effector_pose()
            # self._h0 still holds picking_position[2] (the ORIGINAL,
            # offset-biased target height from event 1's interpolation
            # formula) - not where the arm actually stopped. Since the lift
            # phase (event 4) interpolates FROM self._h0, leaving it stale
            # means lift starts by commanding that old (lower, past-surface)
            # height first before rising - i.e. the arm dips back down before
            # lifting. Overwrite it with the actual frozen height so lift
            # starts exactly from where we really are.
            self._h0 = self._frozen_target_pos[2]
            self._event = 2
            self._t = 0.0

        if self._frozen_target_pos is not None and self._event in (2, 3):
            action = self._cspace_controller.forward(
                target_end_effector_position=self._frozen_target_pos,
                target_end_effector_orientation=self._frozen_target_orient,
            )
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
            return action

        # place descend: cut short the instant the held object's own bottom
        # face reaches the ground, instead of continuing to interpolate down
        # toward the (possibly past-floor) place target for the rest of its
        # time budget. Jumping straight to event 7 (open) is safe here since
        # that phase doesn't command any further arm movement on its own.
        # Also capture the actual achieved height here, same reasoning as the
        # pick side: event 8 (rise after place) otherwise interpolates FROM
        # the raw placing_position height (the base class recomputes this
        # fresh each call, so simply overwriting a stored self._h0-like
        # attribute doesn't work here - event 8 is handled explicitly below
        # instead) - not from wherever we actually stopped, so it would dip
        # down again first before rising.
        # 목표 자체가 바닥 근처(실제 지면/랙)일 때만 "바닥에 닿는 순간 컷오프"가
        # 유효하다 - 목표가 높은 곳(예: AMR에 붙은 보관함 선반, z~0.4 이상)이면
        # 팔이 그리로 가는 도중 일시적으로 낮게 스치기만 해도 이 체크가 잘못
        # 발동해서 목표보다 훨씬 낮은 실제 지면에다 놔버리는 회귀가 있었다
        # (헤드리스로 확인 - 보관함에 넣으려던 박스가 계속 바닥에 떨어짐).
        _place_target_for_ground_check = kwargs.get("placing_position")
        if _place_target_for_ground_check is None and len(args) >= 2:
            _place_target_for_ground_check = args[1]
        _target_is_near_ground = (
            _place_target_for_ground_check is None or _place_target_for_ground_check[2] < 0.15
        )
        if (
            self._event == 6
            and not self._place_cut_short
            and _target_is_near_ground
            and self._is_attached_object_touching_ground()
        ):
            self._frozen_place_pos, self._frozen_place_orient = self._get_current_end_effector_pose()
            self._event = 7
            self._t = 0.0
            self._place_cut_short = True

        if self._frozen_place_pos is not None and self._event == 8:
            placing_position = kwargs.get("placing_position")
            if placing_position is None and len(args) >= 2:
                placing_position = args[1]
            end_effector_orientation = kwargs.get("end_effector_orientation")
            alpha = self._mix_sin(self._t)
            blended_z = self._combine_convex(self._frozen_place_pos[2], self._h1, alpha)
            target_pos = np.array([placing_position[0], placing_position[1], blended_z])
            action = self._cspace_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=(
                    end_effector_orientation if end_effector_orientation is not None else self._frozen_place_orient
                ),
            )
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
            return action

        return super().forward(*args, **kwargs)

    def reset(self, *args, **kwargs) -> None:
        super().reset(*args, **kwargs)
        self._frozen_target_pos = None
        self._frozen_target_orient = None
        self._frozen_place_pos = None
        self._frozen_place_orient = None
        self._place_cut_short = False
