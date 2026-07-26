from pathlib import Path
from typing import Optional

import numpy as np
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """우리 surface gripper(Plate + Cone_0~3)가 붙은 UR5e용 RMPFlow controller.

    Isaac Sim에 내장된 UR5e RMPflow 설정("UR5e"/"RMPflow")은 flange/tool0까지만
    알고 있고 우리가 만든 GripperBase 구조는 전혀 모른다. 그래서 이 폴더의
    ur5e_surface_gripper.urdf / ur5e_surface_gripper_robot_description.yaml /
    ur5e_surface_gripper_rmpflow_config.yaml 세 파일을 직접 준비했다:

    - urdf: wrist_3_link 밑에 gripper_base_link(플레이트/콘 조립 원점, GripperBase의
      실측 pose)와 gripper_tip_link(4개 콘 끝이 만드는 흡착 평면 중심)를 fixed
      joint로 추가.
    - robot_description.yaml: gripper_base_link에 대한 collision sphere 추가
      (self-collision/장애물 회피용).
    - rmpflow_config.yaml: body_collision_controllers에 gripper_base_link 추가.

    end_effector_frame_name 기본값은 "gripper_tip_link" (흡착 지점 기준으로
    타겟을 지정하고 싶을 때). flange 기준으로 제어하고 싶으면 "flange"를 넘기면
    된다.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        urdf_path: Optional[str] = None,
        robot_description_path: Optional[str] = None,
        rmpflow_config_path: Optional[str] = None,
        end_effector_frame_name: str = "gripper_tip_link",
        maximum_substep_size: float = 0.00334,
        robot_base_position: Optional[object] = None,
        robot_base_orientation: Optional[object] = None,
        chassis_obstacle_prim_path: Optional[str] = None,
        chassis_obstacle_exclude_names: tuple = (),
        chassis_obstacle_max_z: Optional[float] = None,
    ) -> None:
        """
        robot_base_position/robot_base_orientation: URDF의 root_link(=UR5e의
        base_link)가 world 상 어디에 있는지 명시적으로 지정. Nova Carter처럼
        UR5e가 다른 articulation(chassis)의 일부로 합쳐진 경우
        robot_articulation.get_world_pose()는 chassis의 pose를 돌려주기 때문에
        (= UR5e base_link 위치와 다름) 반드시 넘겨줘야 한다. 단일 UR5e
        articulation일 때는 생략하면 robot_articulation의 world pose를 그대로
        사용한다.
        """
        base_dir = Path(__file__).resolve().parent
        urdf_path = str(Path(urdf_path) if urdf_path else base_dir / "ur5e_surface_gripper.urdf")
        robot_description_path = str(
            Path(robot_description_path) if robot_description_path
            else base_dir / "ur5e_surface_gripper_robot_description.yaml"
        )
        rmpflow_config_path = str(
            Path(rmpflow_config_path) if rmpflow_config_path
            else base_dir / "ur5e_surface_gripper_rmpflow_config.yaml"
        )

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            urdf_path=urdf_path,
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=maximum_substep_size,
        )

        if chassis_obstacle_prim_path is not None:
            self._add_chassis_obstacle(
                chassis_obstacle_prim_path,
                exclude_names=chassis_obstacle_exclude_names,
                max_z=chassis_obstacle_max_z,
            )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        if robot_base_position is not None and robot_base_orientation is not None:
            self._default_position, self._default_orientation = robot_base_position, robot_base_orientation
        else:
            self._default_position, self._default_orientation = (
                self._articulation_motion_policy._robot_articulation.get_world_pose()
            )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def _add_chassis_obstacle(
        self, chassis_prim_path: str, exclude_names: tuple = (), max_z: Optional[float] = None
    ) -> None:
        """Nova Carter 차체(chassis_link)의 월드 바운딩 박스를 RMPflow에
        static obstacle로 등록해서 팔이 차체를 관통하지 않게 한다.

        urdf 자체엔 mobile base가 없어서(팔만 정의됨) RMPflow가 차체 존재를
        전혀 모르기 때문에, 실제 USD 지오메트리에서 측정한 bbox를 하나의
        투명 VisualCuboid로 만들어 별도로 등록한다.

        exclude_names에 자식 prim 이름을 넘기면 그 자식(들)은 bbox 계산에서
        빼고 나머지 자식들의 bbox만 합친다 - 예를 들어 chassis_link 밑에
        StorageBox(로봇팔이 실제로 손을 뻗어 들어가야 하는 적재 선반)를 붙인
        경우, 그걸 포함한 전체 bbox를 장애물로 등록하면 "그 자리에 도달해야
        하는" 목표와 "그 자리를 피해야 하는" 장애물 회피가 서로 충돌해서
        팔이 불안정하게 움직이다 실제로 부딪히는 문제가 생겼다(헤드리스로
        확인한 회귀 - StorageBox 추가 직후 팔이 접근하는 동안 섀시가 수 미터
        밀려나는 현상으로 나타남).
        """
        import omni.usd
        from pxr import UsdGeom
        from isaacsim.core.api.objects import VisualCuboid

        stage = omni.usd.get_context().get_stage()
        chassis_prim = stage.GetPrimAtPath(chassis_prim_path)
        if not chassis_prim.IsValid():
            return
        bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])

        if exclude_names:
            bmin = None
            bmax = None
            for child in chassis_prim.GetChildren():
                if child.GetName() in exclude_names:
                    continue
                child_rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
                child_min, child_max = np.array(child_rng.GetMin()), np.array(child_rng.GetMax())
                if child_min[0] > child_max[0]:
                    continue  # empty range (no geometry under this child)
                bmin = child_min if bmin is None else np.minimum(bmin, child_min)
                bmax = child_max if bmax is None else np.maximum(bmax, child_max)
            if bmin is None:
                return
        else:
            rng = bbox_cache.ComputeWorldBound(chassis_prim).ComputeAlignedRange()
            bmin, bmax = np.array(rng.GetMin()), np.array(rng.GetMax())

        if max_z is not None:
            # exclude_names로 StorageBox 자체는 빼도, 나머지 형제 prim("visual"
            # 같은 차체 전체를 뭉뚱그린 단일 메시)의 bbox가 그 자리 상공까지
            # 그대로 덮는 경우가 있다 - StorageBox(1.35~1.40m)보다 훨씬 높은
            # 1.5m 근처까지 뻗어있어서, 보관함 위 칸(slot)에 내려놓으려 하면
            # RMPflow가 실제로는 비어있는 그 상공을 장애물로 착각해 밀어내는
            # 바람에 목표보다 한참 위에서 멈춰버렸다(2026-07-26, 사용자 확인
            # "저장소까지 안 내려가짐" - check_chassis_obstacle_box.py로 실측:
            # 장애물 top=1.457m인데 슬롯 목표는 1.467m로 거의 붙어있었음).
            # 호출 측에서 보관함 칸이 실제로 필요로 하는 높이까지만 장애물
            # 윗면을 깎아서 넘기면, 그 이상은 장애물로 취급하지 않는다.
            bmax = bmax.copy()
            bmax[2] = max(bmin[2], min(bmax[2], max_z))

        center = (bmin + bmax) / 2.0
        size = bmax - bmin

        obstacle_path = "/World/_rmpflow_chassis_obstacle"
        if not stage.GetPrimAtPath(obstacle_path).IsValid():
            obstacle = VisualCuboid(
                prim_path=obstacle_path,
                position=center,
                scale=size,
                visible=False,
            )
        else:
            obstacle = VisualCuboid(prim_path=obstacle_path)
        self.rmp_flow.add_cuboid(obstacle, static=True)

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
