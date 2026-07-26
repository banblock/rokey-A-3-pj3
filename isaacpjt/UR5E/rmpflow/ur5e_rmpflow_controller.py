from typing import Optional

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import interface_config_loader


class RMPFlowController(mg.MotionPolicyController):
    """UR5e용 RMPFlow controller.

    Isaac Sim 에셋에 UR5e RMPflow 설정(URDF + robot_description.yaml +
    rmpflow_config.yaml)이 이미 내장되어 있어(policy_map.json의 "UR5e" 항목),
    별도 파일 없이 interface_config_loader로 그대로 불러와 사용한다.
    커스텀 로봇(M0609)과 달리 이 파일들을 프로젝트에 복제하지 않는다.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        urdf_path: Optional[str] = None,
        robot_description_path: Optional[str] = None,
        rmpflow_config_path: Optional[str] = None,
        end_effector_frame_name: Optional[str] = None,
        maximum_substep_size: Optional[float] = None,
    ) -> None:
        rmp_config = interface_config_loader.load_supported_motion_policy_config("UR5e", "RMPflow")

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=robot_description_path or rmp_config["robot_description_path"],
            rmpflow_config_path=rmpflow_config_path or rmp_config["rmpflow_config_path"],
            urdf_path=urdf_path or rmp_config["urdf_path"],
            end_effector_frame_name=end_effector_frame_name or rmp_config["end_effector_frame_name"],
            maximum_substep_size=maximum_substep_size or rmp_config["maximum_substep_size"],
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
