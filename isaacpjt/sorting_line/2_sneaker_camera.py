
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import numpy as np
import omni.usd
import omni.graph.core as og
import usdrt.Sdf
from pxr import UsdGeom, UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000.0)

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 신발 스폰                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
SNEAKER_USD = "/home/rokey/Downloads/sneakers.usd"
SNEAKER_POS = Gf.Vec3d(0.0, 0.0, 0.0)

add_reference_to_stage(usd_path=SNEAKER_USD, prim_path="/World/Sneaker")
sneaker_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Sneaker"))
sneaker_xform.ClearXformOpOrder()
sneaker_xform.AddTranslateOp().Set(SNEAKER_POS)

world.scene.add_default_ground_plane()

# ╔══════════════════════════════════════════════════════════════╗
# ║  B. RealSense D455 흉내 카메라 — 신발 위에서 아래를 내려다봄           ║
# ╚══════════════════════════════════════════════════════════════╝
CAMERA_PATH = "/World/D455_Camera"
CAMERA_HEIGHT = 0.6  # 신발 위 0.6m

camera_prim = UsdGeom.Camera(stage.DefinePrim(CAMERA_PATH, "Camera"))
xform_api = UsdGeom.XformCommonAPI(camera_prim)
xform_api.SetTranslate(Gf.Vec3d(SNEAKER_POS[0], SNEAKER_POS[1], CAMERA_HEIGHT))
xform_api.SetRotate((0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # 카메라 기본(무회전) 방향이 이미 로컬 -Z = 아래쪽

# RealSense D455 RGB 스트림 근사치 (수평 FOV ~87도, 1280x720)
camera_prim.GetHorizontalApertureAttr().Set(20.955)
camera_prim.GetVerticalApertureAttr().Set(11.787)
camera_prim.GetFocalLengthAttr().Set(9.5)
camera_prim.GetFocusDistanceAttr().Set(CAMERA_HEIGHT)
camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5.0))

simulation_app.update()

# ╔══════════════════════════════════════════════════════════════╗
# ║  C. ROS2 카메라 퍼블리시 그래프                                     ║
# ╚══════════════════════════════════════════════════════════════╝
ROS_CAMERA_GRAPH_PATH = "/ROS_D455_Camera"
keys = og.Controller.Keys

(ros_camera_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
            ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("cameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("cameraHelperRgb.inputs:frameId", "d455_camera"),
            ("cameraHelperRgb.inputs:topicName", "/d455/color/image_raw"),
            ("cameraHelperRgb.inputs:type", "rgb"),
            ("cameraHelperInfo.inputs:frameId", "d455_camera"),
            ("cameraHelperInfo.inputs:topicName", "/d455/color/camera_info"),
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_PATH)]),
        ],
    },
)

og.Controller.evaluate_sync(ros_camera_graph)
simulation_app.update()

world.reset()

print("\n" + "=" * 60)
print("[환경] 스니커 스폰 + D455 카메라(하향) + ROS2 퍼블리시 시작")
print("  RGB   : /d455/color/image_raw")
print("  info  : /d455/color/camera_info")
print("=" * 60)

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
