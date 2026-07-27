"""
Isaac Sim Script Editor용 버전 (isaac_shoe_sdg.py 참고).

이미 켜져 있는 Isaac Sim GUI의 Window > Script Editor에 이 파일 내용을
그대로 붙여넣고 실행(Ctrl+Enter)하면 된다. SimulationApp을 새로 띄우지
않고, 이미 실행 중인 kit 인스턴스 위에서 씬 구성/캡처만 수행한다.

실행 전 NUM_FRAMES/SEED 값만 필요하면 바꿔서 재실행.
"""

import os
import random

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

NUM_FRAMES = 10
SEED = 42

random.seed(SEED)
rep.set_global_seed(SEED)

# ------------------------------------------------------------------------
# CONFIG (isaac_shoe_sdg.py와 동일하게 맞춰서 사용)
# ------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.getcwd(), "_out_shoe_sdg")
RESOLUTION = (1440, 812)

PLACEMENT_AREA_MIN = (-0.25, -0.15)
PLACEMENT_AREA_MAX = (0.25, 0.15)

SHOE_ASSET_URL = "/World/Assets/shoes/sneaker_red.usd"

NUM_SHOES_PER_FRAME = 6
DEFECT_RATIO = 0.5
CLASSES = ["normal", "defect"]

CAMERAS = {
    "cam1": {"position": (0.0, -0.35, 0.45), "rotation": (55.0, 0.0, 90.0), "focal_length": 35.0},
}


# ------------------------------------------------------------------------
# 씬 구성
# ------------------------------------------------------------------------

def set_transform(prim, location=None, rotation=None):
    xform = UsdGeom.Xformable(prim)
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            xform.AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*location))
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            xform.AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*rotation))


def create_conveyor_background(stage):
    plane = stage.DefinePrim("/World/Conveyor/Belt", "Cylinder")
    plane.GetAttribute("radius").Set(1.0)
    plane.GetAttribute("height").Set(0.02)
    set_transform(plane, location=(0, 0, -0.01))

    mat_path = Sdf.Path("/World/Looks/BeltMaterial")
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendPath("Shader"))
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15, 0.15, 0.16))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(plane).Bind(material)


def make_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_condition_materials(stage):
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), (0.65, 0.05, 0.05), 0.35)
    defect_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeDefect"), (0.18, 0.14, 0.11), 0.9)
    return {"normal": normal_mat, "defect": defect_mat}


def bind_material_recursive(prim, material):
    for desc in Usd.PrimRange(prim):
        if desc.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(desc).Bind(material)


def spawn_shoes(stage, materials):
    prims = []
    for i in range(NUM_SHOES_PER_FRAME):
        prim_path = f"/World/Shoes/shoe_{i}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(SHOE_ASSET_URL)
        prims.append(prim)
    return prims


def randomize_shoe(prim, materials):
    x = random.uniform(PLACEMENT_AREA_MIN[0], PLACEMENT_AREA_MAX[0])
    y = random.uniform(PLACEMENT_AREA_MIN[1], PLACEMENT_AREA_MAX[1])
    yaw = random.uniform(0, 360)
    set_transform(prim, location=(x, y, 0.0), rotation=(0, 0, yaw))

    label = "defect" if random.random() < DEFECT_RATIO else "normal"
    bind_material_recursive(prim, materials[label])
    add_labels(prim, labels=[label], instance_name="class")


def build_cameras(stage):
    cams = {}
    for name, cfg in CAMERAS.items():
        cam_prim = stage.DefinePrim(f"/World/Cameras/{name}", "Camera")
        cam_prim.GetAttribute("focalLength").Set(cfg["focal_length"])
        set_transform(cam_prim, location=cfg["position"], rotation=cfg["rotation"])
        cams[name] = cam_prim
    return cams


def run():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    dome_light = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(600.0)
    distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
    distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1500.0)
    set_transform(distant_light, rotation=(0, 60, 0))

    create_conveyor_background(stage)
    materials = create_condition_materials(stage)
    shoe_prims = spawn_shoes(stage, materials)
    cameras = build_cameras(stage)

    render_products = [
        rep.create.render_product(cam_prim.GetPath(), RESOLUTION, name=name)
        for name, cam_prim in cameras.items()
    ]

    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Output directory: {OUTPUT_DIR}")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        bounding_box_2d_tight=True,
        instance_segmentation=True,
        colorize_instance_segmentation=False,
    )
    writer.attach(render_products)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    for i in range(NUM_FRAMES):
        for prim in shoe_prims:
            randomize_shoe(prim, materials)

        print(f"[SDG] Capturing frame {i + 1}/{NUM_FRAMES}")
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)

    writer.detach()
    for rp in render_products:
        rp.destroy()
    rep.orchestrator.wait_until_complete()
    timeline.stop()
    print("[SDG] Done.")


run()
