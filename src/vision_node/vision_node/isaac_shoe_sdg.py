"""
Isaac Sim Replicator - 신발 정상/훼손 2클래스 합성 데이터 생성 (1차 단순 버전)

지금 단계 목표: 색상/사이즈 클래스 다 빼고, 빨간 신발 하나로
정상(normal) / 훼손(defect) 두 클래스만 랜덤 생성해서 바로 YOLO-seg 학습까지 붙이는 것.
->  이 스크립트로 데이터 생성  ->  convert_sdg_to_yolo.py 로 YOLO-seg 포맷 변환
    ->  train_yolo_seg.py 로 학습

카메라는 일단 1대(cam1)만 사용. 파이프라인 검증되면 cam2/cam3 다시 붙이면 됨
(지금은 변수를 줄이는 게 우선).

실행 (GPU 서버, headless):
    ./python.sh isaac_shoe_sdg.py --headless --num_frames 300
    # 먼저 --num_frames 3 정도로 짧게 돌려서 출력 구조/파일명부터 확인 추천

⚠ TODO (자리표시자 - 값 채워야 함):
    - SHOE_ASSET_URL: 실제 빨간 신발 USD 경로
    - CAMERAS["cam1"]: 실제 카메라 위치/각도
    - PLACEMENT_AREA_MIN/MAX: 실제 컨베이어 배치 영역 크기
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_frames", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import os
import random

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

random.seed(args.seed)
rep.set_global_seed(args.seed)

# ------------------------------------------------------------------------
# CONFIG (placeholder - 실제 값 확정되면 교체)
# ------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.getcwd(), "_out_shoe_sdg")
RESOLUTION = (1440, 812)

PLACEMENT_AREA_MIN = (-0.25, -0.15)
PLACEMENT_AREA_MAX = (0.25, 0.15)

# TODO: 실제 빨간 신발 USD 경로로 교체
SHOE_ASSET_URL = "/World/Assets/shoes/sneaker_red.usd"

NUM_SHOES_PER_FRAME = 6      # 한 프레임에 흩뿌릴 신발 개수
DEFECT_RATIO = 0.5           # 훼손 비율 (클래스 균형용, 필요시 조정)
CLASSES = ["normal", "defect"]

# TODO: 실제 카메라 위치/각도로 교체
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
    # 정상: 깨끗한 빨강 / 훼손: 어둡고 거친 오염 표면 (극단값 위주 라벨링 - vision_design_summary 3.4 방향과 동일)
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), (0.65, 0.05, 0.05), 0.35)
    defect_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeDefect"), (0.18, 0.14, 0.11), 0.9)
    return {"normal": normal_mat, "defect": defect_mat}


def bind_material_recursive(prim, material):
    """부모 Xform에 바인딩하면 하위 mesh의 기존 바인딩에 안 먹힐 수 있어서
    실제 mesh 프림들을 순회하며 직접 바인딩한다."""
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
    # add_labels는 이전 'class' 라벨을 덮어쓰므로 매 프레임 재호출해도 안전함
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

    for i in range(args.num_frames):
        for prim in shoe_prims:
            randomize_shoe(prim, materials)

        print(f"[SDG] Capturing frame {i + 1}/{args.num_frames}")
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)

    writer.detach()
    for rp in render_products:
        rp.destroy()
    rep.orchestrator.wait_until_complete()
    timeline.stop()
    print("[SDG] Done.")


run()

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()