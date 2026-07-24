"""
Isaac Sim Replicator - 신발 정상/훼손 2클래스 합성 데이터 생성 (1차 단순 버전)

지금 단계 목표: 색상/사이즈 클래스 다 빼고, 빨간 신발 하나로
정상(normal) / 훼손(defect) 두 클래스만 랜덤 생성해서 바로 YOLO-seg 학습까지 붙이는 것.
->  이 스크립트로 데이터 생성  ->  convert_sdg_to_yolo.py 로 YOLO-seg 포맷 변환
    ->  train_yolo_seg.py 로 학습

카메라는 일단 1대(cam1)만 사용. 파이프라인 검증되면 cam2/cam3 다시 붙이면 됨
(지금은 변수를 줄이는 게 우선).

STAGE_PATH(실제 컨베이어 환경)를 열어서 그 안의 조명/바닥 재질은 그대로 두고
PLACEMENT_AREA_MIN/MAX(실측 좌표) 범위 안에서 신발만 겹치지 않게 배치한다.

실행 (GPU 서버, headless):
    ./python.sh isaac_shoe_sdg.py --headless --num_frames 300
    # 먼저 --num_frames 3 정도로 짧게 돌려서 출력 구조/파일명부터 확인 추천
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_frames", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import math
import os
import random

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

random.seed(args.seed)
rep.set_global_seed(args.seed)

# ------------------------------------------------------------------------
# CONFIG (placeholder - 실제 값 확정되면 교체)
# ------------------------------------------------------------------------

OUTPUT_DIR = "/home/rokey/cobot3_ws/src/vision_node/_out_shoe_sdg"
RESOLUTION = (640, 640)

# 실제 컨베이어 환경 stage. 이 안에 이미 벨트/조명/AMR 등이 다 구성되어 있어서
# new_stage()로 빈 스테이지를 만들지 않고 이걸 그대로 연다.
STAGE_PATH = "/home/rokey/cobot3_ws/isaacpjt/stage_v10z/stage_v10.usd"

# 실측 컨베이어 배치 영역 (4개 코너 좌표에서 X/Y 범위만 추출, Z는 고정)
PLACEMENT_AREA_MIN = (-1.38818, 2.42414)
PLACEMENT_AREA_MAX = (-1.13057, 3.0775)
PLACEMENT_Z = 1.90685

SHOE_ASSET_URL = "/home/rokey/Downloads/sneaker_240.usd"

NUM_SHOES_PER_FRAME = 2      # 한 프레임에 흩뿌릴 신발 켤레 수 (좁은 컨베이어라 2켤레로 축소)
DEFECT_RATIO = 1           # 훼손 비율 (클래스 균형용, 필요시 조정)
CLASSES = ["normal", "defect"]

# defect 재질에 입힐 tear/scratch 절차적 텍스처. 색은 normal과 동일 베이스에 찢어짐/
# 긁힘 패턴만 얹는다. 매번 다른 패턴을 쓰도록 몇 개를 미리 생성해 랜덤으로 고른다.
DEFECT_TEXTURE_DIR = "/home/rokey/cobot3_ws/src/vision_node/_defect_textures"
DEFECT_TEXTURE_COUNT = 6
DEFECT_TEXTURE_SIZE = 1024

# 페어를 자유롭게(0~360도) 회전시키면 가로 폭(0.354m)이 세로로 눕혀지면서 옆 구간을
# 침범해 겹침이 생긴다. 벨트 방향(0도) 근처로만 살짝 흔들어서 Y 방향 차지 폭을 좁게 유지.
YAW_JITTER_DEG = 10

# 재질을 바꾼 직후 낮은 subframe으로는 RTX 누적 버퍼가 새 재질에 다 수렴하기 전에
# 캡처가 끝나서 이전 프레임 색이 살짝 섞여 나오는 문제(rgb_0001에서 발생)가 있었음.
# subframe을 넉넉히 줘서 매 프레임 캡처 전에 확실히 수렴하게 한다.
RT_SUBFRAMES = 16

# 배치 영역 정중앙 위에서 수직으로 내려다보는 탑뷰.
# 로컬 무회전(0,0,0)이 이미 -Z(아래쪽)을 보는 방향이라 회전 불필요.
_CAM_CENTER_X = (PLACEMENT_AREA_MIN[0] + PLACEMENT_AREA_MAX[0]) / 2
_CAM_CENTER_Y = (PLACEMENT_AREA_MIN[1] + PLACEMENT_AREA_MAX[1]) / 2
CAMERAS = {
    "cam1": {
        "position": (_CAM_CENTER_X, _CAM_CENTER_Y, PLACEMENT_Z + 2.3),
        "rotation": (0.0, 0.0, 0.0),
        "focal_length": 35.0,
    },
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


def make_material(stage, path, color, roughness):
    # 기존 환경의 신발 재질(/World/sneakers/_materials/sneaker_blue)과 동일한
    # UsdPreviewSurface를 써야 같은 RGB 값이 실제로 같은 색으로 렌더링된다.
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def generate_tear_scratch_texture(path, seed, base_color=(255, 64, 64)):
    """찢어짐(굵고 어두운 지그재그 선) + 긁힘(얇고 밝은 직선) 패턴을 절차적으로 그려서
    base_color 위에 합성한 텍스처를 저장한다. 실제 손상 사진이 없어도 UV 위에 바로 쓸 수 있음."""
    rng = random.Random(seed)
    size = DEFECT_TEXTURE_SIZE
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)

    dark = tuple(max(0, c - 150) for c in base_color)
    light = tuple(min(255, c + 70) for c in base_color)

    # tear: 지그재그로 꺾이는 굵은 어두운 선 + 가장자리 밝은 하이라이트
    for _ in range(rng.randint(2, 4)):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(size * 0.09, size * 0.21)
        steps = rng.randint(6, 10)
        points = [(x, y)]
        for _ in range(steps):
            angle += rng.uniform(-0.5, 0.5)
            step_len = length / steps
            x += math.cos(angle) * step_len
            y += math.sin(angle) * step_len
            points.append((x, y))
        width = rng.randint(int(size * 0.003), int(size * 0.008))
        draw.line(points, fill=dark, width=width, joint="curve")
        draw.line(points, fill=light, width=max(1, width // 3))

    # scratch: 짧고 얇은 직선 여러 개
    for _ in range(rng.randint(3, 6)):
        x1, y1 = rng.uniform(0, size), rng.uniform(0, size)
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(size * 0.04, size * 0.125)
        x2, y2 = x1 + math.cos(angle) * length, y1 + math.sin(angle) * length
        draw.line([(x1, y1), (x2, y2)], fill=light, width=rng.randint(1, max(2, int(size * 0.002))))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def make_textured_material(stage, path, texture_path, roughness):
    """diffuseColor를 상수 대신 UV 텍스처 파일로 연결하는 UsdPreviewSurface 재질."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, path.AppendPath("stReader"))
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, path.AppendPath("DiffuseTexture"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    # 텍스처 파일은 기본적으로 sRGB로 해석되어 렌더러가 linear로 변환하는데, 우리 PNG의
    # RGB 값은 이미 상수 재질(diffuseColor)과 같은 값으로 만들어둔 것이라 raw로 읽어야
    # normal 재질과 같은 색으로 보인다 (sRGB 디코드 시 채도가 과하게 높아지는 문제 있었음).
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_condition_materials(stage):
    # normal: 지정된 단색(1.0, 0.25, 0.25)
    # defect: 같은 베이스 색 위에 tear/scratch 절차적 텍스처를 입힌 변형 여러 개 중 랜덤 선택
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), (1.0, 0.25, 0.25), 0.55)

    defect_mats = []
    for i in range(DEFECT_TEXTURE_COUNT):
        tex_path = os.path.join(DEFECT_TEXTURE_DIR, f"defect_{i}.png")
        generate_tear_scratch_texture(tex_path, seed=args.seed * 1000 + i)
        defect_mats.append(
            make_textured_material(stage, Sdf.Path(f"/World/Looks/ShoeDefect_{i}"), tex_path, 0.9)
        )

    return {"normal": normal_mat, "defect": defect_mats}


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
        prim.SetInstanceable(False)   # ← 추가: 인스턴스 공유 해제

        # sneakers.usd 안에 자체 DomeLight(env_light)가 들어있는데, DomeLight는 계층과
        # 무관하게 스테이지 전역으로 작동해서 신발 수만큼 중복 누적되어 장면 전체 톤이
        # 왜곡된다. 우리 씬 조명(DomeLight+DistantLight)만 쓰도록 꺼둔다.
        env_light = stage.GetPrimAtPath(f"{prim_path}/env_light")
        if env_light.IsValid():
            env_light.SetActive(False)

        prims.append(prim)
    return prims


def compute_belt_segments(n):
    """컨베이어 길이(Y) 방향을 n개 구간으로 나누고, 각 구간 양 끝을 깎아
    옆 구간과 여백이 생기게 한다. 페어의 가로 폭(0.354m)이 세로 길이(0.241m)보다 커서
    회전이 크면 Y 방향 차지 폭이 확 늘어나므로, YAW_JITTER_DEG로 회전을 벨트 방향
    근처로 좁혀서 Y 폭을 예측 가능한 범위로 묶어두고 그에 맞춰 여백(margin)을 잡는다."""
    total = PLACEMENT_AREA_MAX[1] - PLACEMENT_AREA_MIN[1]
    span = total / n
    margin = 0.14   # YAW_JITTER_DEG 범위에서 페어가 Y 방향으로 차지하는 반폭 기준 여백(m)
    segments = []
    for i in range(n):
        y0 = PLACEMENT_AREA_MIN[1] + i * span
        segments.append((y0 + margin, y0 + span - margin))
    return segments


def randomize_shoe(prim, materials, y_segment):
    x = random.uniform(PLACEMENT_AREA_MIN[0], PLACEMENT_AREA_MAX[0])
    y = random.uniform(*y_segment)

    yaw = random.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)
    set_transform(prim, location=(x, y, PLACEMENT_Z), rotation=(0, 0, yaw))

    label = "defect" if random.random() < DEFECT_RATIO else "normal"
    material = random.choice(materials["defect"]) if label == "defect" else materials["normal"]
    bind_material_recursive(prim, material)
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
    print(f"[SDG] step: open_stage {STAGE_PATH}", flush=True)
    omni.usd.get_context().open_stage(STAGE_PATH)
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    # auto-exposure(히스토그램 기반 자동 노출)가 켜져 있으면 직전 프레임 밝기를 보고
    # 다음 프레임 노출을 조정해서, 같은 재질인데도 프레임마다 밝기/톤이 달라 보인다
    # (rgb_0001에서 같은 'normal' 재질이 갈색으로 보인 원인). 고정 노출로 끈다.
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    # 실제 환경(RectLight 등 기존 조명·바닥 재질)을 그대로 쓰므로 별도 조명/바닥을 만들지 않는다.

    print("[SDG] step: create_condition_materials", flush=True)
    materials = create_condition_materials(stage)
    print("[SDG] step: spawn_shoes (AddReference sneakers.usd x N)", flush=True)
    shoe_prims = spawn_shoes(stage, materials)
    print("[SDG] step: build_cameras", flush=True)
    cameras = build_cameras(stage)

    print("[SDG] step: create render_product (RTX init - can take a while on first run)", flush=True)
    render_products = [
        rep.create.render_product(cam_prim.GetPath(), RESOLUTION, name=name)
        for name, cam_prim in cameras.items()
    ]
    print("[SDG] step: render_product done", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    belt_segments = compute_belt_segments(len(shoe_prims))

    # 재질을 새로 바인딩한 직후 첫 스텝은 RTX가 셰이더를 아직 컴파일 중이라 색이
    # 제대로 안 나올 수 있어서(0000 프레임에서 회색으로 나온 원인), writer를 붙이기 전에
    # 워밍업 스텝을 한 번 돌려 셰이더가 준비된 뒤에 실제 캡처를 시작한다.
    print("[SDG] step: warm-up (shader compile)", flush=True)
    for prim, y_segment in zip(shoe_prims, belt_segments):
        randomize_shoe(prim, materials, y_segment)
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0)

    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Output directory: {OUTPUT_DIR}", flush=True)
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        bounding_box_2d_tight=True,
        instance_segmentation=True,
        colorize_instance_segmentation=False,
    )
    writer.attach(render_products)

    for i in range(args.num_frames):
        for prim, y_segment in zip(shoe_prims, belt_segments):
            randomize_shoe(prim, materials, y_segment)

        print(f"[SDG] Capturing frame {i + 1}/{args.num_frames}", flush=True)
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0)

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
