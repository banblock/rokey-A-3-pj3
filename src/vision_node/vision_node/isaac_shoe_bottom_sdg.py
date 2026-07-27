"""
Isaac Sim Replicator - 바닥면 카메라(D455_3)용 신발 OBB 학습 데이터 생성.

isaac_shoe_sdg.py(위에서 보는 D455_1/D455_2, tear/scratch 손상 검출용)와는 목적이
다르다 — 이건 cam3(바닥/실측용 카메라)로 신발을 찍어서, 나중에 실제 길이(mm)를 재기
위한 **OBB(회전된 bounding box) 검출 모델** 학습용 데이터를 만드는 스크립트다.

신발 애셋(sneaker_240.usd)은 이름 그대로 실제 길이 240mm 기준으로 만들어져 있다.
실제 현장엔 240/260/280mm 신발이 섞여 들어오므로, 스폰할 때마다 그 중 하나로 무작위
스케일링(240 대비 배율)해서 다양한 크기의 신발을 학습 데이터에 섞는다 — 라벨 자체는
"shoe" 클래스 하나뿐이고, 크기(mm)는 학습에 안 들어간다. 실제 길이는 나중에 카메라
캘리브레이션으로 OBB의 픽셀 길이를 mm로 환산해서 구한다 (그러려면 모델이 다양한
크기/각도에서도 신발 외곽선에 딱 맞는 회전 박스를 잘 잡아야 하니, 크기 다양성이 필요).

한 켤레(Sneaker_L_1 + Sneaker_R_1)를 한 프레임에 하나만 스폰한다 — "shoe" 라벨은
왼발/오른발을 합친 페어 전체에 붙여서, 페어 하나당 OBB 하나가 나오게 한다(top-view
isaac_shoe_sdg.py의 "shoe" 라벨과 동일한 방식). 회전은 실사용 환경처럼 0~360도
전부 허용한다(OBB는 회전에 안 걸리므로 top-view 스크립트처럼 각도를 좁힐 이유가 없음).

instance_segmentation 마스크에서 인스턴스별로 cv2.minAreaRect를 구해 OBB 라벨(YOLO-OBB
포맷: class x1 y1 x2 y2 x3 y3 x4 y4, 0~1 정규화)로 바로 저장한다 — 별도 변환 스크립트
없이 이 스크립트 하나로 dataset_shoe_bottom_obb/{images,labels}/{train,val}까지 만든다.

실행 (GPU 서버, headless):
    ./python.sh isaac_shoe_bottom_sdg.py --headless --num_frames 300
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_frames", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--val_ratio", type=float, default=0.15)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

import json
import random
import shutil

import carb.settings
import cv2
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

random.seed(args.seed)
rep.set_global_seed(args.seed)

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------

RAW_OUTPUT_DIR = "/home/rokey/cobot3_ws/src/vision_node/_out_shoe_bottom_sdg"
DATASET_DIR = "/home/rokey/cobot3_ws/src/vision_node/dataset_shoe_bottom_obb"
RESOLUTION = (1280, 805)

STAGE_PATH = "/home/rokey/cobot3_ws/isaacpjt/stage_v10z/stage_v10.usd"

PLACEMENT_AREA_MIN = (-1.35669, 2.69612)
PLACEMENT_AREA_MAX = (-1.23014, 2.80214)
PLACEMENT_Z = 1.89

SHOE_ASSET_URL = "/home/rokey/Downloads/sneaker_240.usd"
BASE_SHOE_LENGTH_MM = 240
SHOE_LENGTH_CHOICES_MM = [240, 260, 280]

CLASSES = ["shoe"]

RT_SUBFRAMES = 16

# 바닥/실측용 D455_3 카메라를 그대로 쓴다 (다른 두 카메라와 같은 리그, 셋 다 실제 물리
# 카메라라서 위치를 새로 잡을 필요 없음).
CAMERA_PRIM_PATH = "/World/camera/D455_3/Sensor/RSD455/Camera_OmniVision_OV9782_Color"


# ------------------------------------------------------------------------
# 씬 구성
# ------------------------------------------------------------------------

def set_transform(prim, location=None, rotation=None, scale=None):
    xform = UsdGeom.Xformable(prim)
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            xform.AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*location))
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            xform.AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*rotation))
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            xform.AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*scale))


def make_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def spawn_shoe(stage, material):
    prim_path = "/World/BottomShoe"
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(SHOE_ASSET_URL)

    env_light = stage.GetPrimAtPath(f"{prim_path}/env_light")
    if env_light.IsValid():
        env_light.SetActive(False)

    meshes = [UsdGeom.Mesh(d) for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh)]
    for m in meshes:
        UsdShade.MaterialBindingAPI(m).Bind(material)
    # 왼발/오른발을 따로가 아니라 페어 전체를 하나의 instance로 잡는다 (top-view
    # isaac_shoe_sdg.py의 "shoe" 라벨과 동일한 방식 — 페어 전체에 라벨을 붙이면
    # 그 안의 두 mesh가 모두 하나의 인스턴스로 묶인다).
    add_labels(prim, labels=["shoe"], instance_name="class")

    return {"prim": prim}


def randomize_shoe(shoe):
    x = random.uniform(PLACEMENT_AREA_MIN[0], PLACEMENT_AREA_MAX[0])
    y = random.uniform(PLACEMENT_AREA_MIN[1], PLACEMENT_AREA_MAX[1])
    # OBB는 회전에 안 걸리니, top-view 스크립트처럼 각도를 벨트 방향 근처로 좁힐 필요가
    # 없다 — 실사용 환경처럼 완전 무작위 회전을 다 허용해서 학습 다양성을 높인다.
    yaw = random.uniform(0, 360)
    length_mm = random.choice(SHOE_LENGTH_CHOICES_MM)
    # 실제 신발은 사이즈가 커져도 폭/높이는 거의 그대로고 길이만 늘어난다. 메시 로컬
    # 좌표에서 실측으로 확인한 결과 Y축이 길이 방향(extent_y≈0.241m≈240mm)이라, Y만
    # 스케일하고 X/Z(폭/높이)는 1.0으로 고정한다 (전체를 동일 배율로 키우지 않음).
    length_scale = length_mm / BASE_SHOE_LENGTH_MM
    set_transform(shoe["prim"], location=(x, y, PLACEMENT_Z), rotation=(0, 0, yaw),
                  scale=(1.0, length_scale, 1.0))
    return length_mm


def build_camera(stage):
    cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"카메라 프림을 찾을 수 없음: {CAMERA_PRIM_PATH}")
    return cam_prim


# ------------------------------------------------------------------------
# 세그멘테이션 마스크 -> YOLO-OBB 라벨
# ------------------------------------------------------------------------

def instance_class_from_mapping(mapping, key):
    entry = mapping.get(key)
    if entry is None:
        return None
    cls = entry.get("class")
    if isinstance(cls, list):
        cls = cls[0] if cls else None
    return cls


def seg_to_obb_lines(seg_path, mapping_path, img_w, img_h):
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg is None:
        return []
    with open(mapping_path) as f:
        mapping = json.load(f)

    lines = []
    unique_vals = np.unique(seg)
    for val in unique_vals:
        key = str(int(val))
        cls_name = instance_class_from_mapping(mapping, key)
        if cls_name != "shoe":
            continue
        mask = (seg == val).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # 페어 전체가 하나의 instance라, 왼발/오른발이 화면상 안 붙어있으면(각도에 따라
        # 흔함) contour가 두 조각으로 나뉜다. 제일 큰 조각만 쓰면 한쪽 발이 통째로
        # 빠지므로, 작은 노이즈(<20px)만 걸러내고 나머지 조각을 전부 합쳐서 그 위에
        # OBB를 씌운다 — 두 발을 함께 감싸는 하나의 회전 박스가 나온다.
        pts_all = [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) >= 20]
        if not pts_all:
            continue
        merged = np.concatenate(pts_all, axis=0)
        rect = cv2.minAreaRect(merged)
        box = cv2.boxPoints(rect).astype(np.float32)
        box[:, 0] /= img_w
        box[:, 1] /= img_h
        coords = " ".join(f"{v:.6f}" for v in box.flatten())
        lines.append(f"0 {coords}")
    return lines


# ------------------------------------------------------------------------

def run():
    # 이 스크립트는 ROS2 토픽을 안 쓴다(Replicator render_product로 직접 캡처) —
    # isaac_shoe_sdg.py에 넣었던 ROS2 브릿지 확장 활성화는 여기선 불필요하고,
    # 이전에 이 호출이 원인 불명으로 오래 멈추는 걸 겪어서 아예 뺐다.
    print(f"[SDG] step: open_stage {STAGE_PATH}", flush=True)
    omni.usd.get_context().open_stage(STAGE_PATH)
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    # isaac_shoe_sdg.py(tear/scratch)는 같은 mesh의 면(topology)을 매 프레임 잘랐다
    # 복구해서 RTX Real-Time에 잔상 버그가 있었지만, 이 스크립트는 randomize_shoe()가
    # translate/rotate/scale만 바꾸고 topology는 안 건드린다 — 실측으로 확인한 결과
    # 이 종류의 변경(이동/숨김)은 Real-Time에서도 잔상이 없었으므로, 훨씬 빠른
    # RaytracedLighting을 headless에서도 그대로 쓴다 (PathTracing 대비 7~10배 빠름).
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

    shoe_material = make_material(stage, Sdf.Path("/World/Looks/BottomShoeNormal"), (1.0, 0.25, 0.25), 0.55)
    shoe = spawn_shoe(stage, shoe_material)
    cam_prim = build_camera(stage)

    render_product = rep.create.render_product(cam_prim.GetPath(), RESOLUTION, name="D455_3")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print("[SDG] step: warm-up (shader compile)", flush=True)
    randomize_shoe(shoe)
    rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0)

    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Output directory: {RAW_OUTPUT_DIR}", flush=True)
    writer.initialize(
        output_dir=RAW_OUTPUT_DIR,
        rgb=True,
        instance_segmentation=True,
        colorize_instance_segmentation=False,
    )
    writer.attach([render_product])

    import pathlib
    dataset_dir = pathlib.Path(DATASET_DIR)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for i in range(args.num_frames):
        length_mm = randomize_shoe(shoe)

        print(f"[SDG] Capturing frame {i + 1}/{args.num_frames} (length={length_mm}mm)", flush=True)
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0)
        rep.orchestrator.wait_until_complete()

        idx = f"{i:04d}"
        rgb_path = pathlib.Path(RAW_OUTPUT_DIR) / f"rgb_{idx}.png"
        seg_path = pathlib.Path(RAW_OUTPUT_DIR) / f"instance_segmentation_{idx}.png"
        map_path = pathlib.Path(RAW_OUTPUT_DIR) / f"instance_segmentation_semantics_mapping_{idx}.json"
        if not (rgb_path.exists() and seg_path.exists() and map_path.exists()):
            print(f"[skip] frame {idx}: 출력 파일 누락", flush=True)
            continue

        lines = seg_to_obb_lines(seg_path, map_path, RESOLUTION[0], RESOLUTION[1])
        if not lines:
            print(f"[skip] frame {idx}: shoe instance 없음", flush=True)
            continue

        split = "val" if random.random() < args.val_ratio else "train"
        out_name = f"D455_3_{idx}"
        shutil.copy(rgb_path, dataset_dir / "images" / split / f"{out_name}.png")
        (dataset_dir / "labels" / split / f"{out_name}.txt").write_text("\n".join(lines))
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    data_yaml = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (dataset_dir / "data.yaml").write_text(data_yaml)
    print(f"[done] train={n_train} val={n_val} -> {dataset_dir/'data.yaml'}", flush=True)

    writer.detach()
    rep.orchestrator.wait_until_complete()
    if args.headless:
        render_product.destroy()
        timeline.stop()
    print("[SDG] Done.")


run()

# headless는 GUI로 볼 사람이 없으니 끝나면 바로 닫는다. is_running()이 자연스럽게
# False가 되길 기다리는 idle 루프에 헤드리스 프로세스가 여러 개 계속 걸려있는 채로
# 안 꺼지고 GPU 메모리를 계속 붙잡고 있던 문제(OOM 유발)가 있어서, headless일 땐
# 그 루프를 아예 안 타고 바로 종료하도록 바꿨다.
if args.headless:
    simulation_app.close()
else:
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()
