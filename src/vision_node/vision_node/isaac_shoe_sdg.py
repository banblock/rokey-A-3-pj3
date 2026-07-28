"""
Isaac Sim Replicator - 신발 정상/훼손 2클래스 합성 데이터 생성 (1차 단순 버전)

지금 단계 목표: 색상/사이즈 클래스 다 빼고, 빨간 신발 하나로
정상(normal) / 훼손(defect) 두 클래스만 랜덤 생성해서 바로 YOLO-seg 학습까지 붙이는 것.
->  이 스크립트로 데이터 생성  ->  convert_sdg_to_yolo.py 로 YOLO-seg 포맷 변환
    ->  train_yolo_seg.py 로 학습

카메라는 우리가 새로 만들지 않고, 스테이지에 이미 있는 실제 D455_1/D455_2 카메라
프림(ROS2로 /d455_1, /d455_2/color/image_raw를 발행하는 바로 그 카메라)을 그대로
가져다 render_product로 쓴다 (CAMERA_PRIM_PATHS 참고). 카메라가 2대라 출력이
<out>/<cam_name>/rgb/, <out>/<cam_name>/instance_segmentation/ 형태의 서브폴더
구조로 나뉘며, convert_sdg_to_yolo.py도 이 구조를 인식하도록 되어 있다.

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
import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

random.seed(args.seed)
rep.set_global_seed(args.seed)

# ------------------------------------------------------------------------
# CONFIG (placeholder - 실제 값 확정되면 교체)
# ------------------------------------------------------------------------

OUTPUT_DIR = "/home/rokey/cobot3_ws/src/vision_node/_out_shoe_sdg"
# 실제 D455 컬러 카메라 aspect(horizontalAperture/verticalAperture ≈ 3.896/2.453 ≈ 1.59)에 맞춤.
# D455 컬러 스트림 실제 해상도(1280x800)에 가깝게 올림.
RESOLUTION = (1280, 805)

# 실제 컨베이어 환경 stage. 이 안에 이미 벨트/조명/AMR 등이 다 구성되어 있어서
# new_stage()로 빈 스테이지를 만들지 않고 이걸 그대로 연다.
STAGE_PATH = "/home/rokey/cobot3_ws/isaacpjt/stage_v10z/stage_v10.usd"

# 실측 컨베이어 배치 영역 (4개 코너 좌표에서 X/Y 범위만 추출, Z는 고정)
PLACEMENT_AREA_MIN = (-1.38818, 2.42414)
PLACEMENT_AREA_MAX = (-1.13057, 3.0775)
PLACEMENT_Z = 1.89

SHOE_ASSET_URL = "/home/rokey/Downloads/sneaker_240.usd"

NUM_SHOES_PER_FRAME = 1      # 한 프레임에 흩뿌릴 신발 켤레 수 (좁은 컨베이어라 2켤레로 축소)
DEFECT_RATIO = 0.7       # 훼손(tear 또는 scratch) 비율 (클래스 균형용, 필요시 조정)
# "shoe"는 신발 전체(항상 부여, 정상/훼손 구분 없음). tear/scratch는 훼손 부위에 실제로
# 만드는 별도의 작은 프림(패치)에 붙이는 라벨이라, bbox/segmentation이 훼손 부위만 잡는다.
# (GeomSubset에 라벨을 걸어봤는데 Replicator의 instance_segmentation/bbox 애노테이터가
# subset 단위 시맨틱을 인식하지 못해서, 실제 프림으로 바꿈)
CLASSES = ["shoe", "tear", "scratch"]

# tear(구멍 자리 어두운 패치)/scratch(긁힌 자국 밝은 패치) 모양 크기.
# tear는 cut_faces_near_random_point의 반경과 반드시 맞춰야 구멍 크기와 패치 크기가 일치함.
TEAR_RX_VALUE = 0.02
TEAR_RY_RATIO_RANGE = (0.4, 0.6)
SCRATCH_LENGTH_RANGE = (0.06, 0.16)     # bbox 대각선 대비 비율
SCRATCH_WIDTH_RATIO_RANGE = (0.15, 0.25)  # 길이 대비 두께 비율

# 페어를 자유롭게(0~360도) 회전시키면 가로 폭(0.354m)이 세로로 눕혀지면서 옆 구간을
# 침범해 겹침이 생긴다. 벨트 방향(0도) 근처로만 살짝 흔들어서 Y 방향 차지 폭을 좁게 유지.
YAW_JITTER_DEG = 10

# 재질을 바꾼 직후 낮은 subframe으로는 RTX 누적 버퍼가 새 재질에 다 수렴하기 전에
# 캡처가 끝나서 이전 프레임 색이 살짝 섞여 나오는 문제(rgb_0001에서 발생)가 있었음.
# subframe을 넉넉히 줘서 매 프레임 캡처 전에 확실히 수렴하게 한다.
RT_SUBFRAMES = 16

# 우리가 임의로 만든 탑뷰 카메라 대신, 스테이지에 이미 있는 실제 D455 카메라 리그를
# 그대로 쓴다 (ROS2로 /d455_1, /d455_2/color/image_raw를 발행하는 바로 그 카메라).
# 위치/각도를 우리가 새로 잡을 필요 없이 실제 물리 카메라 시점 그대로 캡처된다.
CAMERA_PRIM_PATHS = {
    "D455_1": "/World/camera/D455_1/Sensor/RSD455/Camera_OmniVision_OV9782_Color",
    "D455_2": "/World/camera/D455_2/Sensor/RSD455/Camera_OmniVision_OV9782_Color",
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


def create_condition_materials(stage):
    # normal: 신발 전체 기본 색. tear/scratch 패치는 이제 실제 별도 프림이라, 각자
    # 자기 재질(어두운 색/밝은 색)을 그 작은 패치에만 바른다 (신발 몸체는 항상 normal).
    base_color = (1.0, 0.25, 0.25)
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), base_color, 0.55)
    tear_mat = make_material(stage, Sdf.Path("/World/Looks/TearPatch"), tuple(max(0, c - 0.85) for c in base_color), 0.9)
    scratch_mat = make_material(stage, Sdf.Path("/World/Looks/ScratchPatch"), tuple(max(0, c - 0.85) for c in base_color), 0.9)
    return {"normal": normal_mat, "tear": tear_mat, "scratch": scratch_mat}


def _cache_mesh_topology(mesh):
    """면 삭제로 구멍을 냈다가 'normal'로 돌아갈 때 원래 모양으로 복구할 수 있도록,
    스폰 시점(아직 아무것도 자르기 전)의 원본 위상 정보를 한 번만 캐싱해둔다.
    face centroid도 여기서 미리 계산해서, 매 프레임 반복되는 절단 연산은 numpy로 가볍게 만든다."""
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)

    offsets = np.concatenate(([0], np.cumsum(counts)))
    centroids = np.array([
        points[indices[offsets[i]:offsets[i + 1]]].mean(axis=0) for i in range(len(counts))
    ])

    # 각 면 자체의 정점 순서(winding)로 구한 "진짜" 법선. PCA 법선은 이웃 면들의
    # 분산으로 축만 구하는 거라 부호(안/밖)가 seed마다 무작위로 나오는데, 이 면
    # 법선은 winding이 고정이라 부호가 항상 같은 쪽(Hydra가 렌더링에 쓰는 쪽)을
    # 가리킨다. PCA 법선의 부호를 여기에 맞춰 고정하는 데 쓴다.
    face_normals = np.zeros_like(centroids)
    for i in range(len(counts)):
        verts = indices[offsets[i]:offsets[i + 1]]
        if len(verts) >= 3:
            p0, p1, p2 = points[verts[0]], points[verts[1]], points[verts[2]]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                face_normals[i] = n / norm

    st_primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    st_indices = None
    if st_primvar and st_primvar.IsIndexed():
        st_indices = np.array(st_primvar.GetIndices(), dtype=np.int64)

    # 이 메시는 face-varying(코너별) authored normals를 갖고 있다. counts/indices만
    # 자르고 복구하면 normals 배열 길이가 안 맞아서 Hydra가 매 면 삭제 이후로 계속
    # smooth-normal로 대체 렌더링해버리는 버그가 있었다 — normals도 같은 corner_mask로
    # 같이 잘라내고, 복구 시 원본 값으로 되돌려야 한다.
    normals_attr = mesh.GetNormalsAttr()
    normals = np.array(normals_attr.Get(), dtype=np.float64) if normals_attr.HasAuthoredValue() else None

    # 원본 에셋에 face-varying "Col" 컬러 primvar(정점 채색/마스크용, 47920개, non-indexed)도
    # 있어서 normals와 똑같은 이유로 같이 잘라내고 복구해야 한다.
    col_primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("Col")
    col_values = None
    if col_primvar and col_primvar.HasAuthoredValue() and not col_primvar.IsIndexed():
        col_values = np.array(col_primvar.Get(), dtype=np.float64)

    return {
        "counts": counts,
        "indices": indices,
        "centroids": centroids,
        "face_normals": face_normals,
        "st_primvar": st_primvar,
        "st_indices": st_indices,
        "normals_attr": normals_attr,
        "normals": normals,
        "col_primvar": col_primvar,
        "col_values": col_values,
    }


def _local_tangent_frame(centroids, seed, rng, rough_radius_range, ref_normal=None):
    """centroids[seed] 근처 면들의 국소 접평면(PCA)을 구해 그 지점의 두 접선축
    (axis1, axis2)과 법선축(normal, 분산이 가장 작은 축), 국소 이웃 마스크를 반환한다.
    tear(구멍)와 scratch(패치) 둘 다 이 프레임 위에서 위치/모양을 계산한다.

    PCA로 구한 normal은 축만 정해지고 부호(안/밖)는 SVD가 임의로 정해서 seed마다
    무작위로 뒤집힐 수 있다. ref_normal(그 seed 면 자체의 winding 기반 법선)을 주면
    그 방향과 같은 쪽을 향하도록 부호를 고정한다 — 안 그러면 패치 오프셋이 표면
    안쪽으로 들어가 카메라에서 가려지는 경우가 생긴다."""
    bbox_size = centroids.max(axis=0) - centroids.min(axis=0)
    scale = float(np.linalg.norm(bbox_size))
    rough_radius = scale * rng.uniform(*rough_radius_range)

    dists = np.linalg.norm(centroids - centroids[seed], axis=1)
    local_mask = dists < rough_radius
    centered = centroids[local_mask] - centroids[seed]
    # PCA: 분산이 큰 두 축 = 이 지점의 국소 접평면, 가장 작은 축 = 법선
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis1, axis2, normal = vt[0], vt[1], vt[2]
    if ref_normal is not None and np.dot(normal, ref_normal) < 0:
        normal = -normal
    return axis1, axis2, normal, local_mask, scale


def cut_faces_near_random_point(mesh, topo, rng):
    """topo에 캐싱된 원본 위상에서, 무작위로 고른 한 면 근처를 타원형 경계로 잘라
    제외하고 재구성해서 실제로 구멍이 뚫린 것처럼 만든다. 단순 구형 반경 대신, 시드
    주변 면들의 국소 평면(PCA)을 구해 그 평면 위에서 타원 판정을 하기 때문에 표면
    곡률을 따라 자연스러운 타원 형태가 나온다 (면 자체가 크면 여전히 각질 수 있음).
    faceVertexIndices뿐 아니라 같은 face-corner 순서를 공유하는 st(UV) 인덱스도
    같이 걸러줘야 어긋나지 않는다.

    반환값: (center, axis1, axis2, normal, rx, ry) — 이 자리에 tear 패치 프림을
    똑같은 크기/방향으로 만들기 위한 정보."""
    counts, indices, centroids = topo["counts"], topo["indices"], topo["centroids"]
    n_faces = len(counts)
    seed = rng.randrange(n_faces)

    # 지금 tear 패치처럼 자연스럽게 둥글어 보이려면 경계에 걸치는 면이 충분히 많아야
    # 해서(면이 크면 어차피 각질 수밖에 없음), 이전보다 크게 잡는다.
    axis1, axis2, normal, _, scale = _local_tangent_frame(
        centroids, seed, rng, rough_radius_range=(0.05, 0.07), ref_normal=topo["face_normals"][seed])

    rel = centroids - centroids[seed]
    proj1 = rel @ axis1
    proj2 = rel @ axis2

    rx = scale * TEAR_RX_VALUE
    ry = rx * rng.uniform(*TEAR_RY_RATIO_RANGE)
    ellipse_t = (proj1 / rx) ** 2 + (proj2 / ry) ** 2
    keep_face = ellipse_t > 1.0
    if not keep_face.any():   # 전부 사라지는 극단적인 경우 방지
        keep_face[:] = True

    corner_mask = np.repeat(keep_face, counts)
    new_counts = counts[keep_face]
    new_indices = indices[corner_mask]
    mesh.GetFaceVertexCountsAttr().Set(new_counts.tolist())
    mesh.GetFaceVertexIndicesAttr().Set(new_indices.tolist())

    if topo["st_indices"] is not None:
        topo["st_primvar"].SetIndices(Vt.IntArray(topo["st_indices"][corner_mask].tolist()))

    if topo["normals"] is not None:
        topo["normals_attr"].Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in topo["normals"][corner_mask]]))

    if topo["col_values"] is not None:
        topo["col_primvar"].Set(Vt.Vec4fArray([Gf.Vec4f(*c) for c in topo["col_values"][corner_mask]]))

    return centroids[seed], axis1, axis2, normal, rx, ry


def pick_scratch_placement(topo, rng):
    """원본 위상(잘라내지 않음)에서 무작위 지점 근처의 국소 접평면을 구해, scratch
    패치를 놓을 위치/방향/길이/두께를 정한다. 면을 제거하지는 않는다."""
    centroids = topo["centroids"]
    n_faces = len(centroids)
    seed = rng.randrange(n_faces)

    axis1, axis2, normal, _, scale = _local_tangent_frame(
        centroids, seed, rng, rough_radius_range=(0.08, 0.12), ref_normal=topo["face_normals"][seed])
    angle = rng.uniform(0, 2 * math.pi)
    dir_axis = axis1 * math.cos(angle) + axis2 * math.sin(angle)
    side_axis = -axis1 * math.sin(angle) + axis2 * math.cos(angle)

    length = scale * rng.uniform(*SCRATCH_LENGTH_RANGE)
    width = length * rng.uniform(*SCRATCH_WIDTH_RATIO_RANGE)
    return centroids[seed], dir_axis, side_axis, normal, length, width


def _make_ellipse_patch(stage, path, sides=16):
    """tear 패치용으로 매 프레임 위치/모양을 갱신할, 부채꼴(fan) 삼각분할 원반 프림을
    한 번만 만들어둔다 (점/면 개수는 고정, 점 좌표만 매 프레임 다시 Set)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * (sides + 1))
    counts = [3] * sides
    indices = []
    for i in range(sides):
        indices += [0, 1 + i, 1 + (i + 1) % sides]
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    return mesh


# 패치를 표면과 완전히 같은 높이에 놓으면 원본 메시 면과 겹쳐 z-fighting(깜빡임)이
# 날 수 있어서, 법선 방향으로 아주 살짝(0.5mm) 띄운다.
_PATCH_OFFSET = 0.0005


def _update_ellipse_patch(mesh, center, axis1, axis2, normal, rx, ry, sides=16):
    base = center + normal * _PATCH_OFFSET
    points = [Gf.Vec3f(*base)]
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        p = base + axis1 * (rx * math.cos(theta)) + axis2 * (ry * math.sin(theta))
        points.append(Gf.Vec3f(*p))
    mesh.GetPointsAttr().Set(points)


def _make_rect_patch(stage, path):
    """scratch 패치용 사각형(quad) 프림을 한 번만 만들어둔다."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * 4)
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    return mesh


def _update_rect_patch(mesh, center, dir_axis, side_axis, normal, length, width):
    base = center + normal * _PATCH_OFFSET
    half_l, half_w = length / 2, width / 2
    corners = [
        base - dir_axis * half_l - side_axis * half_w,
        base + dir_axis * half_l - side_axis * half_w,
        base + dir_axis * half_l + side_axis * half_w,
        base - dir_axis * half_l + side_axis * half_w,
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in corners])


def restore_mesh_topology(mesh, topo):
    """'normal'로 돌아갈 때 스폰 시점의 원본 위상으로 되돌린다."""
    mesh.GetFaceVertexCountsAttr().Set(topo["counts"].tolist())
    mesh.GetFaceVertexIndicesAttr().Set(topo["indices"].tolist())
    if topo["st_indices"] is not None:
        topo["st_primvar"].SetIndices(Vt.IntArray(topo["st_indices"].tolist()))
    if topo["normals"] is not None:
        topo["normals_attr"].Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in topo["normals"]]))
    if topo["col_values"] is not None:
        topo["col_primvar"].Set(Vt.Vec4fArray([Gf.Vec4f(*c) for c in topo["col_values"]]))


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

        meshes = [UsdGeom.Mesh(desc) for desc in Usd.PrimRange(prim) if desc.IsA(UsdGeom.Mesh)]
        mesh_entries = []
        for m in meshes:
            # tear/scratch 훼손 부위만 따로 bbox/segmentation이 잡히도록, 실제 별도 프림
            # (패치)을 하나씩 만들어두고 매 프레임 모양/위치/보임여부만 갱신한다.
            # 패치를 메시 프림의 자식으로 둬서, 메시 로컬 좌표(centroid 등)를 그대로 쓸 수
            # 있게 한다 (형제로 두면 메시 자체의 로컬 변환까지 따로 계산해야 함).
            tear_patch = _make_ellipse_patch(stage, m.GetPath().AppendChild("TearPatch"))
            scratch_patch = _make_rect_patch(stage, m.GetPath().AppendChild("ScratchPatch"))
            UsdShade.MaterialBindingAPI(tear_patch).Bind(materials["tear"])
            UsdShade.MaterialBindingAPI(scratch_patch).Bind(materials["scratch"])
            add_labels(tear_patch.GetPrim(), labels=["tear"], instance_name="class")
            add_labels(scratch_patch.GetPrim(), labels=["scratch"], instance_name="class")
            UsdGeom.Imageable(tear_patch).MakeInvisible()
            UsdGeom.Imageable(scratch_patch).MakeInvisible()
            mesh_entries.append({
                "mesh": m,
                "topo": _cache_mesh_topology(m),
                "tear_patch": tear_patch,
                "scratch_patch": scratch_patch,
            })

        prims.append({"prim": prim, "meshes": mesh_entries})
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


def randomize_shoe(shoe, materials, y_segment):
    prim = shoe["prim"]
    x = random.uniform(PLACEMENT_AREA_MIN[0], PLACEMENT_AREA_MAX[0])
    y = random.uniform(*y_segment)

    yaw = random.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)
    set_transform(prim, location=(x, y, PLACEMENT_Z), rotation=(0, 0, yaw))

    # "shoe"는 훼손 여부와 무관하게 신발 전체에 항상 붙는 라벨 (정상/훼손 구분 없음).
    # add_labels는 이전 'class' 라벨을 덮어쓰므로 매 프레임 재호출해도 안전함
    add_labels(prim, labels=["shoe"], instance_name="class")

    has_damage = random.random() < DEFECT_RATIO
    damage_type = random.choice(["tear", "scratch"]) if has_damage else None

    for entry in shoe["meshes"]:
        mesh, topo = entry["mesh"], entry["topo"]
        tear_patch, scratch_patch = entry["tear_patch"], entry["scratch_patch"]

        # 항상 원본 위상에서 다시 시작해야 tear가 누적으로 계속 뚫리지 않는다.
        restore_mesh_topology(mesh, topo)
        UsdShade.MaterialBindingAPI(mesh).Bind(materials["normal"])   # 신발 몸체 기본 재질

        UsdGeom.Imageable(tear_patch).MakeInvisible()
        UsdGeom.Imageable(scratch_patch).MakeInvisible()

        if damage_type == "tear":
            center, axis1, axis2, normal, rx, ry = cut_faces_near_random_point(mesh, topo, random)
            _update_ellipse_patch(tear_patch, center, axis1, axis2, normal, rx, ry)
            UsdGeom.Imageable(tear_patch).MakeVisible()
        elif damage_type == "scratch":
            center, dir_axis, side_axis, normal, length, width = pick_scratch_placement(topo, random)
            _update_rect_patch(scratch_patch, center, dir_axis, side_axis, normal, length, width)
            UsdGeom.Imageable(scratch_patch).MakeVisible()


def build_cameras(stage):
    """새 카메라를 만들지 않고, 스테이지에 이미 있는 실제 D455 카메라 프림을 그대로 가져온다."""
    cams = {}
    for name, path in CAMERA_PRIM_PATHS.items():
        cam_prim = stage.GetPrimAtPath(path)
        if not cam_prim.IsValid():
            raise RuntimeError(f"카메라 프림을 찾을 수 없음: {path}")
        cams[name] = cam_prim
    return cams


def run():
    # 스테이지(카메라의 ROS_D455_1 등 OmniGraph)를 열기 전에 ROS2 브릿지 확장을 먼저
    # 켜야 한다. 확장이 꺼진 상태로 스테이지를 열면 ROS2CameraHelper 같은 노드 타입이
    # "unregistered node type"으로 실패한 채 굳어버리고, 나중에 확장을 켜도(재생을
    # 껐다 켜도) 그 실패한 노드 인스턴스는 자동으로 재바인딩되지 않는다 — 스테이지를
    # 다시 열어야만 고쳐지는데, 그럴 바엔 애초에 열기 전에 켜두는 게 안전하다.
    # set_extension_enabled_immediate는 앱이 아직 다 초기화되기 전에 동기적으로 강제
    # 등록하려다 omni.graph.core 쪽과 충돌해서 세그폴트가 났다. Isaac Sim 표준 헬퍼
    # (enable_extension)를 쓰면 내부적으로 안전한 절차로 처리된다.
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    print(f"[SDG] step: open_stage {STAGE_PATH}", flush=True)
    omni.usd.get_context().open_stage(STAGE_PATH)
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    # auto-exposure(히스토그램 기반 자동 노출)가 켜져 있으면 직전 프레임 밝기를 보고
    # 다음 프레임 노출을 조정해서, 같은 재질인데도 프레임마다 밝기/톤이 달라 보인다
    # (rgb_0001에서 같은 'normal' 재질이 갈색으로 보인 원인). 고정 노출로 끈다.
    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    # RTX Real-Time(RayTracedLighting)은 프레임 간 누적 버퍼가 완전히 안 지워져서, tear를
    # 껐다 켰다 해도 이전 프레임의 흔적이 잔상처럼 옅게 남는 문제가 실측으로 확인됐다.
    # PathTracing(RTX-Interactive)은 매 프레임 독립적으로 수렴해서 이 잔상이 없다 — 그래서
    # 실제 데이터 생성(headless)은 PathTracing을 쓴다. GUI로 띄울 때는 데이터 생성용이
    # 아니라 씬 확인용이고, PathTracing은 GUI에서 재생 버튼 등과 충돌해 크래시 나는 걸
    # 확인했으므로 GUI는 그냥 Real-Time(RayTracedLighting)으로 켠다.
    if args.headless:
        carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
        carb.settings.get_settings().set("/rtx/pathtracing/spp", 16)
        carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", 16)
    else:
        carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
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
    for shoe, y_segment in zip(shoe_prims, belt_segments):
        randomize_shoe(shoe, materials, y_segment)
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
        for shoe, y_segment in zip(shoe_prims, belt_segments):
            randomize_shoe(shoe, materials, y_segment)

        print(f"[SDG] Capturing frame {i + 1}/{args.num_frames}", flush=True)
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0)

    writer.detach()
    rep.orchestrator.wait_until_complete()
    # headless는 이 직후 바로 종료되니 안전하게 정리하지만, GUI 모드는 사용자가 생성 후
    # 씬을 계속 보거나 재생 버튼을 누를 수 있어서 render_product를 파괴/timeline 정지하면
    # (이미 파괴된 render_product를 참조하게 돼) 재생 시 크래시가 난다. GUI에선 그대로 둔다.
    if args.headless:
        for rp in render_products:
            rp.destroy()
        timeline.stop()
    print("[SDG] Done.")


run()

# headless는 GUI로 볼 사람이 없으니 끝나면 바로 닫는다. is_running()이 자연스럽게
# False가 되길 기다리는 idle 루프에 헤드리스 프로세스가 여러 개 안 꺼진 채로 계속
# GPU 메모리를 붙잡고 쌓여서 OOM을 유발한 적이 있어서, headless일 땐 그 루프를
# 아예 안 타고 바로 종료하도록 바꿨다.
if args.headless:
    simulation_app.close()
else:
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()
