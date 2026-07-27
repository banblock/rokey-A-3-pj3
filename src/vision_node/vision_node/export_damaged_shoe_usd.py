"""
tear/scratch 손상을 적용한 신발(sneaker_240.usd 켤레) 하나를 잘라내서 별도의 .usd
파일로 저장한다. isaac_shoe_sdg.py는 카메라로 렌더링한 "이미지"만 저장하고 신발
지오메트리 자체는 저장하지 않아서, 손상 적용된 신발 애셋 자체가 필요할 때는 이
스크립트를 쓴다. 컨베이어/카메라/ROS 아무것도 안 건드리고 신발 하나만 빠르게 처리하므로
isaac_shoe_sdg.py를 GUI로 띄워둘 필요가 없다.

사용:
    ./python.sh export_damaged_shoe_usd.py --out /home/rokey/Downloads/sneaker_240_tear.usd
    ./python.sh export_damaged_shoe_usd.py --out out.usd --damage scratch --seed 7
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, help="저장할 .usd 경로")
parser.add_argument("--damage", choices=["tear", "scratch", "random"], default="random")
parser.add_argument("--seed", type=int, default=None, help="지정 안 하면 매번 다른 랜덤 결과")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

import math
import random

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

SHOE_ASSET_URL = "/home/rokey/Downloads/sneaker_240.usd"
TEAR_RX_VALUE = 0.02
TEAR_RY_RATIO_RANGE = (0.4, 0.6)
SCRATCH_LENGTH_RANGE = (0.06, 0.16)
SCRATCH_WIDTH_RATIO_RANGE = (0.15, 0.25)

rng = random.Random(args.seed)


def make_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_condition_materials(stage):
    base_color = (1.0, 0.25, 0.25)
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), base_color, 0.55)
    tear_mat = make_material(stage, Sdf.Path("/World/Looks/TearPatch"), tuple(max(0, c - 0.85) for c in base_color), 0.9)
    scratch_mat = make_material(stage, Sdf.Path("/World/Looks/ScratchPatch"), tuple(max(0, c - 0.85) for c in base_color), 0.9)
    return {"normal": normal_mat, "tear": tear_mat, "scratch": scratch_mat}


def _cache_mesh_topology(mesh):
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)

    offsets = np.concatenate(([0], np.cumsum(counts)))
    centroids = np.array([
        points[indices[offsets[i]:offsets[i + 1]]].mean(axis=0) for i in range(len(counts))
    ])

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

    normals_attr = mesh.GetNormalsAttr()
    normals = np.array(normals_attr.Get(), dtype=np.float64) if normals_attr.HasAuthoredValue() else None

    col_primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("Col")
    col_values = None
    if col_primvar and col_primvar.HasAuthoredValue() and not col_primvar.IsIndexed():
        col_values = np.array(col_primvar.Get(), dtype=np.float64)

    return {
        "counts": counts, "indices": indices, "centroids": centroids,
        "face_normals": face_normals, "st_primvar": st_primvar, "st_indices": st_indices,
        "normals_attr": normals_attr, "normals": normals,
        "col_primvar": col_primvar, "col_values": col_values,
    }


def _local_tangent_frame(centroids, seed, rng, rough_radius_range, ref_normal=None):
    bbox_size = centroids.max(axis=0) - centroids.min(axis=0)
    scale = float(np.linalg.norm(bbox_size))
    rough_radius = scale * rng.uniform(*rough_radius_range)

    dists = np.linalg.norm(centroids - centroids[seed], axis=1)
    local_mask = dists < rough_radius
    centered = centroids[local_mask] - centroids[seed]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis1, axis2, normal = vt[0], vt[1], vt[2]
    if ref_normal is not None and np.dot(normal, ref_normal) < 0:
        normal = -normal
    return axis1, axis2, normal, local_mask, scale


def cut_faces_near_random_point(mesh, topo, rng):
    counts, indices, centroids = topo["counts"], topo["indices"], topo["centroids"]
    n_faces = len(counts)
    seed = rng.randrange(n_faces)

    axis1, axis2, normal, _, scale = _local_tangent_frame(
        centroids, seed, rng, rough_radius_range=(0.05, 0.07), ref_normal=topo["face_normals"][seed])

    rel = centroids - centroids[seed]
    proj1 = rel @ axis1
    proj2 = rel @ axis2

    rx = scale * TEAR_RX_VALUE
    ry = rx * rng.uniform(*TEAR_RY_RATIO_RANGE)
    ellipse_t = (proj1 / rx) ** 2 + (proj2 / ry) ** 2
    keep_face = ellipse_t > 1.0
    if not keep_face.any():
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
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * (sides + 1))
    counts = [3] * sides
    indices = []
    for i in range(sides):
        indices += [0, 1 + i, 1 + (i + 1) % sides]
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    return mesh


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


def main():
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # default prim을 안 정해두면 이 파일을 다른 스테이지에 reference/payload로 넣었을 때
    # USD가 뭘 가져와야 할지 몰라서 아무 것도 안 보인다 ("payload doesn't have a default
    # prim" 경고). /World를 명시적으로 만들고 default prim으로 지정해야, Shoe 지오메트리와
    # Looks 머티리얼이 한 번에 같이 딸려 들어온다.
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    materials = create_condition_materials(stage)

    prim_path = "/World/Shoe"
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(SHOE_ASSET_URL)

    env_light = stage.GetPrimAtPath(f"{prim_path}/env_light")
    if env_light.IsValid():
        env_light.SetActive(False)

    meshes = [UsdGeom.Mesh(d) for d in Usd.PrimRange(prim) if d.IsA(UsdGeom.Mesh)]
    if not meshes:
        print(f"[error] {SHOE_ASSET_URL} 안에서 Mesh를 못 찾음")
        simulation_app.close()
        return

    damage_type = args.damage
    if damage_type == "random":
        damage_type = rng.choice(["tear", "scratch"])

    for m in meshes:
        UsdShade.MaterialBindingAPI(m).Bind(materials["normal"])
        topo = _cache_mesh_topology(m)

        if damage_type == "tear":
            tear_patch = _make_ellipse_patch(stage, m.GetPath().AppendChild("TearPatch"))
            UsdShade.MaterialBindingAPI(tear_patch).Bind(materials["tear"])
            center, axis1, axis2, normal, rx, ry = cut_faces_near_random_point(m, topo, rng)
            _update_ellipse_patch(tear_patch, center, axis1, axis2, normal, rx, ry)
        else:
            scratch_patch = _make_rect_patch(stage, m.GetPath().AppendChild("ScratchPatch"))
            UsdShade.MaterialBindingAPI(scratch_patch).Bind(materials["scratch"])
            center, dir_axis, side_axis, normal, length, width = pick_scratch_placement(topo, rng)
            _update_rect_patch(scratch_patch, center, dir_axis, side_axis, normal, length, width)

    stage.GetRootLayer().Save()
    print(f"[done] damage={damage_type} -> {args.out}")

    simulation_app.close()


main()
