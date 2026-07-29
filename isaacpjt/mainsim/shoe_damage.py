"""신발 메시에 tear/scratch 손상을 적용하는 순수 함수 모음.

원래 export_damaged_shoe_usd.py 안에 있던 로직을 그대로 옮겨온 것으로,
그 스크립트(새 .usd 파일로 내보내기)와 simulation_node.py(라이브 스테이지의
신발 프림에 그때그때 적용) 양쪽에서 공유해서 쓴다. SimulationApp/스테이지
생성 등 실행 환경에 대한 가정이 전혀 없는 순수 pxr/numpy 함수만 담는다 -
호출하는 쪽에서 이미 SimulationApp이 떠 있어야 pxr import가 가능하다.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

TEAR_RX_VALUE = 0.02
TEAR_RY_RATIO_RANGE = (0.4, 0.6)
TEAR_ROUGH_RADIUS_RANGE = (0.05, 0.07)
SCRATCH_LENGTH_RANGE = (0.06, 0.16)
SCRATCH_WIDTH_RATIO_RANGE = (0.15, 0.25)
SCRATCH_ROUGH_RADIUS_RANGE = (0.08, 0.12)

_PATCH_OFFSET = 0.0005


def make_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_condition_materials(stage, base_color=(1.0, 0.25, 0.25)):
    normal_mat = make_material(stage, Sdf.Path("/World/Looks/ShoeNormal"), base_color, 0.55)
    tear_mat = make_material(
        stage, Sdf.Path("/World/Looks/TearPatch"),
        tuple(max(0, c - 0.85) for c in base_color), 0.9)
    scratch_mat = make_material(
        stage, Sdf.Path("/World/Looks/ScratchPatch"),
        tuple(max(0, c - 0.85) for c in base_color), 0.9)
    return {"normal": normal_mat, "tear": tear_mat, "scratch": scratch_mat}


def cache_mesh_topology(mesh):
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


def local_tangent_frame(centroids, seed, rng, rough_radius_range, ref_normal=None):
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


def pick_tear_placement(topo, rng):
    """찢어진 자리를 하나 고른다(위치만 - 실제로 face를 잘라내진 않는다).

    export_damaged_shoe_usd.py의 cut_faces_near_random_point와 같은 위치
    선정 로직이지만, 메시 토폴로지를 건드리지 않는다 - 라이브 스테이지에서
    재사용되는(신발이 활성화/비활성화를 반복하는) 프림은 한 번 잘라내면
    되돌릴 수 없어서, 대신 TearPatch 오버레이 메시만 그 위치로 옮겨서
    tear처럼 보이게 한다(scratch와 같은 방식)."""
    centroids = topo["centroids"]
    n_faces = len(centroids)
    seed = rng.randrange(n_faces)

    axis1, axis2, normal, _, scale = local_tangent_frame(
        centroids, seed, rng, rough_radius_range=TEAR_ROUGH_RADIUS_RANGE,
        ref_normal=topo["face_normals"][seed])

    rx = scale * TEAR_RX_VALUE
    ry = rx * rng.uniform(*TEAR_RY_RATIO_RANGE)
    return centroids[seed], axis1, axis2, normal, rx, ry


def cut_faces_near_random_point(mesh, topo, rng):
    """실제로 face를 잘라내는 파괴적 버전 - 한 번 쓰고 버리는 파일 export
    용도(export_damaged_shoe_usd.py)로만 쓴다. 재사용되는 라이브 프림에는
    쓰지 말 것(pick_tear_placement를 대신 쓴다)."""
    counts, indices, centroids = topo["counts"], topo["indices"], topo["centroids"]
    center, axis1, axis2, normal, rx, ry = pick_tear_placement(topo, rng)

    rel = centroids - center
    proj1 = rel @ axis1
    proj2 = rel @ axis2

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

    return center, axis1, axis2, normal, rx, ry


def pick_scratch_placement(topo, rng):
    centroids = topo["centroids"]
    n_faces = len(centroids)
    seed = rng.randrange(n_faces)

    axis1, axis2, normal, _, scale = local_tangent_frame(
        centroids, seed, rng, rough_radius_range=SCRATCH_ROUGH_RADIUS_RANGE,
        ref_normal=topo["face_normals"][seed])
    angle = rng.uniform(0, 2 * math.pi)
    dir_axis = axis1 * math.cos(angle) + axis2 * math.sin(angle)
    side_axis = -axis1 * math.sin(angle) + axis2 * math.cos(angle)

    length = scale * rng.uniform(*SCRATCH_LENGTH_RANGE)
    width = length * rng.uniform(*SCRATCH_WIDTH_RATIO_RANGE)
    return centroids[seed], dir_axis, side_axis, normal, length, width


def make_ellipse_patch(stage, path, sides=16):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * (sides + 1))
    counts = [3] * sides
    indices = []
    for i in range(sides):
        indices += [0, 1 + i, 1 + (i + 1) % sides]
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    return mesh


def update_ellipse_patch(mesh, center, axis1, axis2, normal, rx, ry, sides=16):
    base = center + normal * _PATCH_OFFSET
    points = [Gf.Vec3f(*base)]
    for i in range(sides):
        theta = 2 * math.pi * i / sides
        p = base + axis1 * (rx * math.cos(theta)) + axis2 * (ry * math.sin(theta))
        points.append(Gf.Vec3f(*p))
    mesh.GetPointsAttr().Set(points)


def make_rect_patch(stage, path):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0)] * 4)
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    return mesh


def update_rect_patch(mesh, center, dir_axis, side_axis, normal, length, width):
    base = center + normal * _PATCH_OFFSET
    half_l, half_w = length / 2, width / 2
    corners = [
        base - dir_axis * half_l - side_axis * half_w,
        base + dir_axis * half_l - side_axis * half_w,
        base + dir_axis * half_l + side_axis * half_w,
        base - dir_axis * half_l + side_axis * half_w,
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in corners])
