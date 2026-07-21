import bpy
import bmesh
import mathutils
import random

# --- 설정 ---
TARGET_OBJ = "Sneaker_R"   # 구멍 뚫을 신발 ("Sneaker_L" or "Sneaker_R")
HOLE_RADIUS = 0.015        # 구멍 크기 (미터 단위). 0.015 = 1.5cm 정도
SEED = 42                  # 랜덤 시드. 바꾸면 구멍 위치 바뀜

# --- 실행 ---
random.seed(SEED)
obj = bpy.data.objects[TARGET_OBJ]

# Edit mode 진입 + bmesh 세팅
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

# 신발 윗면(Z 좌표 상위 60%) 정점 중 하나를 무작위 선택 → 구멍 중심
verts_upper = [v for v in bm.verts if v.co.z > 0]
top_third_z = sorted([v.co.z for v in verts_upper])[int(len(verts_upper) * 0.7)]
candidates = [v for v in verts_upper if v.co.z >= top_third_z]
center_vert = random.choice(candidates)
center = center_vert.co.copy()

print(f"Hole center (local): {center}")

# 반경 안의 face 골라내기
faces_to_delete = []
for f in bm.faces:
    face_center = sum((v.co for v in f.verts), mathutils.Vector()) / len(f.verts)
    if (face_center - center).length < HOLE_RADIUS:
        faces_to_delete.append(f)

print(f"Deleting {len(faces_to_delete)} faces")

# 지우기
bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
bmesh.update_edit_mesh(obj.data)

# Object mode 복귀
bpy.ops.object.mode_set(mode='OBJECT')
print("Done")
