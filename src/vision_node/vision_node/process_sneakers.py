import bpy
import mathutils

FBX_PATH = "/home/woogi/doosan_pjt3/assets/shoes_raw/sneakers.fbx"

# 1) 씬 정리
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for m in list(bpy.data.meshes):
    bpy.data.meshes.remove(m)
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m)

# 2) FBX 임포트
bpy.ops.import_scene.fbx(filepath=FBX_PATH)
print(f"Imported: {[o.name for o in bpy.data.objects if o.type=='MESH']}")

# 3) Separate By Loose Parts
imported = [o for o in bpy.data.objects if o.type=='MESH']
obj = imported[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.separate(type='LOOSE')
bpy.ops.object.mode_set(mode='OBJECT')

# 4) X 좌표 기준 좌우 분류
left_objs, right_objs = [], []
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    lc = sum((mathutils.Vector(c) for c in o.bound_box), mathutils.Vector()) / 8
    wc = o.matrix_world @ lc
    (left_objs if wc.x < 0 else right_objs).append(o)
print(f"Left: {len(left_objs)}, Right: {len(right_objs)}")

# 5) Join
def join_group(objs, new_name):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    bpy.context.active_object.name = new_name
    return bpy.context.active_object

sneaker_l = join_group(left_objs, "Sneaker_L")
sneaker_r = join_group(right_objs, "Sneaker_R")

# 6) Origin + Transform apply
for o in [sneaker_l, sneaker_r]:
    bpy.ops.object.select_all(action='DESELECT')
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# 7) 이름 스왑 (실제 방향에 맞게)
sneaker_l.name = "temp"
sneaker_r.name = "Sneaker_L"
sneaker_l.name = "Sneaker_R"

print("Sneakers reprocessed.")
