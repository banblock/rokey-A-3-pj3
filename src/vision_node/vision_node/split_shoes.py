import bpy
import mathutils

# 씬의 모든 mesh 오브젝트를 X 좌표 기준으로 좌/우 분류
left_objs = []
right_objs = []

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    local_center = sum((mathutils.Vector(c) for c in obj.bound_box), mathutils.Vector()) / 8
    world_center = obj.matrix_world @ local_center
    if world_center.x < 0:
        left_objs.append(obj)
    else:
        right_objs.append(obj)

print(f"Left: {len(left_objs)}, Right: {len(right_objs)}")

# 왼쪽 조각들 join
bpy.ops.object.select_all(action='DESELECT')
for obj in left_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = left_objs[0]
bpy.ops.object.join()
bpy.context.active_object.name = "Sneaker_L"

# 오른쪽 조각들 join
bpy.ops.object.select_all(action='DESELECT')
for obj in right_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = right_objs[0]
bpy.ops.object.join()
bpy.context.active_object.name = "Sneaker_R"

print("Done")
