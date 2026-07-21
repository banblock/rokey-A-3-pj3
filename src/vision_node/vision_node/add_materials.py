import bpy

# 스니커즈 3색상 정의 (R, G, B, Alpha) — 0~1 범위
# 얼룩 팔레트(갈/검/노/검붉)와 명확히 구분되는 원색 계열
SNEAKER_COLORS = {
    "sneaker_red":   (0.85, 0.15, 0.15, 1.0),
    "sneaker_green": (0.20, 0.70, 0.25, 1.0),
    "sneaker_blue":  (0.15, 0.35, 0.85, 1.0),
}

def make_material(name, rgba):
    """이름의 머티리얼이 없으면 새로 생성, 있으면 색만 갱신."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    # Principled BSDF의 Base Color를 설정
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        # 신발은 대체로 매트하므로 Roughness 살짝 올림
        bsdf.inputs["Roughness"].default_value = 0.55
    return mat

# 3개 머티리얼 생성
materials = {name: make_material(name, rgba) for name, rgba in SNEAKER_COLORS.items()}
print(f"Created materials: {list(materials.keys())}")

# 두 신발에 기본 머티리얼(빨강)을 붙여둠
# — Isaac Sim에서 나중에 3개 중 하나로 랜덤 스왑하게 됨
default_mat = materials["sneaker_red"]

for obj_name in ["Sneaker_L", "Sneaker_R"]:
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"WARN: {obj_name} not found")
        continue
    # 기존 머티리얼 슬롯 다 지우고 새로 붙임
    obj.data.materials.clear()
    obj.data.materials.append(default_mat)
    print(f"{obj_name}: material set to {default_mat.name}")

print("Done. 뷰포트 상단 오른쪽의 구슬 아이콘들 중 'Material Preview' 눌러서 색 확인.")
