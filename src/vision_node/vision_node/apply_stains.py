import bpy
import random
from pathlib import Path

# --- 설정 ---
TARGET_OBJ = "Sneaker_R"                          # 얼룩 붙일 신발
STAINS_DIR = "/home/woogi/doosan_pjt3/data_generation/stains"
NUM_STAINS = 3                                     # 붙일 얼룩 개수
BASE_COLOR = (0.85, 0.15, 0.15, 1.0)              # 신발 원래 색 (빨강)
SEED = 7

# --- 실행 ---
random.seed(SEED)
obj = bpy.data.objects[TARGET_OBJ]
mat = obj.data.materials[0]
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# 기존 이미지 노드/합성 노드 다 지우고 시작 (재실행 안전)
for n in list(nodes):
    if n.type in {'TEX_IMAGE', 'MIX', 'MIX_RGB', 'TEX_COORD', 'MAPPING'}:
        nodes.remove(n)

bsdf = nodes.get("Principled BSDF")

# 기본 색 노드 (RGB)
rgb_node = nodes.new('ShaderNodeRGB')
rgb_node.outputs[0].default_value = BASE_COLOR
rgb_node.location = (-800, 200)
current_color_output = rgb_node.outputs[0]

# 얼룩 이미지 무작위로 골라 순서대로 layer 합성
stain_files = list(Path(STAINS_DIR).glob("*.png"))
chosen = random.sample(stain_files, k=NUM_STAINS)
print(f"Applying: {[f.name for f in chosen]}")

for i, stain_path in enumerate(chosen):
    # 이미지 텍스처 노드
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(stain_path))
    tex.image.alpha_mode = 'STRAIGHT'
    tex.location = (-600 + i*50, -200 - i*100)

    # UV 스케일/오프셋 랜덤 (얼룩 위치 다르게)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-800 + i*50, -200 - i*100)
    mapping.inputs['Location'].default_value = (
        random.uniform(-0.5, 0.5),
        random.uniform(-0.5, 0.5),
        0
    )
    scale = random.uniform(1.5, 3.0)
    mapping.inputs['Scale'].default_value = (scale, scale, 1)

    uv = nodes.new('ShaderNodeTexCoord')
    uv.location = (-1000 + i*50, -200 - i*100)
    links.new(uv.outputs['UV'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex.inputs['Vector'])

    # Mix RGB로 알파 기반 블렌드
    mix = nodes.new('ShaderNodeMixRGB')
    mix.location = (-300 + i*200, 100 - i*100)
    mix.blend_type = 'MIX'
    links.new(tex.outputs['Alpha'], mix.inputs['Fac'])
    links.new(current_color_output, mix.inputs['Color1'])
    links.new(tex.outputs['Color'], mix.inputs['Color2'])

    current_color_output = mix.outputs['Color']

# 최종 색을 BSDF Base Color에 연결
links.new(current_color_output, bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 0.6

print("Done. Material Preview로 확인하세요.")
