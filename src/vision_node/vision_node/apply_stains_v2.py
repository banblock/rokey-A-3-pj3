import bpy
import random

TARGET_OBJ = "Sneaker_R"
NUM_STAINS = 4                    # 얼룩 개수
STAIN_SIZE = 0.08                 # 얼룩 크기 (UV 공간, 0.05~0.15 추천)
BASE_COLOR = (0.85, 0.15, 0.15, 1.0)   # 빨강

# 얼룩 색 팔레트 (얼룩 종류별)
STAIN_COLORS = [
    (0.35, 0.22, 0.10, 1.0),  # mud
    (0.05, 0.05, 0.08, 1.0),  # oil
    (0.65, 0.50, 0.15, 1.0),  # food_yellow
    (0.25, 0.12, 0.10, 1.0),  # food_dark
]

SEED = 42

random.seed(SEED)
obj = bpy.data.objects[TARGET_OBJ]
mat = obj.data.materials[0]
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# 기존 이미지/합성 노드 다 지움
for n in list(nodes):
    if n.type in {'TEX_IMAGE', 'MIX', 'MIX_RGB', 'TEX_COORD', 'MAPPING',
                   'RGB', 'GRADIENT_TEX', 'MATH', 'VECTOR_MATH', 'GROUP'}:
        nodes.remove(n)

bsdf = nodes.get("Principled BSDF")

# 시작 색: 기본 빨강
rgb = nodes.new('ShaderNodeRGB')
rgb.outputs[0].default_value = BASE_COLOR
rgb.location = (-1400, 0)
current = rgb.outputs[0]

# UV 좌표 소스
uvmap = nodes.new('ShaderNodeTexCoord')
uvmap.location = (-1600, -300)

# 얼룩 개수만큼 원형 마스크 얹기
for i in range(NUM_STAINS):
    # 얼룩 중심 위치 (UV 공간)
    cx = random.uniform(0.15, 0.85)
    cy = random.uniform(0.15, 0.85)
    color = random.choice(STAIN_COLORS)

    # UV에서 (cx, cy)까지 거리 계산: Vector Math > Distance
    subtract = nodes.new('ShaderNodeVectorMath')
    subtract.operation = 'SUBTRACT'
    subtract.location = (-1200, -300 - i*250)
    subtract.inputs[1].default_value = (cx, cy, 0)
    links.new(uvmap.outputs['UV'], subtract.inputs[0])

    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    length.location = (-1000, -300 - i*250)
    links.new(subtract.outputs['Vector'], length.inputs[0])

    # 거리 < STAIN_SIZE 안쪽이면 얼룩, 밖이면 원래색
    # 부드러운 경계 위해 Smooth step 유사: (size - dist) / size, clamp 0~1
    less = nodes.new('ShaderNodeMath')
    less.operation = 'LESS_THAN'
    less.location = (-800, -300 - i*250)
    less.inputs[1].default_value = STAIN_SIZE
    links.new(length.outputs['Value'], less.inputs[0])

    # 얼룩 색 노드
    stain_rgb = nodes.new('ShaderNodeRGB')
    stain_rgb.outputs[0].default_value = color
    stain_rgb.location = (-800, -100 - i*250)

    # Mix: 마스크 값을 factor로 씀
    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.location = (-500 + i*250, -i*100)
    links.new(less.outputs['Value'], mix.inputs['Fac'])
    links.new(current, mix.inputs['Color1'])
    links.new(stain_rgb.outputs[0], mix.inputs['Color2'])

    current = mix.outputs['Color']

links.new(current, bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 0.6

print(f"Applied {NUM_STAINS} stain spots to {TARGET_OBJ}")
