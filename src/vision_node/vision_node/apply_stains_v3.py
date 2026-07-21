import bpy
import random

TARGET_OBJ = "Sneaker_R"
NUM_STAINS = 4
STAIN_SIZE = 0.02
BASE_COLOR = (0.85, 0.15, 0.15, 1.0)

STAIN_COLORS = [
    (0.35, 0.22, 0.10, 1.0),
    (0.05, 0.05, 0.08, 1.0),
    (0.65, 0.50, 0.15, 1.0),
    (0.25, 0.12, 0.10, 1.0),
]

SEED = 42

random.seed(SEED)
obj = bpy.data.objects[TARGET_OBJ]
mat = obj.data.materials[0]
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

for n in list(nodes):
    if n.type not in {'BSDF_PRINCIPLED', 'OUTPUT_MATERIAL'}:
        nodes.remove(n)

bsdf = nodes.get("Principled BSDF")

rgb = nodes.new('ShaderNodeRGB')
rgb.outputs[0].default_value = BASE_COLOR
rgb.location = (-1400, 0)
current = rgb.outputs[0]

geom = nodes.new('ShaderNodeTexCoord')
geom.location = (-1600, -300)

mesh = obj.data
verts = mesh.vertices
random_verts = random.sample(list(verts), NUM_STAINS)

for i, v in enumerate(random_verts):
    cx = v.co.x
    cy = v.co.y
    cz = v.co.z
    color = random.choice(STAIN_COLORS)
    print("stain", i, "pos", cx, cy, cz)

    subtract = nodes.new('ShaderNodeVectorMath')
    subtract.operation = 'SUBTRACT'
    subtract.location = (-1200, -300 - i*250)
    subtract.inputs[1].default_value = (cx, cy, cz)
    links.new(geom.outputs['Object'], subtract.inputs[0])

    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    length.location = (-1000, -300 - i*250)
    links.new(subtract.outputs['Vector'], length.inputs[0])

    less = nodes.new('ShaderNodeMath')
    less.operation = 'LESS_THAN'
    less.location = (-800, -300 - i*250)
    less.inputs[1].default_value = STAIN_SIZE
    links.new(length.outputs['Value'], less.inputs[0])

    stain_rgb = nodes.new('ShaderNodeRGB')
    stain_rgb.outputs[0].default_value = color
    stain_rgb.location = (-800, -100 - i*250)

    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.location = (-500 + i*250, -i*100)
    links.new(less.outputs['Value'], mix.inputs['Fac'])
    links.new(current, mix.inputs['Color1'])
    links.new(stain_rgb.outputs[0], mix.inputs['Color2'])

    current = mix.outputs['Color']

links.new(current, bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 0.6

print("Done")
