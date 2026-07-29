"""
tear/scratch 손상을 적용한 신발(sneaker_240.usd 켤레) 하나를 잘라내서 별도의 .usd
파일로 저장한다. isaac_shoe_sdg.py는 카메라로 렌더링한 "이미지"만 저장하고 신발
지오메트리 자체는 저장하지 않아서, 손상 적용된 신발 애셋 자체가 필요할 때는 이
스크립트를 쓴다. 컨베이어/카메라/ROS 아무것도 안 건드리고 신발 하나만 빠르게 처리하므로
isaac_shoe_sdg.py를 GUI로 띄워둘 필요가 없다.

tear/scratch를 실제로 만드는 로직은 shoe_damage.py로 옮겨서, simulation_node.py가
라이브 스테이지에서 신발을 활성화할 때도 같은 로직을 재사용한다(단, 그쪽은 프림을
반복 재사용하므로 face를 실제로 잘라내지 않고 오버레이 패치만 씀 - shoe_damage.py의
pick_tear_placement 참고).

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

import os
import random
import sys

from pxr import Usd, UsdGeom, UsdShade

sys.path.insert(0, os.path.expanduser("~/cobot3_ws/isaacpjt/mainsim"))
from shoe_damage import (
    cache_mesh_topology,
    cut_faces_near_random_point,
    create_condition_materials,
    make_ellipse_patch,
    make_rect_patch,
    pick_scratch_placement,
    update_ellipse_patch,
    update_rect_patch,
)

SHOE_ASSET_URL = "/home/rokey/Downloads/sneaker_240.usd"

rng = random.Random(args.seed)


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
        topo = cache_mesh_topology(m)

        if damage_type == "tear":
            tear_patch = make_ellipse_patch(stage, m.GetPath().AppendChild("TearPatch"))
            UsdShade.MaterialBindingAPI(tear_patch).Bind(materials["tear"])
            center, axis1, axis2, normal, rx, ry = cut_faces_near_random_point(m, topo, rng)
            update_ellipse_patch(tear_patch, center, axis1, axis2, normal, rx, ry)
        else:
            scratch_patch = make_rect_patch(stage, m.GetPath().AppendChild("ScratchPatch"))
            UsdShade.MaterialBindingAPI(scratch_patch).Bind(materials["scratch"])
            center, dir_axis, side_axis, normal, length, width = pick_scratch_placement(topo, rng)
            update_rect_patch(scratch_patch, center, dir_axis, side_axis, normal, length, width)

    stage.GetRootLayer().Save()
    print(f"[done] damage={damage_type} -> {args.out}")

    simulation_app.close()


main()
