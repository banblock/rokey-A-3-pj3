from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import time
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

cube_prim_r = DynamicCuboid(                              # 4. Prim
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0, 0.5]),
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene
world.scene.add(cube_prim_r)
step_count = 0

while simulation_app.is_running():
    world.step(render=True)
    time.sleep(0.01)
    step_count += 100
    if step_count % 100 == 0:
        print(f"step:{step_count}")
    elif step_count % 300 == 0:
        cube_prim_r.set_world_pose(position=np.array([0.0, 0.0, 0.5]))
    # 스탑 후 시작 시 step_count = 0으로 리셋 필요

world.reset()

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)

simulation_app.close()