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

world.reset()
count = 0
was_playing = False

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    
    is_playing = world.is_playing()
    
    if is_playing and not was_playing:
        count = 0
        print("시작 버튼 눌림: count 0으로 초기화")

    if is_playing:
        count += 1
        
        if (count % 100) == 0:
            print(count)
            if (count % 300) == 0:
                cube_prim_r.set_world_pose(position=np.array([0.0, 0.0, 1.0]))
                print('큐브 이동')
                
    was_playing = is_playing

simulation_app.close()