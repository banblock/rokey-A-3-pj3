"""Isaac Sim 5.1 standalone sequence for stage_v10(1).usd.

Sequence
1. Open USD and start simulation.
2. Activate the prim whose name is ``sneakers``.
3. After 1.5 seconds, set conveyor ``variables:Velocity`` to 0.0.
4. Send a ROS 2 ``std_srvs/srv/Trigger`` request to the vision service.
5. Five seconds after the stop/service step, set velocity to 1.0.

Run with Isaac Sim's Python, not the system Python.
"""

import os
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
USD_PATH = Path(os.environ.get("USD_PATH", "/mnt/data/stage_v10(1).usd"))
VISION_SERVICE_NAME = os.environ.get("VISION_SERVICE_NAME", "/vision/inspect")
STOP_DELAY_SEC = float(os.environ.get("STOP_DELAY_SEC", "1.5"))
RESTART_DELAY_SEC = float(os.environ.get("RESTART_DELAY_SEC", "5.0"))
START_VELOCITY = float(os.environ.get("START_VELOCITY", "1.0"))
STOP_VELOCITY = 0.0

# ``headless=False`` opens the Isaac Sim GUI.
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# ROS_DOMAIN_ID can still be overridden from the shell.
os.environ.setdefault("ROS_DISTRO", "humble")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": HEADLESS})

import carb
import omni.timeline
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Usd

# Enable ROS 2 before importing rclpy.
enable_extension("isaacsim.ros2.bridge")
for _ in range(10):
    simulation_app.update()

import rclpy
from std_srvs.srv import Trigger


def log(message: str) -> None:
    print(f"[stage_sequence] {message}", flush=True)


def open_stage(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")

    context = omni.usd.get_context()
    log(f"Opening USD: {path}")
    if not context.open_stage(str(path)):
        raise RuntimeError(f"Failed to open USD: {path}")

    # Let references/payloads and Action Graph data finish loading.
    for _ in range(120):
        simulation_app.update()

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available after opening the file.")
    return stage


def find_sneakers_prim(stage: Usd.Stage) -> Usd.Prim:
    candidates = []
    for prim in stage.TraverseAll():
        if prim.GetName().lower() == "sneakers":
            candidates.append(prim)

    if not candidates:
        raise RuntimeError("Could not find a prim named 'sneakers' under /World.")

    # Prefer /World/sneakers when it exists.
    candidates.sort(key=lambda p: (str(p.GetPath()) != "/World/sneakers", len(str(p.GetPath()))))
    return candidates[0]


def activate_sneakers(stage: Usd.Stage) -> str:
    prim = find_sneakers_prim(stage)
    prim.SetActive(True)
    log(f"Activated sneakers prim: {prim.GetPath()}")
    return str(prim.GetPath())


def find_velocity_attributes(stage: Usd.Stage):
    """Find conveyor graph velocity attributes dynamically.

    Primary match: attributes named ``variables:Velocity`` below a prim path
    containing ``Conveyors``. A fallback accepts any attribute ending in
    ``:Velocity`` below a conveyor-related path.
    """
    primary = []
    fallback = []

    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if "conveyor" not in path.lower():
            continue

        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name == "variables:Velocity":
                primary.append(attr)
            elif name.lower().endswith(":velocity") or name.lower() == "velocity":
                fallback.append(attr)

    attrs = primary if primary else fallback
    if not attrs:
        raise RuntimeError(
            "No conveyor Velocity attribute was found. Expected an Action Graph "
            "attribute such as 'variables:Velocity' below a Conveyors prim."
        )
    return attrs


def set_conveyor_velocity(attributes, value: float) -> None:
    failures = []
    for attr in attributes:
        try:
            attr.Set(float(value))
            log(f"Set {attr.GetPath()} = {value}")
        except Exception as exc:  # keep trying the remaining conveyor graphs
            failures.append(f"{attr.GetPath()}: {exc}")

    if len(failures) == len(attributes):
        raise RuntimeError("Failed to set every conveyor velocity attribute: " + "; ".join(failures))
    if failures:
        log("Some velocity attributes could not be set: " + "; ".join(failures))


def call_vision_service(node) -> None:
    client = node.create_client(Trigger, VISION_SERVICE_NAME)
    log(f"Waiting for vision service: {VISION_SERVICE_NAME}")

    # Keep simulation responsive while waiting, but do not wait forever.
    deadline = time.monotonic() + 10.0
    while simulation_app.is_running() and not client.wait_for_service(timeout_sec=0.0):
        if time.monotonic() >= deadline:
            log(f"Vision service not available after 10 s: {VISION_SERVICE_NAME}")
            node.destroy_client(client)
            return
        rclpy.spin_once(node, timeout_sec=0.0)
        simulation_app.update()
        time.sleep(0.01)

    future = client.call_async(Trigger.Request())
    log("Vision service request sent.")

    # Do not block the sequence until the vision result; just briefly process
    # callbacks and print the response whenever it arrives during the 5 s wait.
    node._vision_future = future
    node._vision_client = client


def process_ros(node) -> None:
    rclpy.spin_once(node, timeout_sec=0.0)
    future = getattr(node, "_vision_future", None)
    if future is not None and future.done():
        try:
            response = future.result()
            log(f"Vision response: success={response.success}, message={response.message!r}")
        except Exception as exc:
            log(f"Vision service call failed: {exc}")
        client = getattr(node, "_vision_client", None)
        if client is not None:
            node.destroy_client(client)
        node._vision_future = None
        node._vision_client = None


def run_for_seconds(node, seconds: float) -> None:
    """Advance Isaac Sim for a wall-clock duration while servicing ROS callbacks."""
    end_time = time.monotonic() + seconds
    while simulation_app.is_running() and time.monotonic() < end_time:
        simulation_app.update()
        process_ros(node)


def main() -> int:
    node = None
    timeline = omni.timeline.get_timeline_interface()

    try:
        stage = open_stage(USD_PATH)
        activate_sneakers(stage)
        velocity_attributes = find_velocity_attributes(stage)

        log("Detected conveyor velocity attributes:")
        for attr in velocity_attributes:
            log(f"  - {attr.GetPath()} (current={attr.Get()})")

        if not rclpy.ok():
            rclpy.init(args=None)
        node = rclpy.create_node("isaac_stage_sequence")

        # Apply the initial running speed before playback.
        set_conveyor_velocity(velocity_attributes, START_VELOCITY)
        timeline.play()
        log("Simulation playback started.")

        run_for_seconds(node, STOP_DELAY_SEC)
        set_conveyor_velocity(velocity_attributes, STOP_VELOCITY)
        log(f"Conveyor stopped after {STOP_DELAY_SEC:.1f} s.")

        call_vision_service(node)

        run_for_seconds(node, RESTART_DELAY_SEC)
        set_conveyor_velocity(velocity_attributes, START_VELOCITY)
        log(f"Conveyor restarted at velocity {START_VELOCITY}.")

        # Keep Isaac Sim open after completing the requested sequence.
        while simulation_app.is_running():
            simulation_app.update()
            process_ros(node)

        return 0

    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    except Exception as exc:
        carb.log_error(f"[stage_sequence] {exc}")
        log(f"ERROR: {exc}")
        return 1
    finally:
        try:
            timeline.stop()
        except Exception:
            pass
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())