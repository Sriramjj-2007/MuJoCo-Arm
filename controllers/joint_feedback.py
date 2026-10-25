
import time
import numpy as np
import mujoco
import mujoco.viewer

# Load the two-link arm
model = mujoco.MjModel.from_xml_path(
    "models/two_link_arm.xml"
)
data = mujoco.MjData(model)

# Target positions in degrees
targets_deg = [
    [-120, 30],
    [-50, 45],
    [-60, 100],
    [-120, 30],
]

target_index = 0
last_switch = time.monotonic()
last_print = 0.0

print("MuJoCo Joint Feedback")
print("-" * 72)
print(
    f"{'Target 1':>10} {'Actual 1':>10} "
    f"{'Error 1':>10} {'Target 2':>10} "
    f"{'Actual 2':>10} {'Error 2':>10}"
)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Change target every 3 seconds
        now = time.monotonic()
        if now - last_switch >= 3:
            target_index = (
                target_index + 1
            ) % len(targets_deg)
            last_switch = now

        # Convert target angles to radians
        target_rad = np.deg2rad(
            targets_deg[target_index]
        )

        # Send commands to the position actuators
        data.ctrl[0] = target_rad[0]
        data.ctrl[1] = target_rad[1]

        # Advance the physics simulation
        mujoco.mj_step(model, data)

        # Read actual joint angles from the simulation
        actual_rad = data.qpos[:2].copy()

        # Convert to degrees
        target_deg = np.rad2deg(data.ctrl[:2])
        actual_deg = np.rad2deg(actual_rad)

        # Calculate tracking error
        error_deg = target_deg - actual_deg

        # Print feedback 5 times per second
        if now - last_print >= 0.2:
            print(
                f"{target_deg[0]:10.2f} "
                f"{actual_deg[0]:10.2f} "
                f"{error_deg[0]:10.2f} "
                f"{target_deg[1]:10.2f} "
                f"{actual_deg[1]:10.2f} "
                f"{error_deg[1]:10.2f}",
                flush=True,
            )
            last_print = now

        viewer.sync()
        time.sleep(model.opt.timestep)
