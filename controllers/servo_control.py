import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "models/two_link_arm.xml"
)
data = mujoco.MjData(model)

targets = [
    [0, 0],
    [30, 45],
    [60, 20],
    [0, 0],
]

target_index = 0
last_switch = time.monotonic()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if time.monotonic() - last_switch >= 2:
            target_index = (target_index + 1) % len(targets)
            last_switch = time.monotonic()

        target = np.deg2rad(targets[target_index])
        data.ctrl[0] = target[0]
        data.ctrl[1] = target[1]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)