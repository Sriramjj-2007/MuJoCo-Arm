import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "models/two_link_arm.xml"
)
data = mujoco.MjData(model)

# Target joint angles, in radians
target = np.array([0.7, 0.9])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = target[0]
        data.ctrl[1] = target[1]

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)