import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "models/two_link_arm.xml"
)
data = mujoco.MjData(model)

dt = model.opt.timestep

# Initial gains: tune them experimentally
kp = np.array([30.0, 15.0])
ki = np.array([0.0, 0.0])
kd = np.array([5.0, 2.5])

target = np.array([0.7, -0.9])
integral = np.zeros(2)
previous_error = np.zeros(2)

torque_limit = np.array([10.0, 5.0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Both joints are hinges; qpos and qvel are radians
        q = data.qpos[:2].copy()
        qd = data.qvel[:2].copy()

        error = target - q
        integral += error * dt

        derivative = -qd

        torque = (
            kp * error
            + ki * integral
            + kd * derivative
        )

        torque = np.clip(
            torque, -torque_limit, torque_limit
        )

        data.ctrl[:2] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)