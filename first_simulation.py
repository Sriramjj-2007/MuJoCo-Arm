import mujoco
import mujoco.viewer

model_xml = """
<mujoco model="first_robot">
    <option timestep="0.002"/>

    <worldbody>
        <light pos="0 0 3"/>

        <geom name="ground"
              type="plane"
              size="2 2 0.1"
              rgba="0.8 0.8 0.8 1"/>

        <body name="robot_link" pos="0 0 1">
            <freejoint/>

            <geom name="link"
                  type="capsule"
                  fromto="0 0 0 0 0 0.5"
                  size="0.08"
                  mass="1"
                  rgba="0.1 0.5 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

print("Model loaded successfully!")
print("Simulation timestep:", model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
