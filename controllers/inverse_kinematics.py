
import time
import numpy as np
import mujoco
import mujoco.viewer

# --------------------------------------------------
# 1. LOAD ROBOT MODEL
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path(
    "models/two_link_arm.xml"
)
data = mujoco.MjData(model)

# Link lengths in metres
L1 = 0.5
L2 = 0.4

# Shoulder height in world coordinates
BASE_Z = 0.58

# Joint limits in radians: [shoulder, elbow]
JOINT_MIN = np.deg2rad([-123, 25])
JOINT_MAX = np.deg2rad([-35, 116])

# Base rotation limits
BASE_MIN = np.deg2rad(-180)
BASE_MAX = np.deg2rad(180)

# --------------------------------------------------
# 2. INVERSE KINEMATICS
# --------------------------------------------------

def inverse_kinematics(x, y, z):
    """
    Calculate [base, shoulder, elbow] angles in radians.

    Target coordinates are expressed in world coordinates.
    """

    # Base rotation about the vertical Z-axis
    base_angle = np.arctan2(y, x)

    if not BASE_MIN <= base_angle <= BASE_MAX:
        raise ValueError("Base rotation exceeds joint limits.")

    # Convert 3D target to a radial X-Z plane target
    radial_distance = np.hypot(x, y)
    px = radial_distance
    py = BASE_Z - z

    distance_sq = px**2 + py**2

    # Cosine rule for the elbow angle
    c2 = (
        distance_sq - L1**2 - L2**2
    ) / (2 * L1 * L2)

    if c2 < -1.0 - 1e-9 or c2 > 1.0 + 1e-9:
        raise ValueError(
            "Target is outside the arm's reachable workspace."
        )

    c2 = np.clip(c2, -1.0, 1.0)

    candidates = []

    # Try both mathematical elbow configurations
    for sign in (1, -1):
        theta2 = sign * np.arccos(c2)

        theta1 = (
            np.arctan2(py, px)
            - np.arctan2(
                L2 * np.sin(theta2),
                L1 + L2 * np.cos(theta2),
            )
        )

        # Convert the standard planar solution to MuJoCo
        # angles for a +Y hinge axis.
        q1 = -theta1
        q2 = -theta2

        q = np.array([q1, q2])

        # Check shoulder and elbow limits independently
        if (
            np.all(q >= JOINT_MIN)
            and np.all(q <= JOINT_MAX)
        ):
            candidates.append(q)

    if not candidates:
        raise ValueError(
            "Target is geometrically reachable, but no solution "
            "satisfies the shoulder and elbow limits."
        )

    # Select the solution closest to the current arm pose.
    current = data.qpos[1:3].copy()

    arm_angles = min(
        candidates,
        key=lambda q: np.linalg.norm(q - current),
    )

    return np.array([
        base_angle,
        arm_angles[0],
        arm_angles[1],
    ])


# --------------------------------------------------
# 3. FORWARD KINEMATICS
# --------------------------------------------------

def forward_kinematics(q):
    """
    Calculate endpoint [x, y, z] from joint angles.

    q = [base rotation, shoulder, elbow]
    """

    base_angle, q1, q2 = q

    # Radial distance from the base axis
    r = (
        L1 * np.cos(q1)
        + L2 * np.cos(q1 + q2)
    )

    # Vertical endpoint coordinate
    z = (
        BASE_Z
        - L1 * np.sin(q1)
        - L2 * np.sin(q1 + q2)
    )

    # Rotate radial distance into world X-Y coordinates
    x = r * np.cos(base_angle)
    y = r * np.sin(base_angle)

    return np.array([x, y, z])


# --------------------------------------------------
# 4. TARGET COORDINATES
# --------------------------------------------------

# Example target, in metres
target_x = 0.0
target_y = 0.65
target_z = 0.10

target_xyz = np.array([
    target_x,
    target_y,
    target_z,
])

# Calculate target joint angles
target_angles = inverse_kinematics(
    target_x,
    target_y,
    target_z,
)

print("\nInverse Kinematics Solution")
print("---------------------------")
print(
    f"Target XYZ: "
    f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f}) m"
)
print(f"Base rotation: {np.rad2deg(target_angles[0]):.2f} deg")
print(f"Shoulder:      {np.rad2deg(target_angles[1]):.2f} deg")
print(f"Elbow:         {np.rad2deg(target_angles[2]):.2f} deg")


# --------------------------------------------------
# 5. SIMULATION
# --------------------------------------------------

# Find joints and actuators by name instead of assuming
# their order in the XML.
joint_names = [
    "base_rotation",
    "joint1",
    "joint2",
]

actuator_names = [
    "base_servo",
    "shoulder_servo",
    "elbow_servo",
]

joint_qpos_addresses = []
actuator_ids = []

for name in joint_names:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )

    if joint_id == -1:
        raise ValueError(f"Joint '{name}' not found in XML.")

    joint_qpos_addresses.append(
        model.jnt_qposadr[joint_id]
    )

for name in actuator_names:
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    )

    if actuator_id == -1:
        raise ValueError(f"Actuator '{name}' not found in XML.")

    actuator_ids.append(actuator_id)


print("\nStarting MuJoCo viewer...")
print("Close the viewer window to stop the simulation.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    last_print = 0.0

    while viewer.is_running():

        # Command each position actuator
        for i, actuator_id in enumerate(actuator_ids):
            data.ctrl[actuator_id] = target_angles[i]

        # Advance simulation
        mujoco.mj_step(model, data)

        # Read actual joint positions in the correct order
        actual_angles = np.array([
            data.qpos[address]
            for address in joint_qpos_addresses
        ])

        # Calculate endpoint position
        actual_xyz = forward_kinematics(actual_angles)

        # Update viewer
        viewer.sync()

        # Limit terminal output frequency
        now = time.time()

        if now - last_print >= 0.2:
            print(
                f"\rActual XYZ: "
                f"X={actual_xyz[0]:.3f} m, "
                f"Y={actual_xyz[1]:.3f} m, "
                f"Z={actual_xyz[2]:.3f} m  ",
                end="",
                flush=True,
            )
            last_print = now

        time.sleep(model.opt.timestep)

print("\nSimulation stopped.")
