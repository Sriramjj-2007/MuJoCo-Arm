import numpy as np
import matplotlib.pyplot as plt

# Dimensions and joint limits from the supplied MuJoCo XML
L1 = 0.5
L2 = 0.4
BASE_Z = 0.58

base_angles = np.deg2rad(np.linspace(-45, 45, 41))
shoulder_angles = np.deg2rad(np.linspace(-123, -35, 61))
elbow_angles = np.deg2rad(np.linspace(25, 116, 61))

B, Q1, Q2 = np.meshgrid(
    base_angles, shoulder_angles, elbow_angles, indexing="ij"
)

# Endpoint position for two +Y-axis hinges, followed by base Z rotation.
r = L1 * np.cos(Q1) + L2 * np.cos(Q1 + Q2)
X = r * np.cos(B)
Y = r * np.sin(B)
Z = BASE_Z - L1 * np.sin(Q1) - L2 * np.sin(Q1 + Q2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=0.5, alpha=0.12)

# Show the base axis and shoulder height
ax.plot([0, 0], [0, 0], [0.0, BASE_Z], linewidth=2)
ax.scatter([0], [0], [BASE_Z], s=35)

ax.set_title("3D End-Effector Workspace (Joint-Limit Sample)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=24, azim=-55)
plt.tight_layout()
plt.show()
