import pybullet as p
import pybullet_data
import numpy as np
import time

# ================= PyBullet init =================
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# ================= World =================
plane = p.loadURDF("plane.urdf")

table = p.loadURDF(
    "table/table.urdf",
    basePosition=[0.0, 0.0, 0.0],
    useFixedBase=True
)

cube = p.loadURDF(
    "cube_small.urdf",
    basePosition=[0.3, 0.0, 0.65],
    useFixedBase=False
)

# ================= Load Franka =================
robot = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[-0.5, 0.0, 0.5],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# ================= Constants =================
ARM_JOINTS = list(range(7))
FINGER_JOINTS = [9, 10]
EE_LINK = 11   # panda_hand

# ================= Ready pose =================
def set_ready_pose():
    q = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]
    for j, val in zip(ARM_JOINTS, q):
        p.resetJointState(robot, j, val)
    for j in FINGER_JOINTS:
        p.resetJointState(robot, j, 0.04)

set_ready_pose()

for _ in range(240):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

# ================= EE pose =================
def get_ee_pos():
    state = p.getLinkState(robot, EE_LINK, computeForwardKinematics=True)
    return np.array(state[0])

def get_ee_orn():
    state = p.getLinkState(robot, EE_LINK, computeForwardKinematics=True)
    return state[1]

fixed_orn = get_ee_orn()   # 固定姿態（非常重要）

print("Initial EE position:", get_ee_pos())

# ================= Cartesian sliders =================
ee_init = get_ee_pos()

slider_x = p.addUserDebugParameter("EE_X", ee_init[0] - 0.2, ee_init[0] + 0.4, ee_init[0])
slider_y = p.addUserDebugParameter("EE_Y", -0.3, 0.3, ee_init[1])
slider_z = p.addUserDebugParameter("EE_Z", 0.3, 1.0, ee_init[2])

gripper_slider = p.addUserDebugParameter("Gripper", 0.0, 0.04, 0.04)

# ================= Main loop =================
print("✅ Cartesian slider control active")
print("✅ Drag EE_X / EE_Y / EE_Z sliders")
print("✅ Close window to exit")

while p.isConnected():

    # --- Read Cartesian sliders ---
    target_pos = np.array([
        p.readUserDebugParameter(slider_x),
        p.readUserDebugParameter(slider_y),
        p.readUserDebugParameter(slider_z)
    ])

    # --- IK solve ---
    joint_targets = p.calculateInverseKinematics(
        robot,
        EE_LINK,
        target_pos,
        fixed_orn,
        maxNumIterations=200,
        residualThreshold=1e-4
    )

    # --- Apply joint control ---
    for j in ARM_JOINTS:
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=joint_targets[j],
            force=90
        )

    # --- Gripper ---
    grip = p.readUserDebugParameter(gripper_slider)
    for j in FINGER_JOINTS:
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=grip,
            force=40
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
