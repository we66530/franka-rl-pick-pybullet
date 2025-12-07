from stable_baselines3 import PPO
import time
import numpy as np
import cv2
import pybullet as p
import math

from franka_rl_pick_constraint import FrankaPickEnv

# =====================================================
# Environment & Model
# =====================================================
env = FrankaPickEnv(render=True, max_steps=300)
model = PPO.load("ppo_franka_pick", env=env)

# =====================================================
# Parameters
# =====================================================
GRASP_DIST_THRESH = 0.05
TARGET_GRIP_WIDTH = 0.028
LIFT_HEIGHT = 0.50
POST_GRASP_WAIT = 0.5
FPS = 60

# Camera params
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_FOV = 60
CAM_NEAR = 0.01
CAM_FAR = 2.0

# =====================================================
# Helpers
# =====================================================
def distance_to_cube(obs):
    return np.linalg.norm(obs[6:9])


def get_ee_camera_image():
    """
    Render RGB image from end-effector frame
    """
    link_state = p.getLinkState(
        env.robot_id,
        env.EE_LINK,
        computeForwardKinematics=True,
        physicsClientId=env.client_id
    )

    ee_pos = np.array(link_state[0])
    ee_orn = np.array(link_state[1])  # quaternion

    # EE forward direction (Z axis)
    rot_mat = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    cam_forward = rot_mat[:, 2]
    cam_up = -rot_mat[:, 1]

    cam_eye = ee_pos + cam_forward * 0.02
    cam_target = cam_eye + cam_forward * 0.10

    view_mat = p.computeViewMatrix(
        cameraEyePosition=cam_eye,
        cameraTargetPosition=cam_target,
        cameraUpVector=cam_up
    )

    proj_mat = p.computeProjectionMatrixFOV(
        fov=CAM_FOV,
        aspect=CAM_WIDTH / CAM_HEIGHT,
        nearVal=CAM_NEAR,
        farVal=CAM_FAR
    )

    _, _, rgb, _, _ = p.getCameraImage(
        CAM_WIDTH,
        CAM_HEIGHT,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=env.client_id
    )

    img = np.reshape(rgb, (CAM_HEIGHT, CAM_WIDTH, 4))[:, :, :3]
    img = img.astype(np.uint8)
    return img


# =====================================================
# Play
# =====================================================
num_episodes = 5

for ep in range(num_episodes):

    obs, info = env.reset()
    done = False
    truncated = False

    print(f"\n▶ Episode {ep + 1}")

    phase = "approach"
    grasp_time = None

    while not (done or truncated):

        action = np.zeros(4, dtype=np.float32)
        dist = distance_to_cube(obs)

        # ----------------------
        # Phase 1: Approach (RL)
        # ----------------------
        if phase == "approach":
            action, _ = model.predict(obs, deterministic=True)
            action[3] = +1.0

            if dist < GRASP_DIST_THRESH:
                phase = "grasp"

        # ----------------------
        # Phase 2: Grasp
        # ----------------------
        elif phase == "grasp":
            delta = TARGET_GRIP_WIDTH - obs[-1]
            action[3] = np.clip(delta / env.grip_scale, -1.0, 1.0)

            if abs(delta) < 0.002:
                print("✅ Gripper closed")
                grasp_time = time.time()
                phase = "wait"

        # ----------------------
        # Phase 3: Wait
        # ----------------------
        elif phase == "wait":
            if time.time() - grasp_time > POST_GRASP_WAIT:
                print("✅ Start LIFT")
                env.ee_target_pos[2] += LIFT_HEIGHT
                phase = "lift"

        # ----------------------
        # Phase 4: Lift
        # ----------------------
        elif phase == "lift":
            action[:] = 0.0

        # ----------------------
        # Step
        # ----------------------
        obs, reward, done, truncated, info = env.step(action)

        # ----------------------
        # EE Camera
        # ----------------------
        img = get_ee_camera_image()
        cv2.imshow("EE Camera", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            truncated = True

        time.sleep(1 / FPS)

    print("⏹ Episode done")

env.close()
cv2.destroyAllWindows()
