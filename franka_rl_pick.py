import os
import time
import csv
import numpy as np
from collections import deque

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_checker import check_env

import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank, seed=0):
    def _init():
        env = FrankaPickEnv(render=False, max_steps=150)
        env.reset(seed=seed + rank)
        return env
    return _init

# =========================================================
# CSV Logger Callback
# =========================================================

class CSVLoggerCallback(BaseCallback):
    """
    Log per-episode statistics to CSV and keep recent stats for tqdm display
    """

    def __init__(self, log_path="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path

        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_dists = []
        self.current_episode_cube_heights = []

        self.episode_count = 0

        # recent stats (for tqdm display)
        self.recent_success = deque(maxlen=50)
        self.recent_mean_dist = deque(maxlen=50)

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "episode",
                "episode_reward",
                "episode_length",
                "mean_dist",
                "max_cube_height",
                "success"
            ])

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_episode_reward += reward
        self.current_episode_length += 1

        if "dist" in info:
            self.current_episode_dists.append(info["dist"])
        if "cube_height" in info:
            self.current_episode_cube_heights.append(info["cube_height"])

        if done:
            self.episode_count += 1

            mean_dist = (
                np.mean(self.current_episode_dists)
                if len(self.current_episode_dists) > 0 else 0.0
            )
            max_cube_height = (
                max(self.current_episode_cube_heights)
                if len(self.current_episode_cube_heights) > 0 else 0.0
            )

            success = 1 if max_cube_height > 0.75 else 0

            self.recent_success.append(success)
            self.recent_mean_dist.append(mean_dist)

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    self.current_episode_reward,
                    self.current_episode_length,
                    mean_dist,
                    max_cube_height,
                    success
                ])

            # reset buffers
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
            self.current_episode_dists = []
            self.current_episode_cube_heights = []

        return True


# =========================================================
# tqdm Progress Callback (reads CSVLogger stats)
# =========================================================

class TqdmProgressCallback(BaseCallback):
    """
    Show training progress with live success_rate and mean_dist
    """

    def __init__(self, total_timesteps, csv_logger, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.csv_logger = csv_logger
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            ncols=120
        )

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True

        self.pbar.n = self.num_timesteps

        if len(self.csv_logger.recent_success) > 0:
            success_rate = np.mean(self.csv_logger.recent_success)
            mean_dist = np.mean(self.csv_logger.recent_mean_dist)
        else:
            success_rate = 0.0
            mean_dist = 0.0

        self.pbar.set_description(
            f"Training | step: {self.num_timesteps:6d} "
            f"| success_rate: {success_rate:.2f} "
            f"| mean_dist: {mean_dist:.2f}"
        )
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()


# =========================================================
# Franka Pick Environment
# =========================================================

class FrankaPickEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, max_steps=150):
        super().__init__()

        self.render = render
        self.max_steps = max_steps

        if p.isConnected():
            p.disconnect()

        self.client_id = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        self.ARM_JOINTS = list(range(7))
        self.FINGER_JOINTS = [9, 10]
        self.EE_LINK = 11

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        self.pos_scale = 0.03
        self.grip_scale = 0.02
        self.workspace_low = np.array([-0.1, -0.3, 0.4], dtype=np.float32)
        self.workspace_high = np.array([0.6, 0.3, 1.0], dtype=np.float32)

        self._setup_world()
        self.fixed_orn = self._get_ee_orn()

    def _setup_world(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.loadURDF("table/table.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)

        cube_x = 0.25 + 0.1 * np.random.uniform(-1, 1)
        cube_y = 0.05 * np.random.uniform(-1, 1)
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            [cube_x, cube_y, 0.65],
            physicsClientId=self.client_id
        )

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [-0.5, 0.0, 0.5],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client_id
        )

        q = [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]
        for j, v in zip(self.ARM_JOINTS, q):
            p.resetJointState(self.robot_id, j, v, physicsClientId=self.client_id)
        for j in self.FINGER_JOINTS:
            p.resetJointState(self.robot_id, j, 0.04, physicsClientId=self.client_id)

        self.ee_target_pos = self._get_ee_pos()
        self.grip_width = 0.04
        self.step_counter = 0

        for _ in range(240):
            p.stepSimulation(physicsClientId=self.client_id)

    def _get_ee_pos(self):
        return np.array(
            p.getLinkState(self.robot_id, self.EE_LINK, True, physicsClientId=self.client_id)[0],
            dtype=np.float32
        )

    def _get_ee_orn(self):
        return p.getLinkState(self.robot_id, self.EE_LINK, True, physicsClientId=self.client_id)[1]

    def _get_cube_pos(self):
        return np.array(
            p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client_id)[0],
            dtype=np.float32
        )

    def _get_obs(self):
        ee = self._get_ee_pos()
        cube = self._get_cube_pos()
        diff = cube - ee
        grip = p.getJointState(self.robot_id, self.FINGER_JOINTS[0], physicsClientId=self.client_id)[0]
        return np.concatenate([ee, cube, diff, [grip]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_world()
        self.fixed_orn = self._get_ee_orn()
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1
        dx, dy, dz, dg = np.clip(action, -1, 1)

        self.ee_target_pos += np.array([dx, dy, dz]) * self.pos_scale
        self.ee_target_pos = np.clip(self.ee_target_pos, self.workspace_low, self.workspace_high)
        self.grip_width = float(np.clip(self.grip_width + dg * self.grip_scale, 0.0, 0.04))

        joints = p.calculateInverseKinematics(
            self.robot_id,
            self.EE_LINK,
            self.ee_target_pos,
            self.fixed_orn,
            physicsClientId=self.client_id
        )

        for j in self.ARM_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                joints[j],
                force=90,
                physicsClientId=self.client_id
            )

        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                self.grip_width,
                force=40,
                physicsClientId=self.client_id
            )

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.render:
                time.sleep(1 / 240)

        ee = self._get_ee_pos()
        cube = self._get_cube_pos()
        dist = float(np.linalg.norm(cube - ee))
        reward = -dist - 0.01

        done = False
        if cube[2] > 0.75 and dist < 0.08:
            reward += 10
            done = True

        truncated = self.step_counter >= self.max_steps
        info = {"dist": dist, "cube_height": cube[2]}

        return self._get_obs(), reward, done, truncated, info

    def close(self):
        p.disconnect(self.client_id)


# =========================================================
# Training
# =========================================================

if __name__ == "__main__":
    mp.freeze_support()

    num_envs = 4   # 👈 先試 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=512,        # 512 * 4 = 2048
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    total_timesteps = 200_000

    csv_logger = CSVLoggerCallback("franka_pick_training.csv")
    tqdm_logger = TqdmProgressCallback(total_timesteps, csv_logger)
    callback = CallbackList([csv_logger, tqdm_logger])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model.save("ppo_franka_pick")
    env.close()
