import cv2 as cv
import mujoco
import gymnasium as gym
from envs import DiffRobotCfg
import numpy as np



def main():
    cfg = DiffRobotCfg()
    cfg.param_path = "/home/lion/data/curriculum/20260530_171939/parameters.json"
    env = gym.make(
        "gymnasium_env/DiffRobotEnv-v0",
        xml_path="./models/navigation.xml",
        cfg = cfg,
        visual_options={mujoco.mjtVisFlag.mjVIS_RANGEFINDER: False},
        width=1280,
        height=960,
        frame_skip=2,
        camera_name="main_camera",
        render_mode="human",
    )
    env.reset()
    samples = np.array([0,-3.14], dtype=np.float32)
    for i in range(1000):
        observation, reward, terminated, truncated, info = env.step(samples)
        if i %10 ==0:
            observation, info = env.reset()
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
