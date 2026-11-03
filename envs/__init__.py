import gymnasium as gym
from .diff_robot_env import DiffRobotEnv,DiffRobotCfg
gym.register(
    id="gymnasium_env/DiffRobotEnv-v0",
    entry_point=DiffRobotEnv
)
