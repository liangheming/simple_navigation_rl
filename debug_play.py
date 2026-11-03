import torch
import mujoco
import argparse
from envs import DiffRobotCfg
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


def play(args):
    cfg = DiffRobotCfg()
    cfg.param_path = args.file
    env_params = {
        "xml_path":"./models/navigation.xml",
        "cfg": cfg,
        "visual_options":{mujoco.mjtVisFlag.mjVIS_RANGEFINDER: False},
        "width":1280,
        "height":960,
        "frame_skip":2,
        "camera_name":"main_camera",
        "render_mode":"human"
    }
    env =  make_vec_env("gymnasium_env/DiffRobotEnv-v0", n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_params)
    agent = PPO.load(args.checkpoint, env=env,print_system_info=True)
    obs = env.reset()
    while True:
        # 捕获键盘中断
        try:
            with torch.inference_mode():
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()
        except KeyboardInterrupt:
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff Robot RL")
    parser.add_argument(
        "--file", type=str, required=True, help="path of map params json file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint"
    )
    args = parser.parse_args()
    play(args)
