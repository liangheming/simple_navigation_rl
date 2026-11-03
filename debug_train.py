import os
import mujoco
import datetime
import argparse
from envs import DiffRobotCfg
from networks import ScanFeatureExtractor, CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        self.model.dump_logs()

    def _on_rollout_end(self):
        rollout_buffer = self.locals["rollout_buffer"]
        mean_reward = rollout_buffer.rewards.mean()
        self.logger.record("mean_reward", f"{mean_reward:.4f}")


def train(args):
    cfg = DiffRobotCfg()
    cfg.param_path = args.file

    if args.num_envs >1:
        args.show = False
        args.render_lidar = False
    env_params = {
        "xml_path":"./models/navigation.xml",
        "cfg": cfg,
        "visual_options":{mujoco.mjtVisFlag.mjVIS_RANGEFINDER: True if args.render_lidar else False},
        "width":1280,
        "height":960,
        "frame_skip":2,
        "camera_name":"main_camera",
        "render_mode":"human" if args.show else "rgb_array"
    }
    env =  make_vec_env("gymnasium_env/DiffRobotEnv-v0", n_envs=args.num_envs, vec_env_cls=SubprocVecEnv,env_kwargs=env_params)



    policy_kwargs = dict(
        features_extractor_class=ScanFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    agent = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device="cuda:0" if not args.cpu else "cpu",
        n_epochs=10,
        n_steps=16384,
        batch_size=128,
        ent_coef=0.05,
        vf_coef=1.0,
        use_sde=False,
    )
    logger_dir = os.path.join(
        os.path.dirname(__file__),
        datetime.datetime.now().strftime("task_log-%Y-%m-%d-%H-%M-%S-%f"),
    )
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    logger = configure(folder=logger_dir, format_strings=["stdout", "log"])
    agent.set_logger(logger)

    agent.learn(
        total_timesteps=1e7,
        callback=[
            CheckpointCallback(
                save_freq=16384 * 2, name_prefix="rl_model", save_path=logger_dir
            ),
            CustomCallback(verbose=1),
        ],
        progress_bar=True,
    )
    agent.save(os.path.join(logger_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff Robot RL")
    parser.add_argument(
        "--file", type=str, required=True, help="path of map params json file"
    )
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("--num_envs", type=int,default=1, help="number of enviroments")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--render_lidar", default=False, action="store_true")
    args = parser.parse_args()
    train(args)
