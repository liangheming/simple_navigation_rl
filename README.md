# RL导航
这是一个基于RL完成移动机器人AB点导航的项目，机器人驱动方式为两轮差速，传感器数据主要是单线激光雷达，仿真器为mujoco，主要面向仿真开发（不包含sim2real相关的参数辨识、域随机化等技巧），偏向于基础的场景构建和RL训练，旨在了解RL应用于机器人导航的算法实现.
## 环境安装
1. 安装uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. 初始化环境
```shell
uv venv env_mujoco --python=python3.11
source env_mujoco/bin/activate
```
3. 安装依赖项目
```shell
uv pip install mujoco
uv pip install opencv-python
uv pip install gymnasium
uv pip install imageio
uv pip install scipy
uv pip install stable-baselines3[extra]
```
## 脚本
0. 直接试玩
```shell
python debug_play.py --file assets/parameters.json --checkpoint assets/checkpoint.zip
```
1. 生成地图
```shell
python obstacle_generation.py --num 2000 --num_obstacles 10 --save_images output_dir {data_dir}
```
- num: 创建地图的点数
- num_obstacles: 创建地图的障碍物数量
- resolution: 地图的分辨率
- robot_radius: 机器人的半径
- margin_grid: 障碍物之间的间隔
- area_size: 地图的大小
- output_dir: 输出图片的目录
- save_images: 保存障碍物的栅格地图
- sparse: 稀疏障碍物模式：即障碍物之间至少可以隔着一个机器人的距离

2. 启动训练
```shell
python debug_train.py --file {json_file}
```
- cpu: 使用CPU进行训练
- file: 第一步生成的json文件地址
- show: 是否可视化训练过程
- render_lidar: 是否可视化激光雷达数据【需要在可视化训练过程中才打开】
- num_envs: 多环境训练的数量
3. 启动推理
```shell
python debug_play.py --file {json_file} --checkpoint {checkpoint_zip_file}
```

## 关于定义 reward
```python
def _get_reward(self):
    # 这里定义奖励函数
    return 0.0
```
## 最后
**have fun !!!**