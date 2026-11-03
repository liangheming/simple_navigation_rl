
import os

import json
import mujoco
import cv2 as cv
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

import xml.etree.ElementTree as ET

class DiffRobotCfg:
    goal_tolerance = 0.4
    distance_coef = 1.0
    collision_coef = 160.0
    goal_reach_coef = 100.0
    collision_tolerance = 0.18
    lin_acc_coef = 0.01
    rot_acc_coef = 0.01
    max_duration = 240
    wheel_radius = 0.03725
    wheel_base_half = 0.11016
    max_wheel_vel = 10.81
    dangerous_margin = 0.32
    dangerous_min_vel = 0.1
    dangerous_coef = 2.0
    forward_coef = 0.5
    wandering_coef = 0.0
    rotation_coef = 0.1

    max_lin_vel = 0.4
    max_ang_vel = 3.14
    param_path = None
    target_min_distance = 1.0
    

class DiffRobotEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(self, xml_path, cfg: DiffRobotCfg, **kwargs):
        utils.EzPickle.__init__(
            self,
            xml_path,
            cfg,
            **kwargs,
        )
        self.cfg = cfg
        MujocoEnv.__init__(self, model_path=xml_path, observation_space=None, **kwargs)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self._init_observation_params()
        self._init_state_params()

        self.target_pos = np.array([3.0, 2.0], dtype=np.float32)
        self.step_count = 0
        self.goal_reach_count = 0
        self.collison_count = 0
    
    def _initialize_simulation(
        self,
    ) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
        with open(self.fullpath,"r") as f:
            xml_str = f.read()
        tree = ET.ElementTree(ET.fromstring(xml_str))
        root = tree.getroot()
        worldbody = root.find("worldbody")
        

        with open(self.cfg.param_path, "r") as f:
            self.obstacle_params = json.load(f)
        cylinder_canidates = self.obstacle_params["cylinder_candidates"]
        cube_candidates = self.obstacle_params["cube_candidates"]
        self.obstacle_default_pose = []
        x_pos = 0.0
        for i , radius in enumerate(cylinder_canidates):
            x_pos+=radius+0.1
            clylinder_half_height = 0.5
            geom_name = f"cylinder_{i}"
            self.obstacle_default_pose.append((geom_name, np.array([x_pos,-4.0,0.5],dtype=np.float32)))
            body_ele = ET.SubElement(worldbody,"body", name=geom_name)
            ET.SubElement(body_ele, "freejoint", name =geom_name)

            ET.SubElement(body_ele, "geom", name=geom_name, type="cylinder", 
                              size=f"{radius} {clylinder_half_height}", rgba="0.1 0.8 0.1 1")
            x_pos+=radius+0.1
        x_pos = 0.0
        for i, size in enumerate(cube_candidates):
            xl,yl = size
            x_pos+=xl*0.5 + 0.1
            cube_half_height = 0.5
            geom_name = f"cube_{i}"
            self.obstacle_default_pose.append((geom_name,np.array([x_pos, 4.0, 0.5],dtype=np.float32)))
            body_ele = ET.SubElement(worldbody,"body", name=geom_name)
            ET.SubElement(body_ele, "freejoint", name =geom_name)
            ET.SubElement(body_ele, "geom", name=geom_name, type="box", 
                              size=f"{xl*0.5} {yl*0.5} {cube_half_height}", rgba="0.1 0.1 0.8 1")
            x_pos+=xl*0.5+0.1
        body_ele = ET.SubElement(worldbody,"body", name="target")
        ET.SubElement(body_ele, "freejoint", name ="target")
        ET.SubElement(body_ele, "geom", name="target", type="cylinder", size=f"0.05 0.05", rgba="0.1 0.1 0.8 0.0")
        ET.SubElement(body_ele, "site", name="target", type="cylinder", size=f"0.2 0.05", rgba="0.8 0.1 0.1 1.0")


        modified_xml_str = ET.tostring(root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(modified_xml_str)
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def _init_observation_params(self):
        sensor_names = [
            "robot_pos",
            "robot_quat",
            "robot_linvel",
            "robot_angvel",
            "left_joint_vel",
            "right_joint_vel",
            "lidar",
        ]
        total_dim = 0
        self.obervation_params = {}
        for name in sensor_names:
            sensor_name = name
            if name == "lidar":
                sensor_name = "lidar_rf_000"
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            data_idx = self.model.sensor_adr[sensor_id]
            data_dim = self.model.sensor_dim[sensor_id]
            if name == "lidar":
                data_dim = 360
            total_dim += data_dim
            self.obervation_params[name] = (int(data_idx), int(data_dim))
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )
        self.prev_state = self.data.sensordata.copy()
        self.current_state = self.prev_state.copy()

    def _init_state_params(self):
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        pos_st_idx = []
        vel_st_idx = []
        self.state_params = {}
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            pos_idx = self.model.jnt_qposadr[joint_id]
            vel_idx = self.model.jnt_dofadr[joint_id]
            pos_st_idx.append(pos_idx)
            vel_st_idx.append(vel_idx)
        pos_st_idx = np.array(pos_st_idx)
        vel_st_idx = np.array(vel_st_idx)
        pos_dim = pos_st_idx[1:] - pos_st_idx[:-1]
        vel_dim = vel_st_idx[1:] - vel_st_idx[:-1]
        total_pos_dim = self.model.nq
        total_vel_dim = self.model.nv
        pos_st_idx = pos_st_idx.tolist()
        vel_st_idx = vel_st_idx.tolist()
        pos_dim = np.append(pos_dim, total_pos_dim - pos_dim.sum()).tolist()
        vel_dim = np.append(vel_dim, total_vel_dim - vel_dim.sum()).tolist()
        for idx, joint_name in enumerate(joint_names):
            self.state_params[joint_name] = {
                "pos": (pos_st_idx[idx], pos_dim[idx]),
                "vel": (vel_st_idx[idx], vel_dim[idx]),
            }
        self.init_state_data_pos = self.data.qpos.copy()
        self.init_state_data_vel = self.data.qvel.copy()

    def _get_state(self):
        ret_data = []
        for _, param in self.obervation_params.items():
            data_idx, data_dim = param
            data = self.data.sensordata[data_idx : data_idx + data_dim].copy()
            ret_data.append(data)
        ret_data = np.concatenate(ret_data).astype(np.float32)
        return ret_data

    def _get_observation(self):
        lidar_data = self._get_lidar(self.current_state)
        lidar_group = lidar_data.reshape(36,-1)
        ground_lidar = lidar_group.min(axis=-1)
        robot_pos = self._get_pos_w(self.current_state)
        direction_expand = np.append(self.target_pos - robot_pos, 0.0)
        direction_body = (
            R.from_quat(self._get_quat_w(self.current_state), scalar_first=True)
            .inv()
            .apply(direction_expand)[:2]
            .astype(np.float32)
        )
        lin_vel_body = self._get_lin_vel_b(self.current_state)
        rot_vel_body = self._get_rot_vel_b(self.current_state)
        observation = np.append(
            np.concat([ground_lidar, direction_body]), [lin_vel_body, rot_vel_body]
        )

        return observation

    def _get_lidar(self, state):
        lidar_idx, lidar_dim = self.obervation_params["lidar"]
        return state[lidar_idx : lidar_idx + lidar_dim]

    def _get_pos_w(self, state):
        pos_idx, pos_dim = self.obervation_params["robot_pos"]
        return state[pos_idx : pos_idx + pos_dim][:2]

    def _get_quat_w(self, state):
        quat_idx, quat_dim = self.obervation_params["robot_quat"]
        return state[quat_idx : quat_idx + quat_dim]

    def _get_lin_vel_w(self, state):
        linvel_idx, linvel_dim = self.obervation_params["robot_linvel"]
        return state[linvel_idx : linvel_idx + linvel_dim]

    def _get_lin_vel_b(self, state):
        robot_vel = self._get_lin_vel_w(state)  # [x,y,z]
        robot_quat = self._get_quat_w(state)  # [w,x,y,z]
        robot_vel_b = (
            R.from_quat(robot_quat, scalar_first=True)
            .inv()
            .apply(robot_vel)
            .astype(np.float32)
        )
        return robot_vel_b[0]

    def _get_rot_vel_b(self, state):
        robot_rot_vel = self._get_rot_vel_w(state)
        robot_quat = self._get_quat_w(state)
        robot_rot_vel_b = (
            R.from_quat(robot_quat, scalar_first=True)
            .inv()
            .apply(robot_rot_vel)
            .astype(np.float32)
        )
        return robot_rot_vel_b[-1]

    def _get_rot_vel_w(self, state):
        rotvel_idx, rotvel_dim = self.obervation_params["robot_angvel"]
        return state[rotvel_idx : rotvel_idx + rotvel_dim]

    def _get_wheel_vel(self, state):
        left_joint_vel_idx, left_joint_vel_dim = self.obervation_params[
            "left_joint_vel"
        ]
        right_joint_vel_idx, right_joint_vel_dim = self.obervation_params[
            "right_joint_vel"
        ]
        left_wheel_vel = state[
            left_joint_vel_idx : left_joint_vel_idx + left_joint_vel_dim
        ]
        right_wheel_vel = state[
            right_joint_vel_idx : right_joint_vel_idx + right_joint_vel_dim
        ]
        return np.array([left_wheel_vel, right_wheel_vel]).flatten()

    def step(self, action):
        self.step_count += 1
        self.prev_state = self._get_state()
        self.action = action
        self._apply_action(action)
        self.current_state = self._get_state()
        reward = self._get_reward()
        observations = self._get_observation()
        done, truncated = self._get_done()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observations, reward, done, truncated, info

    def _apply_action(self, action):
        robot_vel =np.tanh(action) * np.array([self.cfg.max_lin_vel, self.cfg.max_ang_vel],dtype=np.float32)
        wheel_action = self.robot_vel_to_wheel_vel(robot_vel)        
        np.clip(wheel_action, -self.cfg.max_wheel_vel, self.cfg.max_wheel_vel, out=wheel_action)
        self.do_simulation(wheel_action, self.frame_skip)

    def _get_reward(self):
        pre_distance = np.linalg.norm(
            self._get_pos_w(self.prev_state) - self.target_pos
        )
        cur_distance = np.linalg.norm(
            self._get_pos_w(self.current_state) - self.target_pos
        )
        distance_reward = (pre_distance - cur_distance)/(self.cfg.max_wheel_vel * self.cfg.wheel_radius * self.dt)
        goal_reached_reward = 1 if cur_distance < self.cfg.goal_tolerance else 0
        current_lidar = self._get_lidar(self.current_state)
        
        lidar_angle = np.linspace(0, 2 * np.pi, 360)
        valid_mask = current_lidar != -1
        valid_lidar = current_lidar[valid_mask]
        valid_angle = lidar_angle[valid_mask]
        min_idx = np.argmin(valid_lidar)
        min_lidar = valid_lidar[min_idx]
        min_angle = valid_angle[min_idx]
    
        collision_reward = -1 if min_lidar < self.cfg.collision_tolerance else 0

        dangerous_weights = 1.0  if  min_lidar < (self.cfg.dangerous_margin + self.cfg.collision_tolerance) else 0.0
        dangerous_distance = min_lidar - self.cfg.dangerous_margin-self.cfg.collision_tolerance
        angle_weights = np.cos(min_angle)
        lin_cur_vel = self._get_lin_vel_b(self.current_state)
        lin_vel_abs  = np.abs(lin_cur_vel)
        vel_weights = self.cfg.dangerous_min_vel if lin_vel_abs < self.cfg.dangerous_min_vel else lin_vel_abs
        vel_weights = np.sign(lin_cur_vel) * vel_weights
        dangerous_reward = dangerous_weights * dangerous_distance *(angle_weights * vel_weights + 1.0)


        lin_pre_vel = self._get_lin_vel_b(self.prev_state)
        lin_vel_reward = -((lin_pre_vel - lin_cur_vel) / self.dt)**2.0
        rot_pre_vel = self._get_rot_vel_b(self.prev_state)
        rot_cur_vel = self._get_rot_vel_b(self.current_state)
        rot_vel_reward = -((rot_pre_vel - rot_cur_vel) / self.dt)**2.0 * (1-dangerous_weights)
        forward_reward = np.sign(lin_cur_vel) if np.abs(lin_cur_vel) > 0.0001 else 0.0
        wandering_reward = -1.0

        rotate_reaward = - rot_cur_vel ** 2.0 * (1-dangerous_weights)
        


        total_reward = (
            self.cfg.distance_coef * distance_reward
            + self.cfg.goal_reach_coef * goal_reached_reward
            + self.cfg.collision_coef * collision_reward
            + self.cfg.lin_acc_coef * lin_vel_reward
            + self.cfg.rot_acc_coef * rot_vel_reward
            + self.cfg.dangerous_coef * dangerous_reward
            + self.cfg.forward_coef * forward_reward
            + self.cfg.wandering_coef * wandering_reward
            + self.cfg.rotation_coef * rotate_reaward
        )
        if self.render_mode == "human" and self.step_count % 1000==0:
            print("="*20,"static reward","="*20)
            print(f"distance_reward:        \t{distance_reward:.4f}\t\t{self.cfg.distance_coef * distance_reward:.4f}")
            print(f"goal_reached_reward:    \t{goal_reached_reward:.4f}\t\t{self.cfg.goal_reach_coef * goal_reached_reward:.4f}")
            print(f"collision_reward:       \t{collision_reward:.4f}\t\t{self.cfg.collision_coef * collision_reward:.4f}")
            print(f"lin_vel_reward:         \t{lin_vel_reward:.4f}\t\t{self.cfg.lin_acc_coef * lin_vel_reward:.4f}")
            print(f"rot_vel_reward:         \t{rot_vel_reward:.4f}\t\t{self.cfg.rot_acc_coef * rot_vel_reward:.4f}")
            print(f"dangerous_reward:       \t{dangerous_reward:.4f}\t\t{self.cfg.dangerous_coef * dangerous_reward:.4f}")
            print(f"forward_reward:         \t{forward_reward:.4f}\t\t{self.cfg.forward_coef * forward_reward:.4f}")
            print(f"wandering_reward:       \t{wandering_reward:.4f}\t\t{self.cfg.wandering_coef * wandering_reward:.4f}")
            print(f"rotate_reaward:         \t{rotate_reaward:.4f}\t\t{self.cfg.rotation_coef * rotate_reaward:.4f}")
            print("-"*20,"total reward","-"*20)
            print(f"total_reward:           \t{total_reward:.4f}")

        return total_reward

    def _get_done(self):
        epsodie_termination = self.step_count >= (1/self.dt * self.cfg.max_duration)

        current_pos = self._get_pos_w(self.current_state)
        goal_reach = np.linalg.norm(current_pos - self.target_pos) < self.cfg.goal_tolerance
        if goal_reach:
            self.goal_reach_count+=1
       
        current_lidar = self._get_lidar(self.current_state)
        valid_mask = current_lidar != -1
        min_lidar = np.min(current_lidar[valid_mask])
        collision = min_lidar < self.cfg.collision_tolerance
        if collision:
            self.collison_count += 1
        done = goal_reach or collision
        return done, epsodie_termination

    def reset_model(self):
        self.step_count = 0
        default_qpos_data = self.init_state_data_pos.copy()
        default_qvel_data = self.init_state_data_vel.copy()
        self._reset_obstacles(default_qpos_data,default_qvel_data)
        self._random_obstacle_pos(default_qpos_data, default_qvel_data)
        self.set_state(default_qpos_data, default_qvel_data)
        self.current_state = self._get_state()
        return self._get_observation()
    
    def _reset_obstacles(self,qpos_data, qvel_data):
        for name ,pos in self.obstacle_default_pose:
            self._set_geom_pos(qpos_data, qvel_data,name, pos)
    
    def _random_obstacle_pos(self,qpos_data, qvel_data):
       
        maps_canidates = self.obstacle_params['obstacle_maps']
        maps_num = np.random.choice(len(maps_canidates))
        map_params = maps_canidates[maps_num]
        for obs in map_params['obstacles']:
            type_str = obs['type']
            idx = obs['idx']
            x,y = obs['pos']
            if type_str == 'cube':
                geom_name = f'cube_{idx}'
            elif type_str == 'cylinder':
                geom_name = f'cylinder_{idx}'
            else:
                raise Exception(f'{type_str} is not supported')
            self._set_geom_pos(qpos_data, qvel_data,geom_name, np.array([x,y,0.5],dtype=np.float32))
        
        map_dir = os.path.dirname(self.cfg.param_path)
        image_path = os.path.join(map_dir,map_params['map'])
        map_resolution = self.obstacle_params["resolution"]
        robot_radius = self.obstacle_params["robot_radius"]
        area_size = self.obstacle_params["area_size"]
        safe_grid = int(robot_radius / map_resolution) * 4 + 1
        
        distance_grid = int(self.cfg.target_min_distance / map_resolution) + 1
        map_image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (safe_grid, safe_grid)
        )
        dilated_image = cv.dilate(map_image, kernel, iterations=1)
        xs, ys = np.nonzero(np.bitwise_not(dilated_image))
        random_idx = np.random.choice(len(xs))
        ogx, ogy = xs[random_idx], ys[random_idx]
        dilated_image = cv.circle(
            dilated_image, (ogy, ogx), distance_grid, 255, -1
        )

        sx, sy = (ogx + 0.5) * map_resolution - area_size[0] * 0.5, (
            ogy + 0.5
        ) * map_resolution - area_size[1] * 0.5

        xs, ys = np.nonzero(np.bitwise_not(dilated_image))
        random_idx = np.random.choice(len(xs))
        tgx, tgy = xs[random_idx], ys[random_idx]
        tx, ty = (tgx + 0.5) * map_resolution - area_size[0] * 0.5, (
            tgy + 0.5
        ) * map_resolution - area_size[1] * 0.5

        joint_param = self.state_params["robot_base"]
        p_st,_ = joint_param['pos']
        qpos_data[p_st:p_st+2] = np.array([sx,sy],dtype=np.float32)
        random_yaw_deg = np.random.uniform(0, 360)
        rotation = R.from_euler('z', random_yaw_deg, degrees=True)
        quat_wxyz = rotation.as_quat(scalar_first=True)
        qpos_data[p_st+3:p_st+7] = quat_wxyz.astype(np.float32)


        self.target_pos = np.array([tx, ty], dtype=np.float32)
        self._set_geom_pos(qpos_data, qvel_data,"target",np.array([tx,ty,0.05],dtype=np.float32))
          


    def _set_geom_pos(self,qpos_data, qvel_data,  geom_name, pos):
        joint_param = self.state_params[geom_name]
        p_st,_ = joint_param['pos']
        v_st,v_dim = joint_param['vel']
        qpos_data[p_st:p_st+3] = pos
        qvel_data[v_st:v_st+v_dim] = 0.0
        

    def robot_vel_to_wheel_vel(self, robot_vel):
        robot_lin_vel = robot_vel[0]
        robot_ang_vel = robot_vel[1]
        wheel_radius_inv = 1.0 / self.cfg.wheel_radius
        wheel_left_vel = wheel_radius_inv * (
            robot_lin_vel - robot_ang_vel * self.cfg.wheel_base_half
        )
        wheel_right_vel = wheel_radius_inv * (
            robot_lin_vel + robot_ang_vel * self.cfg.wheel_base_half
        )
        wheel_vel = np.array([wheel_left_vel, wheel_right_vel],dtype=np.float32)
        return wheel_vel
    def _get_info(self):
        return {"goal_reach_count":self.goal_reach_count,"collision_count":self.collison_count}
    
    def _get_reset_info(self):
        return {"goal_reach_count":self.goal_reach_count,"collision_count":self.collison_count}