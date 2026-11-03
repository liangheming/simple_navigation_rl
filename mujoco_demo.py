import mujoco
import mujoco.viewer



def main():
    model = mujoco.MjModel.from_xml_path("models/navigation.xml")
    data = mujoco.MjData(model)
    print(model.nq,model.nv)
    print(data.qpos)
    # pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot_pos")
    # quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot_quat")
    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     while viewer.is_running():
    #         mujoco.mj_step(model, data)
    #         ctrl_data = data.ctrl
    #         sensor_data = data.sensordata
    #         print(len(sensor_data))
            



if __name__ == "__main__":
    main()