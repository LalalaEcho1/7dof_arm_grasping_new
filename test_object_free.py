import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# 选择模型路径（这里改成你的实际 XML 文件）
xml_path = os.path.join(os.path.dirname(__file__), "franka/panda.xml")
# xml_path = os.path.join(os.path.dirname(__file__), "7dof_arm.xml")

# 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Number of actuators:", model.nu)

# 自动获取 gripper actuator 索引
def get_gripper_actuator_indices(model):
    indices = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"Actuator {i}: {name}")  # 打印所有 actuator 名称，方便调试
        if name is not None and "grip" in name.lower():
            indices.append(i)
    return indices

gripper_ids = get_gripper_actuator_indices(model)
print("Gripper actuator indices:", gripper_ids)

# 用 viewer 可视化
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    print("按 ESC 退出可视化窗口。")

    while viewer.is_running():
        # 模拟一步
        mujoco.mj_step(model, data)
        step += 1

        # 每 100 步打印目标物体位置
        if step % 100 == 0:
            if "target_object" in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                                   for i in range(model.nbody)]:
                target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
                pos = data.xpos[target_id]
                print(f"step={step}, target_object pos={pos}")

        # 控制 gripper：在 500 步以后慢慢闭合
        if 500 < step < 800:
            for gid in gripper_ids:
                data.ctrl[gid] = 0.03  # 打开
        elif step >= 800:
            for gid in gripper_ids:
                data.ctrl[gid] = 0.0   # 闭合

        # 同步渲染
        viewer.sync()

        time.sleep(0.002)
