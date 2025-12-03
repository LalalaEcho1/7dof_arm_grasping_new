from envs.seven_dof_arm import SevenDOFArmEnv
import numpy as np
import matplotlib.pyplot as plt

# ====== 初始化环境 ======
env = SevenDOFArmEnv()
obs, info = env.reset()

print("==== 开始调试 ====")

qpos_history = []
fingertip_history = []
object_history = []

# ====== 采样 N 步 ======
N = 50
for step in range(N):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)

    qpos = env.data.qpos.copy()
    fingertip_pos = env.data.site_xpos[env.end_effector_id].copy()
    object_pos = env.data.xpos[env.target_body_id].copy()

    qpos_history.append(qpos)
    fingertip_history.append(fingertip_pos)
    object_history.append(object_pos)

    print(f"\nStep {step+1}")
    print(f"  Arm joints: {qpos[:7]}")
    print(f"  Gripper:    {qpos[7:]}")
    print(f"  Fingertip pos: {fingertip_pos}")
    print(f"  Object pos:    {object_pos}")
    print("-" * 60)

env.close()

# ====== 转换为数组 ======
qpos_arr = np.array(qpos_history)       # [N, nq]
fing_arr = np.array(fingertip_history)  # [N, 3]
obj_arr = np.array(object_history)      # [N, 3]

# ====== 自动分析 ======
arm_var = np.var(qpos_arr[:, :7])
grip_var = np.var(qpos_arr[:, 7:])
fing_var = np.var(fing_arr, axis=0).sum()
obj_var = np.var(obj_arr, axis=0).sum()

print("\n==== 自动分析结果 ====")
if arm_var < 1e-6 and grip_var > 1e-6:
    print("⚠️ 机械臂关节没有动，只有夹爪在动 → 动作可能没传到 7 DOF。")
elif arm_var > 1e-6 and fing_var < 1e-6:
    print("⚠️ 机械臂关节在动，但末端位置不变 → end_effector site 可能绑错了。")
elif arm_var > 1e-6 and fing_var > 1e-6 and obj_var < 1e-6:
    print("⚠️ 机械臂和末端都在动，但物体位置没变 → 物体可能被固定死（<weld>）。")
elif obj_var > 1e-6:
    print("✅ 物体位置有变化，环境交互正常。")
else:
    print("⚠️ 没检测到明显变化，需要进一步检查。")

# ====== 可视化 ======
t = np.arange(N)

# 关节
plt.figure(figsize=(12,6))
for i in range(min(7, qpos_arr.shape[1])):
    plt.plot(t, qpos_arr[:, i], label=f"joint{i+1}")
plt.title("Arm Joint Angles over Time")
plt.xlabel("Step")
plt.ylabel("qpos")
plt.legend()
plt.grid(True)

# Fingertip
plt.figure(figsize=(8,6))
plt.plot(t, fing_arr[:,0], label="x")
plt.plot(t, fing_arr[:,1], label="y")
plt.plot(t, fing_arr[:,2], label="z")
plt.title("Fingertip Position over Time")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

# Object
plt.figure(figsize=(8,6))
plt.plot(t, obj_arr[:,0], label="x")
plt.plot(t, obj_arr[:,1], label="y")
plt.plot(t, obj_arr[:,2], label="z")
plt.title("Object Position over Time")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

plt.show()
