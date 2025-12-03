import numpy as np
import matplotlib.pyplot as plt
from envs.seven_dof_arm import SevenDOFArmEnv

# ========== 初始化环境 ==========
env = SevenDOFArmEnv(render_mode=None)   # 不渲染，只收集数据
obs, info = env.reset()

# 存储历史
qpos_history = []
fingertip_history = []
object_history = []
reward_history = []

N = 200   # 采样步数
for step in range(N):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)

    qpos = env.data.qpos.copy()
    fingertip_pos = env.data.site_xpos[env.end_effector_id].copy()
    object_pos = env.data.xpos[env.target_body_id].copy()

    qpos_history.append(qpos)
    fingertip_history.append(fingertip_pos)
    object_history.append(object_pos)
    reward_history.append(reward)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# ========== 转 numpy ==========
qpos_arr = np.array(qpos_history)        # [N, nq]
fing_arr = np.array(fingertip_history)   # [N, 3]
obj_arr = np.array(object_history)       # [N, 3]
reward_arr = np.array(reward_history)

t = np.arange(len(reward_arr))

np.savez("run_log.npz",
         qpos=qpos_arr,
         fingertip=fing_arr,
         object=obj_arr,
         reward=reward_arr)
print("已保存到 run_log.npz")

# ========== 绘图 ==========
plt.figure(figsize=(12,6))
for i in range(min(7, qpos_arr.shape[1])):
    plt.plot(t, qpos_arr[:, i], label=f"joint{i+1}")
plt.title("Arm Joint Angles")
plt.xlabel("Step")
plt.ylabel("qpos")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(t, fing_arr[:,0], label="x")
plt.plot(t, fing_arr[:,1], label="y")
plt.plot(t, fing_arr[:,2], label="z")
plt.title("End Effector Position")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(t, obj_arr[:,0], label="x")
plt.plot(t, obj_arr[:,1], label="y")
plt.plot(t, obj_arr[:,2], label="z")
plt.title("Object Position")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8,6))
plt.plot(t, reward_arr, label="reward", color="purple")
plt.title("Reward per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

plt.show()
