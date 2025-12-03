import time
import os
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 解决画面卡死/不动的问题：强制使用 EGL 或 OSMesa (如果显卡驱动有问题)
# 或者尝试直接在终端运行 export MUJOCO_GL=glfw
# 这里我们在代码里强制指定后端，试试能不能救回来
os.environ["MUJOCO_GL"] = "glfw"

from envs.seven_dof_arm import SevenDOFArmEnv


def evaluate(model, env, num_episodes=5):
    print(f"🚀 开始测试... (目标: 抓取并抬起)")

    for ep in range(num_episodes):
        obs = env.reset()

        # 🌟 关键修改：强制将难度设为最高级 (Lift)
        # 因为 env 被 DummyVecEnv 包裹了，所以要用 env.envs[0] 访问原始环境
        env.envs[0].success_mode = "lift"

        episode_reward = 0
        done = False
        step_count = 0

        print(f"\n--- Episode {ep + 1} ---")
        print(f"当前考核标准: {env.envs[0].success_mode}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # 尝试渲染
            try:
                env.envs[0].render()
            except Exception as e:
                print(f"渲染出错: {e}")

            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            episode_reward += reward
            step_count += 1

            # 打印每一步的高度，确认它是否真的在动
            # info['object_height'] 是我们在环境里记录的
            if step_count % 10 == 0:
                h = info.get('object_height', 0.0)
                print(f"Step {step_count}: 物体高度 = {h:.4f} m")

            time.sleep(0.02)

            if done:
                status = "✅ 成功抬起!" if info.get('is_success', False) else "❌ 失败"
                print(f"结果: {status} | 总分: {episode_reward:.2f} | 耗时: {step_count}步")
                break


def test(model_path):
    # 1. 创建环境
    base_env = SevenDOFArmEnv(
        render_mode='human',
        model_path="franka/panda.xml",
        render_every=1,
        max_episode_steps=200
    )
    env = DummyVecEnv([lambda: base_env])

    # 2. 加载归一化
    stats_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(stats_path):
        print(f"✅ 加载归一化统计")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    # 3. 加载模型
    # 去掉可能的后缀
    if model_path.endswith(".zip"): model_path = model_path[:-4]

    model = SAC.load(model_path, env=env)

    try:
        evaluate(model, env)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/sac_7dof_final", help="模型路径")
    args = parser.parse_args()
    test(args.model)
