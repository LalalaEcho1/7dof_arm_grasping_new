import time
import numpy as np
from stable_baselines3 import SAC
from envs.seven_dof_arm import SevenDOFArmEnv

def play(model_path="./models/sac_7dof_final.zip", episodes=5, render_every=1, sleep_time=0.02):
    """
    可视化已训练的 7DOF 机械臂策略。
    Args:
        model_path (str): 模型路径
        episodes (int): 演示的回合数
        render_every (int): 渲染间隔（步数）
        sleep_time (float): 每步之间的等待时间（秒）
    """

    print("🎮 正在加载模型并启动渲染环境...")

    # 创建带渲染的环境
    env = SevenDOFArmEnv(
        render_mode='human',
        model_path='franka/panda.xml',
        render_every=render_every
    )

    # 加载训练好的 SAC 模型
    model = SAC.load(model_path, env=env)
    print("✅ 模型加载完成！开始演示...")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 预测动作（确定性模式）
            action, _ = model.predict(obs, deterministic=True)

            # 平滑动作：防止抖动
            action = np.clip(action, -1.0, 1.0)

            # 进行一步
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # 渲染
            env.render()
            time.sleep(sleep_time)

        print(f"🏁 Episode {ep+1}/{episodes} 完成，总奖励: {total_reward:.3f}")

    env.close()
    print("✅ 所有演示完成，窗口已关闭。")


if __name__ == "__main__":
    play(
        model_path="./models/sac_7dof_final.zip",  # 训练生成的模型路径
        episodes=3,                                # 想看几次
        render_every=1,                            # 每步渲染
        sleep_time=0.02                            # 渲染间隔，调大可以减慢速度
    )
