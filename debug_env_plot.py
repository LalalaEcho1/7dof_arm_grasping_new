import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.seven_dof_arm import SevenDOFArmEnv

# 设置matplotlib后端，避免显示问题
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

def debug_run(steps=300):
    try:
        env = SevenDOFArmEnv(render_mode="human")
        obs, _ = env.reset()
        total_reward = 0.0

        # 数据缓存
        rewards, distances, gaps, heights = [], [], [], []

        plt.ion()
        fig, axes = plt.subplots(4, 1, figsize=(6, 10))

        line1, = axes[0].plot([], [], label="Reward")
        line2, = axes[1].plot([], [], label="EEF to Object Distance")
        line3, = axes[2].plot([], [], label="Fingertip Gap")
        line4, = axes[3].plot([], [], label="Object Height")

        axes[0].set_ylabel("Reward")
        axes[1].set_ylabel("Distance (m)")
        axes[2].set_ylabel("Gap (m)")
        axes[3].set_ylabel("Height (m)")
        axes[3].set_xlabel("Step")

        for ax in axes:
            ax.legend()

        for i in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            rewards.append(reward)
            distances.append(info.get("distance", np.nan))
            gaps.append(info.get("fingertip_gap", np.nan))
            heights.append(info.get("object_height", np.nan))

            # 每10步打印一次，减少输出
            if i % 10 == 0 or i < 5:
                print(
                    f"[Step {i}] "
                    f"Reward={reward:.2f}, "
                    f"EEF→Obj={(info.get('distance') or 0):.3f}m, "
                    f"FingerGap={(info.get('fingertip_gap') or 0):.3f}m, "
                    f"ObjHeight={(info.get('object_height') or 0):.3f}m, "
                    f"Success={info.get('is_success', False)}"
                )
            
            # 调试信息：检查site ID
            if i == 0:
                print(f"调试信息:")
                print(f"  left_tip_site_id: {env.left_tip_site_id}")
                print(f"  right_tip_site_id: {env.right_tip_site_id}")
                print(f"  end_effector_id: {env.end_effector_id}")
                if env.left_tip_site_id >= 0 and env.right_tip_site_id >= 0:
                    left_tip = env.data.site_xpos[env.left_tip_site_id].copy()
                    right_tip = env.data.site_xpos[env.right_tip_site_id].copy()
                    print(f"  left_tip位置: {left_tip}")
                    print(f"  right_tip位置: {right_tip}")
                    print(f"  实际计算间隙: {np.linalg.norm(left_tip - right_tip):.6f}")
                else:
                    print("  ⚠️ Site ID无效，使用默认值0.05")

            # 更新曲线
            line1.set_data(range(len(rewards)), rewards)
            line2.set_data(range(len(distances)), distances)
            line3.set_data(range(len(gaps)), gaps)
            line4.set_data(range(len(heights)), heights)

            for ax, data in zip(axes, [rewards, distances, gaps, heights]):
                ax.relim()
                ax.autoscale_view()

            plt.pause(0.01)

            if terminated or truncated:
                print("Episode finished. Resetting...\n")
                obs, _ = env.reset()
                total_reward = 0.0

    except Exception as e:
        print(f"发生错误: {e}")
        print("尝试使用非渲染模式...")
        try:
            env = SevenDOFArmEnv(render_mode=None)
            obs, _ = env.reset()
            total_reward = 0.0
            
            # 数据缓存
            rewards, distances, gaps, heights = [], [], [], []

            plt.ion()
            fig, axes = plt.subplots(4, 1, figsize=(6, 10))

            line1, = axes[0].plot([], [], label="Reward")
            line2, = axes[1].plot([], [], label="EEF to Object Distance")
            line3, = axes[2].plot([], [], label="Fingertip Gap")
            line4, = axes[3].plot([], [], label="Object Height")

            axes[0].set_ylabel("Reward")
            axes[1].set_ylabel("Distance (m)")
            axes[2].set_ylabel("Gap (m)")
            axes[3].set_ylabel("Height (m)")
            axes[3].set_xlabel("Step")

            for ax in axes:
                ax.legend()

            for i in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                rewards.append(reward)
                distances.append(info.get("distance", np.nan))
                gaps.append(info.get("fingertip_gap", np.nan))
                heights.append(info.get("object_height", np.nan))

                # 每20步打印一次，进一步减少输出
                if i % 20 == 0 or i < 3:
                    print(
                        f"[Step {i}] "
                        f"Reward={reward:.2f}, "
                        f"EEF→Obj={(info.get('distance') or 0):.3f}m, "
                        f"FingerGap={(info.get('fingertip_gap') or 0):.3f}m, "
                        f"ObjHeight={(info.get('object_height') or 0):.3f}m, "
                        f"Success={info.get('is_success', False)}"
                    )

                # 更新曲线
                line1.set_data(range(len(rewards)), rewards)
                line2.set_data(range(len(distances)), distances)
                line3.set_data(range(len(gaps)), gaps)
                line4.set_data(range(len(heights)), heights)

                for ax, data in zip(axes, [rewards, distances, gaps, heights]):
                    ax.relim()
                    ax.autoscale_view()

                plt.pause(0.01)

                if terminated or truncated:
                    print("Episode finished. Resetting...\n")
                    obs, _ = env.reset()
                    total_reward = 0.0
                    
        except Exception as e2:
            print(f"仍然出错: {e2}")
    finally:
        try:
            env.close()
        except:
            pass
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    debug_run(steps=100)  # 运行100步来查看完整的数据变化趋势