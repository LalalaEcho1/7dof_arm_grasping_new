#!/usr/bin/env python3
"""
测试模型可视化脚本
用于在单进程环境中验证模型是否正确加载和渲染
"""
import numpy as np
from envs.seven_dof_arm import SevenDOFArmEnv

def test_model_visualization(num_steps=100):
    """测试模型可视化"""
    print("=" * 60)
    print("模型可视化测试")
    print("=" * 60)
    
    # 创建环境（单进程，启用渲染）
    print("\n1. 创建环境...")
    env = SevenDOFArmEnv(render_mode='human', render_every=1, is_training=False)
    print("✅ 环境创建成功")
    
    # Reset
    print("\n2. 重置环境...")
    obs, info = env.reset()
    print(f"✅ Reset成功")
    print(f"   观测空间: {obs.shape}")
    print(f"   动作空间: {env.action_space}")
    
    # 检查关键组件
    print("\n3. 检查关键组件:")
    print(f"   末端执行器ID: {env.end_effector_id}")
    print(f"   目标物体ID: {env.target_body_id}")
    print(f"   目标几何体ID: {env.target_geom_id}")
    print(f"   左夹爪ID: {env.left_finger_body_id}")
    print(f"   右夹爪ID: {env.right_finger_body_id}")
    
    # 检查几何体
    print("\n4. 检查几何体配置:")
    visual_count = sum(1 for i in range(env.model.ngeom) if env.model.geom_group[i] == 2)
    collision_count = sum(1 for i in range(env.model.ngeom) if env.model.geom_group[i] == 3)
    print(f"   Visual几何体 (group=2): {visual_count}")
    print(f"   Collision几何体 (group=3): {collision_count}")
    print(f"   总几何体: {env.model.ngeom}")
    
    if visual_count >= 10:
        print("   ✅ Visual几何体配置正确，应该能看到完整的mesh模型")
    else:
        print("   ⚠️  Visual几何体数量不足，可能只显示简化模型")
    
    # 执行动作并观察
    print(f"\n5. 执行动作（观察渲染窗口）...")
    print("   提示：")
    print("   - 如果看到球体而不是完整模型，请按 'G' 键切换group显示")
    print("   - 渲染窗口会持续运行，观察机器人运动")
    print("   - 按 Enter 键结束测试")
    
    import time
    
    # 等待一下，确保viewer完全启动
    time.sleep(0.5)
    
    print("\n   渲染窗口已打开，正在运行仿真...")
    print("   观察窗口中的机器人运动...")
    print("   按 Enter 键结束...")
    
    step = 0
    try:
        # 运行仿真循环
        while True:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 50 == 0:
                eef_pos = env.data.site_xpos[env.end_effector_id] if env.end_effector_id >= 0 else np.zeros(3)
                obj_pos = env.data.xpos[env.target_body_id] if env.target_body_id >= 0 else np.zeros(3)
                distance = np.linalg.norm(eef_pos - obj_pos) if env.target_body_id >= 0 else 0
                print(f"   Step {step}: reward={reward:.2f}, distance={distance:.3f}m")
            
            if terminated or truncated:
                print(f"\n   Episode在第 {step+1} 步结束，重置环境...")
                obs, info = env.reset()
            
            step += 1
            time.sleep(0.02)  # 控制仿真速度，约50fps
            
            # 检查是否达到最大步数
            if step >= num_steps:
                print(f"\n   已达到最大步数 {num_steps}，继续运行...")
                print("   按 Ctrl+C 结束...")
                # 继续运行，直到用户中断
                while True:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                    time.sleep(0.02)
                    
    except KeyboardInterrupt:
        print("\n\n   用户中断")
    
    print("\n6. 关闭环境...")
    env.close()
    print("✅ 测试完成！")
    print("\n" + "=" * 60)
    print("如果看到:")
    print("  ✅ 完整的机器人mesh模型 → 模型配置正确")
    print("  ❌ 球体或简化模型 → 请按 'G' 键切换group显示")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试模型可视化")
    parser.add_argument("--steps", type=int, default=100, help="测试步数（默认100）")
    args = parser.parse_args()
    
    test_model_visualization(num_steps=args.steps)

