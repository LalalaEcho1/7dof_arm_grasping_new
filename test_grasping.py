#!/usr/bin/env python3
"""
测试抓取环境修复效果
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.seven_dof_arm import SevenDOFArmEnv

def test_grasping_environment():
    """测试抓取环境"""
    print("🔍 测试抓取环境修复效果...")
    
    env = SevenDOFArmEnv(render_mode=None)
    
    # 测试多个episode
    success_count = 0
    total_episodes = 10
    
    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_success = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(100):  # 限制步数
            # 使用随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # 检查是否成功
            if info.get('is_success', False):
                episode_success = True
                success_count += 1
                print(f"✅ Episode {episode + 1} 成功！步数: {step + 1}, 总奖励: {episode_reward:.2f}")
                break
            
            # 每20步打印一次进度
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.2f}, distance={info.get('distance', 0):.3f}, "
                      f"height_gain={info.get('height_gain', 0):.3f}")
        
        if not episode_success:
            print(f"❌ Episode {episode + 1} 失败，总奖励: {episode_reward:.2f}")
    
    success_rate = success_count / total_episodes
    print(f"\n📊 测试结果:")
    print(f"  成功次数: {success_count}/{total_episodes}")
    print(f"  成功率: {success_rate:.1%}")
    
    if success_rate > 0:
        print("🎉 抓取率不再是0！修复成功！")
    else:
        print("⚠️ 抓取率仍为0，需要进一步调试")
    
    env.close()
    return success_rate

if __name__ == "__main__":
    test_grasping_environment()
