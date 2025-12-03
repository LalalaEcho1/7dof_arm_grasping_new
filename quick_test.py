#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境是否正常工作
"""

import numpy as np
from envs.seven_dof_arm import SevenDOFArmEnv

def test_environment():
    """测试环境基本功能"""
    print("开始测试环境...")
    
    try:
        # 创建环境
        env = SevenDOFArmEnv(render_mode=None)
        print("✅ 环境创建成功")
        
        # 测试reset
        obs, info = env.reset()
        print(f"✅ Reset成功，观察空间维度: {obs.shape}")
        print(f"✅ 动作空间维度: {env.action_space.shape}")
        
        # 测试随机动作
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"步骤 {i+1}: 奖励={reward:.3f}, 成功={info.get('is_success', False)}")
            
            if terminated or truncated:
                print("Episode结束")
                break
        
        env.close()
        print("✅ 环境测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("\n🎉 所有测试通过！代码修复成功。")
    else:
        print("\n💥 测试失败，需要进一步调试。")
